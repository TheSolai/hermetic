//! The planner: what has to happen before a model can be served.
//!
//! The terminal asks a model a question, and the answer is never simply
//! "start the server". It might be: convert these weights, build an adapter
//! for that shape, fold it in, and only then start anything. The planner
//! works that out, so that the terminal only has to carry the steps out and
//! show the user what is happening.
//!
//! The interesting decisions are:
//!
//!   * Which runtime should serve this? A GGUF can be served by llama.cpp;
//!     an MLX model by MLX; a model a server already holds by that server.
//!   * Can the weights be changed? If they can, an adapter is built or
//!     found. If they cannot, the model is served with a prompt-level
//!     policy and the terminal says so.

use crate::policy::Policy;
use crate::runtime::{Modification, ServeMode};
use crate::signature::Signature;
use crate::source::{ModelDescriptor, ModelSource};

/// One thing that has to happen first.
#[derive(Debug, Clone)]
pub enum Step {
    /// Turn the weights into a GGUF so that llama.cpp can serve them.
    ConvertToGguf {
        /// Where they are now.
        source: ModelSource,
    },
    /// Download the original weights from the hub.
    Download {
        /// The repository.
        repo: String,
    },
    /// Build a Heretic adapter for this shape.
    BuildAdapter {
        /// The repository whose weights to abliterate.
        donor: String,
        /// The shape the adapter must have.
        signature: Signature,
    },
    /// Fold an adapter into a set of weights, for a runtime that cannot bind
    /// one.
    MergeAdapter {
        /// The repository whose weights to fold it into.
        donor: String,
        /// The shape.
        signature: Signature,
    },
}

/// How a model should be served.
#[derive(Debug, Clone)]
pub struct Plan {
    /// The runtime that should serve it.
    pub runtime: String,
    /// How the adapter will be applied, if it is.
    pub mode: ServeMode,
    /// What has to happen first.
    pub steps: Vec<Step>,
    /// Why this plan, in a sentence the user can read.
    pub reason: String,
}

impl Plan {
    /// A plan with nothing to do first.
    pub fn ready(runtime: &str, mode: ServeMode, reason: &str) -> Plan {
        Plan {
            runtime: runtime.to_string(),
            mode,
            steps: Vec::new(),
            reason: reason.to_string(),
        }
    }

    /// Does this plan need work before the model can be served?
    pub fn needs_work(&self) -> bool {
        !self.steps.is_empty()
    }
}

/// Work out how to serve a model.
///
/// The runtimes are given in the order they should be preferred.
pub fn plan(
    descriptor: &ModelDescriptor,
    runtimes: &[&dyn crate::runtime::ModelRuntime],
    can_modify_weights: bool,
) -> Plan {
    let source = &descriptor.source;

    // A model a server already holds: that server serves it, and there is
    // nothing to change.
    if let ModelSource::Remote { base_url, model } = source {
        return Plan {
            runtime: "openai".to_string(),
            mode: if can_modify_weights {
                ServeMode::WeightModified
            } else {
                ServeMode::PromptPolicy
            },
            steps: Vec::new(),
            reason: format!("{model} is already being served at {base_url}"),
        };
    }

    // The runtimes that can serve this source, in order.
    let mut candidates: Vec<&dyn crate::runtime::ModelRuntime> =
        runtimes.iter().copied().filter(|r| r.accepts(source)).collect();
    candidates.sort_by_key(|r| match r.modification() {
        Modification::Native => 0,
        Modification::ByConversion => 1,
        Modification::PromptOnly => 2,
    });

    let Some(runtime) = candidates.first().copied() else {
        return Plan {
            runtime: "none".to_string(),
            mode: ServeMode::Native,
            steps: Vec::new(),
            reason: format!("no runtime on this machine can serve {}", descriptor.label()),
        };
    };

    let mut steps = Vec::new();
    let mode = match runtime.modification() {
        Modification::PromptOnly => ServeMode::PromptPolicy,
        Modification::Native | Modification::ByConversion => {
            if can_modify_weights {
                // The adapter has to exist. Either it does, or it is built
                // from a donor with the same shape.
                if descriptor.adapter == crate::source::AdapterState::Missing {
                    if let Some(signature) = &descriptor.signature {
                        steps.push(Step::BuildAdapter {
                            donor: String::new(),
                            signature: signature.clone(),
                        });
                    }
                }
                ServeMode::WeightModified
            } else {
                ServeMode::PromptPolicy
            }
        }
    };

    let reason = match mode {
        ServeMode::WeightModified => match descriptor.adapter {
            // The adapter is folded into the weights, not bound to them.
            crate::source::AdapterState::Merged => format!(
                "{} serves {} with the adapter folded into its weights",
                runtime.name(),
                descriptor.label()
            ),
            _ => format!(
                "{} can bind a Heretic adapter to {}",
                runtime.name(),
                descriptor.label()
            ),
        },
        ServeMode::PromptPolicy => format!(
            "{} cannot change the weights of {}, so a prompt-level policy is used",
            runtime.name(),
            descriptor.label()
        ),
        ServeMode::Native => format!("{} serves {} as it is", runtime.name(), descriptor.label()),
    };

    Plan {
        runtime: runtime.id().to_string(),
        mode,
        steps,
        reason,
    }
}

/// The policy to use when a model cannot be modified.
pub fn fallback_policy(descriptor: &ModelDescriptor) -> Policy {
    Policy::for_model(&descriptor.label())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{ModelRuntime, ServeRequest};
    use crate::source::AdapterState;

    struct Fake {
        id: &'static str,
        accepts: bool,
        modification: Modification,
    }

    impl ModelRuntime for Fake {
        fn id(&self) -> &str {
            self.id
        }
        fn name(&self) -> &str {
            self.id
        }
        fn modification(&self) -> Modification {
            self.modification
        }
        fn accepts(&self, _source: &ModelSource) -> bool {
            self.accepts
        }
        fn serve(&self, _request: &ServeRequest) -> anyhow::Result<crate::runtime::ServeHandle> {
            unreachable!()
        }
    }

    fn gguf() -> ModelDescriptor {
        ModelDescriptor::new(ModelSource::Gguf {
            path: "/tmp/x.gguf".into(),
        })
    }

    #[test]
    fn a_runtime_that_can_change_weights_is_preferred() {
        let native = Fake {
            id: "native",
            accepts: true,
            modification: Modification::Native,
        };
        let prompt = Fake {
            id: "prompt",
            accepts: true,
            modification: Modification::PromptOnly,
        };
        let plan = plan(&gguf(), &[&prompt, &native], true);
        assert_eq!(plan.runtime, "native");
        assert_eq!(plan.mode, ServeMode::WeightModified);
    }

    #[test]
    fn a_model_that_cannot_be_modified_falls_back_to_a_policy() {
        let native = Fake {
            id: "native",
            accepts: true,
            modification: Modification::Native,
        };
        let plan = plan(&gguf(), &[&native], false);
        assert_eq!(plan.mode, ServeMode::PromptPolicy);
        assert!(plan.reason.contains("prompt-level policy"));
    }

    #[test]
    fn a_missing_adapter_is_a_step() {
        let native = Fake {
            id: "native",
            accepts: true,
            modification: Modification::Native,
        };
        let mut descriptor = gguf();
        descriptor.signature = Some(Signature {
            arch: "qwen3".into(),
            layers: 36,
            hidden: 4096,
            inter: 12288,
            heads: 32,
            kv_heads: 8,
            experts: 0,
        });
        descriptor.adapter = AdapterState::Missing;

        let plan = plan(&descriptor, &[&native], true);
        assert!(plan.needs_work());
        assert!(matches!(plan.steps[0], Step::BuildAdapter { .. }));
    }

    #[test]
    fn a_remote_model_is_served_where_it_already_is() {
        let descriptor = ModelDescriptor::new(ModelSource::Remote {
            base_url: "http://127.0.0.1:1234/v1".into(),
            model: "local".into(),
        });
        let plan = plan(&descriptor, &[], false);
        assert_eq!(plan.runtime, "openai");
        assert_eq!(plan.mode, ServeMode::PromptPolicy);
        assert!(!plan.needs_work());
    }
}
