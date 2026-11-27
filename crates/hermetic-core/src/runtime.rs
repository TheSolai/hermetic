//! How a model gets in front of the user.
//!
//! This is the second half of the abstraction. A runtime is a way of serving
//! a model: llama.cpp, Ollama's own engine, MLX, or any server that speaks
//! the OpenAI chat API.
//!
//! Every runtime that can be talked to at all exposes the *same* chat API -
//! `POST /v1/chat/completions` - so the terminal has exactly one chat client
//! and does not care what is behind it. That is the whole trick: it is what
//! lets the terminal treat MLX and llama.cpp and a local LM Studio the same
//! way.
//!
//! What the runtimes do *not* share is whether they can change weights. That
//! is what [`Modification`] is for.

use std::path::PathBuf;

use anyhow::Result;

use crate::signature::Signature;
use crate::source::ModelSource;

/// A runtime that can put a model in front of the user.
pub trait ModelRuntime: Send + Sync {
    /// Its name, as a program: `llama.cpp`, `ollama`, `mlx`, `openai`.
    fn id(&self) -> &str;

    /// Its name, for people.
    fn name(&self) -> &str;

    /// What this runtime can do with a Heretic adapter.
    fn modification(&self) -> Modification;

    /// Can this runtime serve this source at all?
    fn accepts(&self, source: &ModelSource) -> bool;

    /// Serve the model.
    fn serve(&self, request: &ServeRequest) -> Result<ServeHandle>;

    /// The models this runtime knows about by itself, if it has any.
    fn models(&self) -> Vec<crate::source::ModelDescriptor> {
        Vec::new()
    }

    /// Is this runtime there? A runtime that is not installed is not an
    /// error; it is simply not an option.
    fn is_available(&self) -> bool {
        true
    }
}

/// What a runtime can do with a Heretic adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modification {
    /// It binds an adapter to the weights as it serves them. Exact, and
    /// instant once the adapter exists.
    Native,
    /// It cannot bind one, but the weights can be turned into something it
    /// can - usually by folding the adapter into them.
    ByConversion,
    /// It cannot touch the weights at all. The model can still be served, but
    /// only with a prompt-level policy, and the terminal says so.
    PromptOnly,
}

impl Modification {
    /// Can this runtime serve a model whose weights have been changed?
    pub fn can_modify(&self) -> bool {
        !matches!(self, Modification::PromptOnly)
    }

    /// A short word for the state.
    pub fn label(&self) -> &'static str {
        match self {
            Modification::Native => "native",
            Modification::ByConversion => "converted",
            Modification::PromptOnly => "prompt only",
        }
    }
}

/// What to serve.
pub struct ServeRequest {
    /// Where the weights are.
    pub source: ModelSource,
    /// The shape of those weights.
    pub signature: Signature,
    /// The adapter to bind, as a GGUF LoRA.
    pub adapter: Option<PathBuf>,
    /// The weights to serve, when they are already a file on disk.
    pub weights: Option<PathBuf>,
    /// A prompt-level policy, for models whose weights cannot be changed.
    pub policy: Option<crate::policy::Policy>,
    /// How many layers of context to ask for.
    pub context: u32,
}

impl ServeRequest {
    /// A request for a model with no adapter yet.
    pub fn new(source: ModelSource, signature: Signature) -> ServeRequest {
        ServeRequest {
            source,
            signature,
            adapter: None,
            weights: None,
            policy: None,
            context: 4096,
        }
    }
}

/// A running model.
pub struct ServeHandle {
    /// Where to talk to it: an OpenAI-compatible base URL.
    pub endpoint: String,
    /// How the adapter was applied, if it was.
    pub mode: ServeMode,
    /// The model's name on that endpoint.
    pub model: String,
    /// Anything worth telling the user about how this was served.
    pub notes: Vec<String>,
    /// How to stop it again.
    pub stop: Box<dyn Fn() + Send + Sync>,
}

impl ServeHandle {
    /// A handle with nothing to stop.
    pub fn new(endpoint: &str, mode: ServeMode, model: &str) -> ServeHandle {
        ServeHandle {
            endpoint: endpoint.to_string(),
            mode,
            model: model.to_string(),
            notes: Vec::new(),
            stop: Box::new(|| {}),
        }
    }
}

/// How a model is being served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeMode {
    /// The weights have been changed: the Heretic adapter is bound in.
    WeightModified,
    /// The weights could not be changed, so a prompt-level policy is in use.
    PromptPolicy,
    /// The model is served as it is.
    Native,
}

impl ServeMode {
    /// A short word for the mode.
    pub fn label(&self) -> &'static str {
        match self {
            ServeMode::WeightModified => "hereticed",
            ServeMode::PromptPolicy => "prompt policy",
            ServeMode::Native => "as it is",
        }
    }

    /// Have the weights really been changed?
    pub fn is_modified(&self) -> bool {
        matches!(self, ServeMode::WeightModified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_native_and_converted_can_change_weights() {
        assert!(Modification::Native.can_modify());
        assert!(Modification::ByConversion.can_modify());
        assert!(!Modification::PromptOnly.can_modify());
    }

    #[test]
    fn the_mode_says_whether_the_weights_changed() {
        assert!(ServeMode::WeightModified.is_modified());
        assert!(!ServeMode::PromptPolicy.is_modified());
        assert_eq!(ServeMode::WeightModified.label(), "hereticed");
    }
}
