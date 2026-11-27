//! The runtimes the terminal can serve a model with.
//!
//! The terminal does not choose a runtime itself: it asks
//! [`hermetic_core::planner`] which one should serve a model, and then hands
//! the request to that runtime. All this module does is put the runtimes in
//! the order they should be tried in, and add the one that is not a program
//! on this machine - a server that is already running.

use anyhow::Result;
use hermetic_core::runtime::ModelRuntime;
use hermetic_core::runtimes::openai::OpenAiRuntime;
use hermetic_core::source::ModelSource;

/// The id of the runtime that serves a model a server already holds.
pub const OPENAI_ID: &str = "openai";

/// Every runtime that can serve this source, in the order they should be
/// tried.
///
/// MLX comes first: it serves an MLX model directly, while llama.cpp cannot
/// read safetensors at all, so llama.cpp must not be preferred for one.
pub fn runtimes_for(source: &ModelSource) -> Vec<Box<dyn ModelRuntime>> {
    let mut out = hermetic_core::runtimes::available();
    out.sort_by_key(|runtime| u8::from(runtime.id() != "mlx"));
    if let ModelSource::Remote { base_url, model } = source {
        let model = if model.is_empty() {
            first_model(base_url).unwrap_or_default()
        } else {
            model.clone()
        };
        out.push(Box::new(OpenAiRuntime::new(base_url, &model)));
    }
    out
}

/// The runtime the planner picked.
pub fn find<'a>(runtimes: &'a [&dyn ModelRuntime], id: &str) -> Option<&'a dyn ModelRuntime> {
    runtimes.iter().find(|runtime| runtime.id() == id).copied()
}

/// The first model a server offers.
pub fn first_model(base_url: &str) -> Option<String> {
    let runtime = OpenAiRuntime::new(base_url, "");
    runtime.models().into_iter().next()
}

/// The address of a server, and whether it answers.
pub fn answers(base_url: &str) -> bool {
    OpenAiRuntime::new(base_url, "").is_answering()
}

/// The models a server offers.
pub fn models_of(base_url: &str) -> Result<Vec<String>> {
    Ok(OpenAiRuntime::new(base_url, "").models())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermetic_core::runtime::{Modification, ServeMode};
    use hermetic_core::source::ModelDescriptor;

    #[test]
    fn a_remote_model_is_served_by_the_openai_runtime() {
        let source = ModelSource::Remote {
            base_url: "http://127.0.0.1:1234/v1".into(),
            model: "local".into(),
        };
        let runtimes = runtimes_for(&source);
        let refs: Vec<&dyn ModelRuntime> = runtimes.iter().map(|r| r.as_ref()).collect();
        let runtime = find(&refs, OPENAI_ID).expect("the openai runtime");
        assert_eq!(runtime.modification(), Modification::PromptOnly);
        assert!(runtime.accepts(&source));
    }

    #[test]
    fn the_planner_says_openai_for_a_model_a_server_holds() {
        let descriptor = ModelDescriptor::new(ModelSource::Remote {
            base_url: "http://127.0.0.1:1234/v1".into(),
            model: "local".into(),
        });
        let runtimes = runtimes_for(&descriptor.source);
        let refs: Vec<&dyn ModelRuntime> = runtimes.iter().map(|r| r.as_ref()).collect();
        let plan = hermetic_core::planner::plan(&descriptor, &refs, false);
        assert_eq!(plan.runtime, OPENAI_ID);
        assert_eq!(plan.mode, ServeMode::PromptPolicy);
    }
}
