//! The runtimes.
//!
//! A runtime is a way of serving a model. Each one says what it can do with
//! a Heretic adapter, and that is the only thing that really differs between
//! them:
//!
//! | runtime | weights | adapter |
//! |---|---|---|
//! | `llama.cpp` | GGUF files | bound at runtime, exactly |
//! | `mlx` | safetensors directories | folded into the weights |
//! | `ollama` | Ollama's store | not changeable - prompt policy instead |
//! | `openai` | whatever a server holds | not changeable - prompt policy instead |
//!
//! The last two are the same runtime in every way that matters: a server
//! that already holds the weights and will not let them be changed.

pub mod llamacpp;
pub mod mlx;
pub mod ollama;
pub mod openai;

use crate::runtime::ModelRuntime;

/// Every runtime that is on this machine.
pub fn available() -> Vec<Box<dyn ModelRuntime>> {
    let mut out: Vec<Box<dyn ModelRuntime>> = Vec::new();
    let llama = llamacpp::LlamaCppRuntime::new();
    if llama.is_available() {
        out.push(Box::new(llama));
    }
    let ollama = ollama::OllamaRuntime::new();
    if ollama.is_available() {
        out.push(Box::new(ollama));
    }
    let mlx = mlx::MlxRuntime::default();
    if mlx.is_available() {
        out.push(Box::new(mlx));
    }
    out
}
