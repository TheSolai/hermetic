//! Ollama: a source of weights, and a runtime that serves them.
//!
//! Ollama is two things at once, and the terminal treats them as two things:
//!
//!   * **A source.** Ollama's store holds the weights, and the terminal can
//!     read them, convert them and bind a Heretic adapter to them. That is
//!     how an Ollama model gets hereticed: the weights are served by
//!     llama.cpp with the adapter, not by Ollama's own engine.
//!   * **A runtime.** Ollama's own engine is fast and needs no conversion,
//!     but it cannot change weights. When a model cannot be hereticed, or
//!     when the user only wants the model, Ollama serves it - through its
//!     OpenAI-compatible API, so the terminal's one chat client still works.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::Value;

use crate::discovery::is_gguf_file;
use crate::runtime::{Modification, ModelRuntime, ServeHandle, ServeMode, ServeRequest};
use crate::source::ModelSource;

/// Where Ollama is listening.
pub fn ollama_addr() -> (String, u16) {
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".into());
    let host = host
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    match host.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse::<u16>().unwrap_or(11434)),
        None => (host.to_string(), 11434),
    }
}

/// Split "name:tag" into its parts.
pub fn split_tag(spec: &str) -> (&str, &str) {
    match spec.rsplit_once(':') {
        Some((base, tag)) if !base.is_empty() && !tag.contains('/') => (base, tag),
        _ => (spec, "latest"),
    }
}

/// The manifest for a model in Ollama's store.
pub fn manifest_of(spec: &str) -> Option<PathBuf> {
    let (name, tag) = split_tag(spec);
    let parts: Vec<&str> = name.split('/').collect();
    let root = crate::discovery::ollama_root();

    // The new layout: ollama.com/<namespace>/<repo>/<tag>.
    let v2 = match parts.as_slice() {
        [repo] => root
            .join("manifests-v2")
            .join("ollama.com")
            .join("library")
            .join(repo)
            .join(tag),
        [ns, repo] => root.join("manifests-v2").join("ollama.com").join(ns).join(repo).join(tag),
        _ => PathBuf::new(),
    };
    if v2.exists() {
        return Some(v2);
    }

    // The old layout: registry.ollama.ai/<namespace>/<repo>/<tag>.
    let v1 = match parts.as_slice() {
        [repo] => root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join(repo)
            .join(tag),
        [ns, repo] => root
            .join("manifests")
            .join("registry.ollama.ai")
            .join(ns)
            .join(repo)
            .join(tag),
        _ => PathBuf::new(),
    };
    if v1.exists() {
        return Some(v1);
    }
    None
}

/// The weights a manifest points at.
pub fn blob_of(manifest: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(manifest).ok()?;
    let json: Value = serde_json::from_str(&text).ok()?;
    let layers = json.get("layers")?.as_array()?;
    for layer in layers {
        let media = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
        if !media.ends_with("image.model") {
            continue;
        }
        let Some(digest) = layer.get("digest").and_then(|d| d.as_str()) else {
            continue;
        };
        let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
        let blob = crate::discovery::ollama_root()
            .join("blobs")
            .join(format!("sha256-{hex}"));
        if blob.exists() {
            return Some(blob);
        }
    }
    None
}

/// The GGUF an Ollama model is stored as, if it is stored as one.
///
/// Ollama keeps some models as safetensors - the MLX builds - which
/// llama.cpp cannot read. Those have to be converted before they can be
/// served with an adapter.
pub fn gguf_of(spec: &str) -> Option<PathBuf> {
    let manifest = manifest_of(spec)?;
    let blob = blob_of(&manifest)?;
    if is_gguf_file(&blob) {
        Some(blob)
    } else {
        None
    }
}

/// Ollama's own engine.
#[derive(Debug, Clone)]
pub struct OllamaRuntime {
    /// The host Ollama is on.
    pub host: String,
    /// The port Ollama is on.
    pub port: u16,
}

impl Default for OllamaRuntime {
    fn default() -> OllamaRuntime {
        let (host, port) = ollama_addr();
        OllamaRuntime { host, port }
    }
}

impl OllamaRuntime {
    /// The runtime, at the address Ollama is on.
    pub fn new() -> OllamaRuntime {
        OllamaRuntime::default()
    }

    /// Is Ollama running?
    pub fn is_running(&self) -> bool {
        crate::http::post_json(&self.host, self.port, "/api/tags", &serde_json::json!({}))
            .is_ok()
    }
}

impl ModelRuntime for OllamaRuntime {
    fn id(&self) -> &str {
        "ollama"
    }

    fn name(&self) -> &str {
        "Ollama"
    }

    fn modification(&self) -> Modification {
        // Ollama's engine serves the weights it has. It cannot bind an
        // adapter to them.
        Modification::PromptOnly
    }

    fn accepts(&self, source: &ModelSource) -> bool {
        matches!(source, ModelSource::Ollama { .. })
    }

    fn serve(&self, request: &ServeRequest) -> Result<ServeHandle> {
        let ModelSource::Ollama { name } = &request.source else {
            anyhow::bail!("Ollama does not serve {}", request.source.label());
        };
        if !self.is_running() {
            anyhow::bail!("Ollama is not answering on {}:{}", self.host, self.port);
        }
        // Ollama speaks the OpenAI API, so the terminal's one chat client
        // works with it unchanged.
        let endpoint = crate::chat::base_url(&self.host, self.port);
        let mut handle = ServeHandle::new(&endpoint, ServeMode::PromptPolicy, name);
        handle.notes.push(
            "served by Ollama's own engine; its weights cannot be changed, \
             so a prompt-level policy is used"
                .to_string(),
        );
        Ok(handle)
    }

    fn is_available(&self) -> bool {
        self.is_running()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tag_is_split_off() {
        assert_eq!(split_tag("qwen3:8b"), ("qwen3", "8b"));
        assert_eq!(split_tag("qwen3"), ("qwen3", "latest"));
    }

    #[test]
    fn ollamas_own_engine_cannot_change_weights() {
        let runtime = OllamaRuntime::new();
        assert_eq!(runtime.id(), "ollama");
        assert_eq!(runtime.modification(), Modification::PromptOnly);
        assert!(runtime.accepts(&ModelSource::Ollama { name: "x".into() }));
        assert!(!runtime.accepts(&ModelSource::Gguf {
            path: "/tmp/x.gguf".into()
        }));
    }
}
