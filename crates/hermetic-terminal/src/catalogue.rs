//! The models the terminal knows about.
//!
//! The old terminal walked Ollama's store and nothing else. This one asks
//! [`hermetic_core::discovery`] instead, so the list holds every model on the
//! machine: the ones in Ollama's store, the GGUF files in a folder, the MLX
//! models in a folder, and the models a running server holds.
//!
//! What the panel shows about a model has not changed - its name, how big it
//! is, and what the terminal can do with it - except that the kind of source
//! is now shown too, because two models can have the same name.

use std::path::PathBuf;

use hermetic_core::discovery::{self, Sources};
use hermetic_core::registry::Registry;
use hermetic_core::runtimes::ollama;
use hermetic_core::signature::Signature;
use hermetic_core::source::{ModelDescriptor, ModelSource, SourceKind};

/// The folders of GGUF files to look in.
pub const GGUF_DIRS_ENV: &str = "HERMETIC_GGUF_DIRS";

/// The folders of MLX models to look in.
pub const MLX_DIRS_ENV: &str = "HERMETIC_MLX_DIRS";

/// The servers that are already running.
pub const REMOTE_ENV: &str = "HERMETIC_REMOTE";

/// One model in the list.
#[derive(Debug, Clone)]
pub struct CatalogueEntry {
    /// The name the user would type.
    pub name: String,
    /// Its shape, when it is known.
    pub signature: Option<Signature>,
    /// How big it is on disk.
    pub size: u64,
    /// Where its weights are.
    pub kind: SourceKind,
    /// What the terminal can do with it right now.
    pub state: String,
    /// The source it comes from, so it can be loaded again.
    pub source: ModelSource,
}

impl CatalogueEntry {
    /// The spec that loads this model again.
    pub fn spec(&self) -> String {
        self.source.id()
    }
}

/// Every model that can be found, from every source.
pub fn catalogue(sources: &Sources, registry: &Registry) -> Vec<CatalogueEntry> {
    discovery::discover(sources, registry)
        .into_iter()
        .map(|descriptor| entry_of(descriptor, registry))
        .collect()
}

/// What the terminal knows about one model.
fn entry_of(descriptor: ModelDescriptor, registry: &Registry) -> CatalogueEntry {
    let state = state_of(registry, &descriptor);
    CatalogueEntry {
        name: descriptor.label(),
        signature: descriptor.signature.clone(),
        size: descriptor.size_bytes.unwrap_or(0),
        kind: descriptor.source.kind(),
        state,
        source: descriptor.source.clone(),
    }
}

/// What the terminal can do with a model right now.
pub fn state_of(registry: &Registry, descriptor: &ModelDescriptor) -> String {
    let Some(signature) = &descriptor.signature else {
        return "unknown".into();
    };
    if registry.merged_for(signature).is_some() {
        return "merged".into();
    }
    if registry.adapter_for(signature).is_some() {
        if has_weights(descriptor) || registry.model_for(signature).is_some() {
            return "ready".into();
        }
        return "no gguf".into();
    }
    "build".into()
}

/// Are this model's weights something that can be served?
fn has_weights(descriptor: &ModelDescriptor) -> bool {
    match &descriptor.source {
        ModelSource::Gguf { .. } => true,
        ModelSource::Ollama { name } => ollama::gguf_of(name).is_some(),
        ModelSource::Mlx { .. } | ModelSource::HuggingFace { .. } | ModelSource::Remote { .. } => {
            false
        }
    }
}

/// Where to look for models.
///
/// Ollama's store is always there. The folders and the servers come from the
/// command line and from the environment.
pub fn sources(
    gguf_dirs: &[PathBuf],
    mlx_dirs: &[PathBuf],
    remote: &[(String, String)],
) -> Sources {
    let mut out = Sources {
        ollama: Some(discovery::ollama_root()),
        gguf_dirs: gguf_dirs.to_vec(),
        mlx_dirs: mlx_dirs.to_vec(),
        remote: remote.to_vec(),
    };

    if let Some(text) = std::env::var_os(GGUF_DIRS_ENV) {
        for directory in split_paths(&text.to_string_lossy()) {
            if !out.gguf_dirs.contains(&directory) {
                out.gguf_dirs.push(directory);
            }
        }
    }
    if let Some(text) = std::env::var_os(MLX_DIRS_ENV) {
        for directory in split_paths(&text.to_string_lossy()) {
            if !out.mlx_dirs.contains(&directory) {
                out.mlx_dirs.push(directory);
            }
        }
    }
    if let Some(text) = std::env::var_os(REMOTE_ENV) {
        for server in text.to_string_lossy().split(',') {
            let server = server.trim();
            if server.is_empty() {
                continue;
            }
            let pair = remote_spec(server);
            if !out.remote.contains(&pair) {
                out.remote.push(pair);
            }
        }
    }

    out
}

/// A folder list: `one:two:three`.
fn split_paths(text: &str) -> Vec<PathBuf> {
    text.split(':')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(PathBuf::from)
        .collect()
}

/// A server, as `base-url#model` or just `base-url`.
pub fn remote_spec(text: &str) -> (String, String) {
    match text.rsplit_once('#') {
        Some((base, model)) => (base.trim().to_string(), model.trim().to_string()),
        None => (text.trim().to_string(), String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ollama_store_is_always_looked_in() {
        let sources = sources(&[], &[], &[]);
        assert!(sources.ollama.is_some());
    }

    #[test]
    fn a_server_is_read_with_its_model() {
        assert_eq!(
            remote_spec("http://127.0.0.1:1234/v1#local"),
            ("http://127.0.0.1:1234/v1".to_string(), "local".to_string())
        );
        assert_eq!(
            remote_spec("http://127.0.0.1:1234/v1"),
            ("http://127.0.0.1:1234/v1".to_string(), String::new())
        );
    }

    #[test]
    fn a_model_without_a_shape_has_no_state() {
        let registry = Registry::new();
        let descriptor = ModelDescriptor::new(ModelSource::Remote {
            base_url: "http://127.0.0.1:1234/v1".into(),
            model: "local".into(),
        });
        assert_eq!(state_of(&registry, &descriptor), "unknown");
    }
}
