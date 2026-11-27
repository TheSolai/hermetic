//! Turning a source into weights a runtime can serve.
//!
//! A runtime serves a file. A source is somewhere weights live. Getting from
//! one to the other is the only conversion there is, and there are three
//! cases:
//!
//!   * A GGUF file: it *is* the weights. Serve it.
//!   * A model in Ollama's store: the blob is a GGUF for most models, and it
//!     is hard-linked into the cache so that one adapter can serve every
//!     quantisation of that shape. The MLX builds are safetensors, and have
//!     to be converted first.
//!   * An MLX model: safetensors, and the adapter has to be folded into them.
//!
//! A model whose weights are held by a server that is already running have
//! no file at all - that is the case the prompt-level policy is for.

use std::path::PathBuf;

use anyhow::Result;

use crate::registry::Registry;
use crate::runtimes::ollama;
use crate::source::ModelSource;

/// The weights to serve for a source, if there are any.
pub fn weights_for(source: &ModelSource, registry: &Registry) -> Option<PathBuf> {
    match source {
        ModelSource::Gguf { path } => Some(path.clone()),
        ModelSource::Ollama { name } => {
            let blob = ollama::gguf_of(name)?;
            let signature = crate::gguf::signature_of(&blob).ok()?;
            registry.adopt_model(&signature, &blob).ok()
        }
        ModelSource::Mlx { path, .. } => {
            // MLX weights are safetensors. They are served by MLX itself, or
            // converted; either way there is no GGUF here yet.
            let _ = path;
            None
        }
        ModelSource::HuggingFace { .. } | ModelSource::Remote { .. } => None,
    }
}

/// The adapter for a shape, if there is one.
pub fn adapter_for(source: &ModelSource, registry: &Registry) -> Option<PathBuf> {
    let signature = crate::gguf::signature_of(&weights_for(source, registry)?).ok()?;
    registry.adapter_for(&signature)
}

/// The model a merged build would produce for a shape, if there is one.
pub fn merged_for(registry: &Registry, signature: &crate::Signature) -> Option<PathBuf> {
    registry.merged_for(signature)
}

/// The weights of a source, and the adapter that goes with them.
pub struct Serving {
    /// The file to serve.
    pub weights: PathBuf,
    /// The adapter to bind, if there is one.
    pub adapter: Option<PathBuf>,
}

/// Work out what to serve.
pub fn plan_serving(
    source: &ModelSource,
    registry: &Registry,
    signature: &crate::Signature,
) -> Result<Serving> {
    // A merged model has the adapter in its weights already.
    if let Some(merged) = registry.merged_for(signature) {
        return Ok(Serving {
            weights: merged,
            adapter: None,
        });
    }
    let weights = weights_for(source, registry).ok_or_else(|| {
        anyhow::anyhow!(
            "the weights of {} are not in a form that can be served yet",
            source.label()
        )
    })?;
    let adapter = registry.adapter_for(signature);
    Ok(Serving { weights, adapter })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gguf_is_its_own_weights() {
        let registry = Registry::new();
        let source = ModelSource::Gguf {
            path: "/tmp/x.gguf".into(),
        };
        assert_eq!(
            weights_for(&source, &registry),
            Some(PathBuf::from("/tmp/x.gguf"))
        );
    }

    #[test]
    fn a_server_held_model_has_no_weights() {
        let registry = Registry::new();
        let source = ModelSource::Remote {
            base_url: "http://x/v1".into(),
            model: "m".into(),
        };
        assert_eq!(weights_for(&source, &registry), None);
    }
}
