//! Where a model's weights come from.
//!
//! This is the first half of the abstraction: a model is identified by
//! *where it lives*, not by which program will eventually serve it. The same
//! weights can be reached through different runtimes, and the same runtime
//! can serve weights from different places.
//!
//! Every source can answer three questions:
//!
//!   * what is it called?
//!   * what shape are its weights?
//!   * can they be modified?
//!
//! The third one is the interesting one. Heretic changes weights, so a
//! source whose weights cannot be written to - a server that already holds
//! them, a closed API - can only be served with a prompt-level policy
//! instead. Saying that plainly is part of the design, not a failure.

use std::path::PathBuf;

use crate::signature::Signature;

/// Where a model's weights come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// A model in Ollama's store.
    Ollama {
        /// The name the user would type: `qwen2.5:0.5b`.
        name: String,
    },
    /// A GGUF file on disk.
    Gguf {
        /// The file.
        path: PathBuf,
    },
    /// An MLX model: a directory of safetensors and a config.
    Mlx {
        /// The directory.
        path: PathBuf,
        /// The hub repository it came from, when it is known.
        repo: Option<String>,
    },
    /// A model on the Hugging Face hub, not downloaded yet.
    HuggingFace {
        /// The repository: `Qwen/Qwen3-8B`.
        repo: String,
    },
    /// A model a server already holds, behind an OpenAI-compatible API.
    Remote {
        /// The base URL: `http://127.0.0.1:1234/v1`.
        base_url: String,
        /// The model's name on that server.
        model: String,
    },
}

impl ModelSource {
    /// A stable identifier: `ollama:qwen2.5:0.5b`.
    pub fn id(&self) -> String {
        match self {
            ModelSource::Ollama { name } => format!("ollama:{name}"),
            ModelSource::Gguf { path } => format!("gguf:{}", path.display()),
            ModelSource::Mlx { path, .. } => format!("mlx:{}", path.display()),
            ModelSource::HuggingFace { repo } => format!("hf:{repo}"),
            ModelSource::Remote { base_url, model } => format!("openai:{base_url}#{model}"),
        }
    }

    /// The name to show the user.
    pub fn label(&self) -> String {
        match self {
            ModelSource::Ollama { name } => name.clone(),
            ModelSource::Gguf { path } => path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string()),
            ModelSource::Mlx { path, repo } => repo.clone().unwrap_or_else(|| {
                path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.display().to_string())
            }),
            ModelSource::HuggingFace { repo } => repo.clone(),
            ModelSource::Remote { model, .. } => model.clone(),
        }
    }

    /// What kind of source this is.
    pub fn kind(&self) -> SourceKind {
        match self {
            ModelSource::Ollama { .. } => SourceKind::Ollama,
            ModelSource::Gguf { .. } => SourceKind::Gguf,
            ModelSource::Mlx { .. } => SourceKind::Mlx,
            ModelSource::HuggingFace { .. } => SourceKind::HuggingFace,
            ModelSource::Remote { .. } => SourceKind::Remote,
        }
    }

    /// Can these weights be written to?
    ///
    /// This is the question that decides whether a model can be hereticed or
    /// only talked to politely.
    pub fn is_modifiable(&self) -> bool {
        match self {
            // Ollama's store holds the weights in files we can read and
            // convert, so the model itself can be modified - even though
            // Ollama's own engine will not do it for us.
            ModelSource::Ollama { .. } => true,
            ModelSource::Gguf { .. } => true,
            ModelSource::Mlx { .. } => true,
            // The hub holds the original weights; we download them.
            ModelSource::HuggingFace { .. } => true,
            // A server holds them, and we do not have them at all.
            ModelSource::Remote { .. } => false,
        }
    }
}

/// The kind of source, for grouping and for icons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Ollama,
    Gguf,
    Mlx,
    HuggingFace,
    Remote,
}

impl SourceKind {
    /// A short name for the kind.
    pub fn label(&self) -> &'static str {
        match self {
            SourceKind::Ollama => "ollama",
            SourceKind::Gguf => "gguf",
            SourceKind::Mlx => "mlx",
            SourceKind::HuggingFace => "hub",
            SourceKind::Remote => "remote",
        }
    }
}

/// What the terminal knows about a model it could load.
#[derive(Debug, Clone)]
pub struct ModelDescriptor {
    /// Where its weights are.
    pub source: ModelSource,
    /// The shape, when it is known.
    pub signature: Option<Signature>,
    /// How big it is on disk, when that is known.
    pub size_bytes: Option<u64>,
    /// How many parameters it has, when that is known.
    pub parameters: Option<u64>,
    /// How it is quantised: `Q4_K_M`.
    pub quantisation: Option<String>,
    /// Whether an adapter for this shape exists.
    pub adapter: AdapterState,
}

impl ModelDescriptor {
    /// A descriptor with nothing known but the source.
    pub fn new(source: ModelSource) -> ModelDescriptor {
        ModelDescriptor {
            source,
            signature: None,
            size_bytes: None,
            parameters: None,
            quantisation: None,
            adapter: AdapterState::Unknown,
        }
    }

    /// The name to show.
    pub fn label(&self) -> String {
        self.source.label()
    }

    /// A stable identifier.
    pub fn id(&self) -> String {
        self.source.id()
    }
}

/// Whether an adapter for a model is ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterState {
    /// An adapter exists for this shape, and the weights to serve it.
    Ready,
    /// The adapter is already folded into a set of weights.
    Merged,
    /// There is no adapter yet: one would have to be built.
    Missing,
    /// The shape is not known, so nothing can be said.
    Unknown,
}

impl AdapterState {
    /// A short word for the state.
    pub fn label(&self) -> &'static str {
        match self {
            AdapterState::Ready => "ready",
            AdapterState::Merged => "merged",
            AdapterState::Missing => "build",
            AdapterState::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_source_has_a_stable_id() {
        let ollama = ModelSource::Ollama {
            name: "qwen2.5:0.5b".into(),
        };
        assert_eq!(ollama.id(), "ollama:qwen2.5:0.5b");
        assert_eq!(ollama.label(), "qwen2.5:0.5b");
        assert_eq!(ollama.kind(), SourceKind::Ollama);
    }

    #[test]
    fn a_remote_model_cannot_be_modified() {
        let remote = ModelSource::Remote {
            base_url: "http://127.0.0.1:1234/v1".into(),
            model: "local".into(),
        };
        assert!(!remote.is_modifiable());
        assert_eq!(remote.kind(), SourceKind::Remote);
    }

    #[test]
    fn weights_on_disk_can_be_modified() {
        let gguf = ModelSource::Gguf {
            path: "/tmp/model.gguf".into(),
        };
        assert!(gguf.is_modifiable());
        assert_eq!(gguf.label(), "model.gguf");
    }
}
