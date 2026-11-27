//! What the user asked for, and where its weights are.
//!
//! A model is named in one of a few ways, and each of them says where the
//! weights live:
//!
//! ```text
//! ollama:qwen2.5:0.5b            a model in Ollama's store
//! gguf:/models/qwen3-8b.gguf     a GGUF file
//! mlx:/models/qwen3-mlx          an MLX model: a directory of safetensors
//! hf:Qwen/Qwen3-8B               a model on the Hugging Face hub
//! openai:http://127.0.0.1:1234/v1#local   a server that is already running
//! ```
//!
//! Those are exactly the identifiers [`hermetic_core::ModelSource`] hands out,
//! so a spec is something the terminal can write down and the interceptor can
//! read back. A bare name - `qwen2.5:0.5b` - is a model in Ollama's store,
//! which is what the user means when they type one.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use hermetic_core::signature::Signature;
use hermetic_core::source::ModelSource;

/// The prefix of a model in Ollama's store.
pub const OLLAMA_PREFIX: &str = "ollama:";

/// The prefix of a GGUF file.
pub const GGUF_PREFIX: &str = "gguf:";

/// The prefix of an MLX model.
pub const MLX_PREFIX: &str = "mlx:";

/// The prefix of a model on the hub.
pub const HUB_PREFIX: &str = "hf:";

/// The prefix of a model a server already holds.
pub const REMOTE_PREFIX: &str = "openai:";

/// Turn a spec into the source it names.
pub fn source_of(spec: &str) -> Result<ModelSource> {
    let spec = spec.trim().trim_matches('\'').trim_matches('"');
    if spec.is_empty() {
        bail!("no model was named");
    }

    if let Some(rest) = spec.strip_prefix(OLLAMA_PREFIX) {
        return Ok(ModelSource::Ollama {
            name: rest.to_string(),
        });
    }
    if let Some(rest) = spec.strip_prefix(GGUF_PREFIX) {
        return Ok(ModelSource::Gguf {
            path: PathBuf::from(rest),
        });
    }
    if let Some(rest) = spec.strip_prefix(MLX_PREFIX) {
        return Ok(ModelSource::Mlx {
            path: PathBuf::from(rest),
            repo: None,
        });
    }
    if let Some(rest) = spec.strip_prefix(HUB_PREFIX) {
        return Ok(ModelSource::HuggingFace {
            repo: rest.to_string(),
        });
    }
    if let Some(rest) = spec.strip_prefix(REMOTE_PREFIX) {
        let (base_url, model) = split_remote(rest);
        return Ok(ModelSource::Remote { base_url, model });
    }

    // A path on disk: a GGUF file, or an MLX model, which is a directory.
    let path = Path::new(spec);
    if path.exists() {
        if path.is_file() {
            return Ok(ModelSource::Gguf {
                path: path.to_path_buf(),
            });
        }
        if hermetic_core::discovery::is_mlx_model(path) {
            return Ok(ModelSource::Mlx {
                path: path.to_path_buf(),
                repo: None,
            });
        }
        bail!("{} is a directory, but not an MLX model", path.display());
    }

    // A model on the hub.
    if let Some(repo) = spec.strip_prefix("hf.co/") {
        return Ok(ModelSource::HuggingFace {
            repo: repo.to_string(),
        });
    }

    // A server that is already running.
    if let Some((base_url, model)) = remote_of(spec) {
        return Ok(ModelSource::Remote { base_url, model });
    }

    // Anything else is a model in Ollama's store.
    Ok(ModelSource::Ollama {
        name: spec.to_string(),
    })
}

/// The shape of a source, when it can be read.
///
/// A server that already holds the weights does not tell us their shape, and
/// does not need to: there is nothing to bind an adapter to.
pub fn shape_of(source: &ModelSource) -> Option<Signature> {
    match source {
        ModelSource::Gguf { path } => Signature::from_gguf(path).ok(),
        ModelSource::Ollama { name } => ollama_shape(name),
        ModelSource::Mlx { path, .. } => hermetic_core::discovery::mlx_signature(path),
        ModelSource::HuggingFace { repo } => hermetic_core::hub::config_of(repo)
            .ok()
            .and_then(|config| Signature::from_hf_config(&config).ok()),
        ModelSource::Remote { .. } => None,
    }
}

/// A shape for a source that has none.
///
/// The request a runtime is given needs a shape even when there is nothing to
/// bind, and this says so honestly rather than making one up.
pub fn placeholder_signature() -> Signature {
    Signature {
        arch: "unknown".to_string(),
        layers: 0,
        hidden: 0,
        inter: 0,
        heads: 0,
        kv_heads: 0,
        experts: 0,
    }
}

/// What else is known about a model, for finding the original weights.
///
/// The name of a model is often a lie - a distilled `deepseek-r1:8b` is really
/// a `Qwen3-8B` - so the search is driven by the shape and the parameter
/// count instead.
pub fn hint_of(source: &ModelSource, signature: &Signature) -> hermetic_core::hub::Hint {
    let mut hint = hermetic_core::hub::Hint {
        parameters: None,
        architecture: Some(signature.arch.clone()),
        size_label: None,
        name: None,
    };

    match source {
        ModelSource::Ollama { name } => {
            if let Some(info) = ollama_info(name) {
                hint.parameters = info.get("general.parameter_count").and_then(|v| v.as_u64());
                hint.architecture = info
                    .get("general.architecture")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .or(hint.architecture);
                hint.size_label = info
                    .get("general.size_label")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
            }
            // The weights know their own name. Ollama's model name does not.
            if let Some(blob) = hermetic_core::runtimes::ollama::gguf_of(name) {
                hint.name = hermetic_core::gguf::gguf_name(&blob);
            }
        }
        ModelSource::Gguf { path } => {
            hint.size_label = hermetic_core::hub::size_label_from_name(&path.display().to_string());
            hint.name = hermetic_core::gguf::gguf_name(path);
        }
        ModelSource::Mlx { path, .. } => {
            if let Some(config) = read_config(path) {
                hint.parameters = config
                    .get("num_parameters")
                    .and_then(|v| v.as_u64())
                    .or_else(|| {
                        config
                            .get("text_config")
                            .and_then(|c| c.get("num_parameters"))
                            .and_then(|v| v.as_u64())
                    });
            }
        }
        ModelSource::HuggingFace { .. } | ModelSource::Remote { .. } => {}
    }

    hint
}

/// A model's `config.json`, when it has one.
fn read_config(directory: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(directory.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// The shape of a model in Ollama's store.
///
/// The GGUF blob is read first, because it is on disk and free to read. Only
/// a model Ollama keeps as safetensors has to be asked about.
fn ollama_shape(name: &str) -> Option<Signature> {
    if let Some(blob) = hermetic_core::runtimes::ollama::gguf_of(name) {
        if let Ok(signature) = Signature::from_gguf(&blob) {
            return Some(signature);
        }
    }
    let info = ollama_info(name)?;
    Signature::from_ollama_info(&info).ok()
}

/// What Ollama knows about one of its models.
///
/// Ollama knows more about its own models than a GGUF header does: the
/// architecture, the exact parameter count, and the family the model came
/// from.
pub fn ollama_info(name: &str) -> Option<serde_json::Value> {
    let (host, port) = hermetic_core::runtimes::ollama::ollama_addr();
    let answer = hermetic_core::http::post_json(
        &host,
        port,
        "/api/show",
        &serde_json::json!({ "name": name }),
    )
    .ok()?;
    answer
        .get("model_info")
        .and_then(|m| m.as_object())
        .map(|m| serde_json::Value::Object(m.clone()))
}

/// Split a remote spec into its base URL and the model on that server.
fn split_remote(rest: &str) -> (String, String) {
    match rest.rsplit_once('#') {
        Some((base, model)) => (normalise_base(base), model.to_string()),
        None => (normalise_base(rest), String::new()),
    }
}

/// The base URL of an OpenAI-compatible server.
///
/// The path is kept when there is one, because a server may put its API
/// somewhere other than `/v1`; but a server that has no path of its own gets
/// the usual `/v1`.
fn normalise_base(base: &str) -> String {
    let base = base.trim().trim_end_matches('/');
    if base.is_empty() {
        return base.to_string();
    }
    if base.contains("://") {
        if base.ends_with("/v1") {
            base.to_string()
        } else {
            format!("{base}/v1")
        }
    } else {
        format!("http://{base}/v1")
    }
}

/// Is this the address of a server that is already running?
///
/// A bare `host:port` is one, but so is `qwen2.5:0.5b` - and that is a model
/// name, not an address. The host has to look like an address for the spec to
/// be read as one.
fn remote_of(spec: &str) -> Option<(String, String)> {
    if spec.contains("://") {
        return Some((normalise_base(spec), String::new()));
    }
    let (host, port) = spec.rsplit_once(':')?;
    port.parse::<u16>().ok()?;
    if host == "localhost" || host.parse::<std::net::IpAddr>().is_ok() {
        return Some((normalise_base(spec), String::new()));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prefixed_spec_names_its_source() {
        let source = source_of("ollama:qwen2.5:0.5b").expect("a source");
        assert_eq!(
            source,
            ModelSource::Ollama {
                name: "qwen2.5:0.5b".into()
            }
        );

        let source = source_of("gguf:/models/qwen3-8b.gguf").expect("a source");
        assert_eq!(
            source,
            ModelSource::Gguf {
                path: PathBuf::from("/models/qwen3-8b.gguf")
            }
        );

        let source = source_of("openai:http://127.0.0.1:1234/v1#local").expect("a source");
        assert_eq!(
            source,
            ModelSource::Remote {
                base_url: "http://127.0.0.1:1234/v1".into(),
                model: "local".into(),
            }
        );
    }

    #[test]
    fn a_bare_name_is_a_model_in_the_ollama_store() {
        let source = source_of("qwen2.5:0.5b").expect("a source");
        assert_eq!(
            source,
            ModelSource::Ollama {
                name: "qwen2.5:0.5b".into()
            }
        );
    }

    #[test]
    fn a_numeric_tag_is_not_an_address() {
        // "llama3:8" is a model, not a server on port 8.
        assert!(remote_of("llama3:8").is_none());
        assert!(remote_of("127.0.0.1:1234").is_some());
        assert!(remote_of("localhost:1234").is_some());
    }

    #[test]
    fn a_spec_survives_the_round_trip() {
        for spec in [
            "ollama:qwen2.5:0.5b",
            "gguf:/models/qwen3-8b.gguf",
            "mlx:/models/qwen3-mlx",
            "hf:Qwen/Qwen3-8B",
        ] {
            let source = source_of(spec).expect("a source");
            assert_eq!(source.id(), spec);
        }
    }
}
