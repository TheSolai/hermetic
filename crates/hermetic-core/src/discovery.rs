//! Finding every model on the machine.
//!
//! A model can live in four places, and the terminal should not care which:
//!
//!   * Ollama's store, under `~/.ollama/models`.
//!   * A GGUF file, anywhere - a folder of them, most likely.
//!   * An MLX model: a directory of safetensors and a config.
//!   * A server that is already running, with models we cannot see.
//!
//! Discovery is deliberately dumb: it lists what is there and reads what it
//! can. Everything hard - reading a GGUF's shape, asking Ollama about a
//! model - happens in the background, because a terminal that waits for
//! ninety models to describe themselves is a terminal that does not start.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::registry::Registry;
use crate::signature::Signature;
use crate::source::{AdapterState, ModelDescriptor, ModelSource};

/// Where to look.
#[derive(Debug, Clone, Default)]
pub struct Sources {
    /// Ollama's store. `None` means do not look at it.
    pub ollama: Option<PathBuf>,
    /// Folders of GGUF files.
    pub gguf_dirs: Vec<PathBuf>,
    /// Folders of MLX models.
    pub mlx_dirs: Vec<PathBuf>,
    /// Servers that are already running.
    pub remote: Vec<(String, String)>,
}

/// The default places a model can be.
pub fn default_sources() -> Sources {
    Sources {
        ollama: Some(ollama_root()),
        gguf_dirs: Vec::new(),
        mlx_dirs: Vec::new(),
        remote: Vec::new(),
    }
}

/// Every model that can be found, from every source.
pub fn discover(sources: &Sources, registry: &Registry) -> Vec<ModelDescriptor> {
    let mut out: Vec<ModelDescriptor> = Vec::new();

    if let Some(root) = &sources.ollama {
        for mut descriptor in ollama_models(root) {
            // An Ollama model is served from the registry like any other
            // one, so it has an adapter state too. Without this the panel
            // said "unknown" for every model in Ollama's store, while the
            // models lying on disk said "ready".
            if let Some(signature) = &descriptor.signature {
                descriptor.adapter = adapter_state(registry, signature);
            }
            out.push(descriptor);
        }
    }
    for directory in &sources.gguf_dirs {
        for descriptor in gguf_models(directory, registry) {
            out.push(descriptor);
        }
    }
    for directory in &sources.mlx_dirs {
        for descriptor in mlx_models(directory, registry) {
            out.push(descriptor);
        }
    }
    for (base_url, model) in &sources.remote {
        let mut descriptor = ModelDescriptor::new(ModelSource::Remote {
            base_url: base_url.clone(),
            model: model.clone(),
        });
        descriptor.adapter = AdapterState::Unknown;
        out.push(descriptor);
    }

    out.sort_by(|a, b| a.label().cmp(&b.label()));
    out.dedup_by(|a, b| a.id() == b.id());
    out
}

// ── Ollama ─────────────────────────────────────────────────────────────────

/// Where Ollama keeps its models.
pub fn ollama_root() -> PathBuf {
    std::env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/"))
                .join(".ollama")
                .join("models")
        })
}

/// Every model in Ollama's store.
pub fn ollama_models(root: &Path) -> Vec<ModelDescriptor> {
    let mut out = Vec::new();
    for base in [
        root.join("manifests-v2").join("ollama.com"),
        root.join("manifests"),
    ] {
        if !base.exists() {
            continue;
        }
        walk_manifests(&base, &base, root, &mut out);
    }
    out.sort_by(|a, b| a.label().cmp(&b.label()));
    out.dedup_by(|a, b| a.id() == b.id());
    out
}

fn walk_manifests(
    base: &Path,
    dir: &Path,
    root: &Path,
    out: &mut Vec<ModelDescriptor>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_manifests(base, &path, root, out);
            continue;
        }
        let Ok(relative) = path.strip_prefix(base) else {
            continue;
        };
        let parts: Vec<String> = relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        let Some((tag, mut rest)) = parts.split_last().map(|(t, p)| (t.clone(), p.to_vec())) else {
            continue;
        };
        if rest.is_empty() {
            continue;
        }
        // `library` is Ollama's default namespace, and the registry host is
        // not part of the name.
        if rest[0] == "registry.ollama.ai" && rest.len() > 1 {
            rest.remove(0);
        }
        if rest[0] == "library" && rest.len() > 1 {
            rest.remove(0);
        }
        let name = if rest[0] == "hf.co" {
            format!("hf.co/{}:{tag}", rest[1..].join("/"))
        } else if rest.len() == 1 {
            format!("{}:{tag}", rest[0])
        } else {
            format!("{}/{tag}", rest.join("/"))
        };

        let Some((blob, size, is_gguf)) = manifest_weights(&path, root) else {
            continue;
        };
        let mut descriptor = ModelDescriptor::new(ModelSource::Ollama { name });
        descriptor.size_bytes = Some(size);
        if let Some(blob) = blob {
            // The shape is read from the GGUF header, which is fast, but not
            // free: a hundred models is a second of work. It is done in the
            // background.
            descriptor.signature = crate::gguf::signature_of(&blob).ok();
        }
        let _ = is_gguf;
        out.push(descriptor);
    }
}

/// The weights a manifest points at, and how big they are.
fn manifest_weights(manifest: &Path, root: &Path) -> Option<(Option<PathBuf>, u64, bool)> {
    let text = std::fs::read_to_string(manifest).ok()?;
    let json: Value = serde_json::from_str(&text).ok()?;
    let layers = json.get("layers")?.as_array()?;
    let mut total = 0u64;
    let mut model_blob = None;
    let mut tensors = false;
    for layer in layers {
        let size = layer.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
        let media = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
        if media.ends_with("image.model") {
            if let Some(digest) = layer.get("digest").and_then(|d| d.as_str()) {
                let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
                let candidate = root.join("blobs").join(format!("sha256-{hex}"));
                if candidate.exists() {
                    model_blob = Some(candidate);
                    total += size;
                }
            }
        } else if media.contains("tensor") {
            tensors = true;
            total += size;
        }
    }
    match model_blob {
        Some(blob) => {
            let is_gguf = is_gguf_file(&blob);
            Some((Some(blob), total, is_gguf))
        }
        // The weights are safetensors: Ollama keeps them as many small
        // tensors, and llama.cpp cannot read them.
        None if tensors => Some((None, total, false)),
        None => None,
    }
}

/// Is this file a GGUF?
pub fn is_gguf_file(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    match std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)) {
        Ok(()) => &magic == b"GGUF",
        Err(_) => false,
    }
}

// ── GGUF ───────────────────────────────────────────────────────────────────

/// Every GGUF file in a folder, and in the folders under it.
///
/// The walk has a depth limit and does not follow symbolic links. A link
/// that points at a parent directory would otherwise make this walk for
/// ever, and a folder of links is a normal thing to have.
pub fn gguf_models(directory: &Path, registry: &Registry) -> Vec<ModelDescriptor> {
    gguf_models_at(directory, registry, 0)
}

/// How deep the walk goes.
const MAX_DEPTH: usize = 8;

fn gguf_models_at(directory: &Path, registry: &Registry, depth: usize) -> Vec<ModelDescriptor> {
    let mut out = Vec::new();
    if depth > MAX_DEPTH {
        return out;
    }
    let Ok(entries) = std::fs::read_dir(directory) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // A link is followed for the file itself - a link to a GGUF is a
        // GGUF - but not for a directory, which can point back up.
        let is_link = std::fs::symlink_metadata(&path)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        if path.is_dir() {
            if is_link {
                continue;
            }
            out.extend(gguf_models_at(&path, registry, depth + 1));
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        let mut descriptor = ModelDescriptor::new(ModelSource::Gguf { path: path.clone() });
        descriptor.size_bytes = std::fs::metadata(&path).ok().map(|m| m.len());
        descriptor.signature = crate::gguf::signature_of(&path).ok();
        if let Some(signature) = &descriptor.signature {
            descriptor.adapter = adapter_state(registry, signature);
        }
        out.push(descriptor);
    }
    out
}

/// Whether an adapter for this shape is ready.
pub fn adapter_state(registry: &Registry, signature: &Signature) -> AdapterState {
    if registry.merged_for(signature).is_some() {
        AdapterState::Merged
    } else if registry.adapter_for(signature).is_some() {
        if registry.model_for(signature).is_some() {
            AdapterState::Ready
        } else {
            AdapterState::Missing
        }
    } else {
        AdapterState::Missing
    }
}

// ── MLX ────────────────────────────────────────────────────────────────────

/// Every MLX model in a folder: a directory with a config and safetensors.
pub fn mlx_models(directory: &Path, registry: &Registry) -> Vec<ModelDescriptor> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(directory) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if !is_mlx_model(&path) {
            continue;
        }
        let mut descriptor = ModelDescriptor::new(ModelSource::Mlx {
            path: path.clone(),
            repo: None,
        });
        descriptor.signature = mlx_signature(&path);
        if let Some(signature) = &descriptor.signature {
            descriptor.adapter = adapter_state(registry, signature);
        }
        out.push(descriptor);
    }
    out
}

/// Is this directory an MLX model?
pub fn is_mlx_model(directory: &Path) -> bool {
    if !directory.join("config.json").is_file() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(directory) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "safetensors")
    })
}

/// The shape of an MLX model, from its config.
pub fn mlx_signature(directory: &Path) -> Option<Signature> {
    let text = std::fs::read_to_string(directory.join("config.json")).ok()?;
    let config: Value = serde_json::from_str(&text).ok()?;
    Signature::from_hf_config(&config).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_without_a_config_is_not_an_mlx_model() {
        let directory = std::env::temp_dir().join(format!("hermetic-mlx-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        assert!(!is_mlx_model(&directory));
        std::fs::write(directory.join("config.json"), "{}").unwrap();
        assert!(!is_mlx_model(&directory));
        std::fs::write(directory.join("weights.safetensors"), "x").unwrap();
        assert!(is_mlx_model(&directory));
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn the_ollama_store_is_where_it_is_said_to_be() {
        let root = ollama_root();
        assert!(root.ends_with("models") || root.to_string_lossy().contains("ollama"));
    }
}
