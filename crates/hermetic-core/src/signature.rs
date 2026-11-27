//! The shape of a model, as far as an adapter cares.
//!
//! Two models with the same shape can share an adapter; two models with
//! different shapes never can. llama.cpp refuses to bind a LoRA whose
//! tensors do not line up with the base model to the digit, so the shape is
//! the key everything else is keyed by.
//!
//! The shape is read from three different places - a GGUF header, a
//! Hugging Face `config.json`, and Ollama's own metadata - and they do not
//! always use the same names for the same thing. This module is the one
//! place that knows how to read all three.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::gguf;

/// The shape of a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// The architecture, as llama.cpp calls it: `qwen3`, `llama`, `gemma2`.
    pub arch: String,
    /// How many layers.
    pub layers: u64,
    /// The width of the model.
    pub hidden: u64,
    /// The width of the feed-forward part.
    pub inter: u64,
    /// How many attention heads.
    pub heads: u64,
    /// How many key-value heads.
    pub kv_heads: u64,
    /// How many experts, for a mixture-of-experts model.
    pub experts: u64,
}

impl Signature {
    /// A filesystem-safe key for this shape: `qwen3-36-4096-12288`.
    pub fn key(&self) -> String {
        format!(
            "{}-{}-{}-{}",
            self.arch, self.layers, self.hidden, self.inter
        )
    }

    /// A human-readable description.
    pub fn describe(&self) -> String {
        let mut text = format!(
            "{} · {} layers · hidden {} · ffn {}",
            self.arch, self.layers, self.hidden, self.inter
        );
        if self.experts > 0 {
            text.push_str(&format!(" · {} experts", self.experts));
        }
        text
    }

    /// Do these two shapes match, as far as an adapter cares?
    ///
    /// The direction matters. The adapter is built from the *donor* and bound
    /// to the *target*, so the donor may have fewer layers than the target -
    /// a model that predicts more than one token at a time carries an extra
    /// block for it, the MTP layer, and llama.cpp counts that block. The
    /// adapter has no tensors for it, and llama.cpp binds the adapter anyway.
    /// The other way round would mean an adapter with tensors the model does
    /// not have, which llama.cpp refuses.
    pub fn matches(&self, other: &Signature) -> bool {
        let layers_fit = other.layers == self.layers
            || (other.layers > self.layers && other.layers - self.layers == 1);
        layers_fit
            && self.hidden == other.hidden
            && self.inter == other.inter
    }

    /// The shape, read from a GGUF header.
    pub fn from_gguf(path: &Path) -> Result<Signature> {
        gguf::signature_of(path)
            .with_context(|| format!("read the GGUF header of {}", path.display()))
    }

    /// The shape, read from a Hugging Face `config.json`.
    pub fn from_hf_config(config: &Value) -> Result<Signature> {
        // A model that is more than a language model keeps its shape in a
        // nested config: `Qwen3.5-35B-A3B` puts everything in `text_config`.
        let config = if config.get("num_hidden_layers").is_some() {
            config
        } else if let Some(text) = config.get("text_config") {
            text
        } else {
            config
        };
        let arch = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let get = |key: &str| config.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
        // A mixture-of-experts model has one feed-forward size per expert.
        let inter = {
            let shared = get("intermediate_size");
            if shared > 0 {
                shared
            } else {
                get("moe_intermediate_size")
            }
        };
        let experts = config
            .get("num_experts")
            .or_else(|| config.get("num_local_experts"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let signature = Signature {
            arch,
            layers: get("num_hidden_layers"),
            hidden: get("hidden_size"),
            inter,
            heads: get("num_attention_heads"),
            kv_heads: get("num_key_value_heads"),
            experts,
        };
        if signature.layers == 0 || signature.hidden == 0 {
            bail!("the config does not describe a model shape");
        }
        Ok(signature)
    }

    /// The shape, read from Ollama's own metadata.
    ///
    /// Ollama names its keys after the architecture, so `qwen3.block_count`
    /// and `qwen3.embedding_length`.
    pub fn from_ollama_info(info: &Value) -> Result<Signature> {
        let arch = info
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let get = |key: &str| -> u64 {
            info.get(&format!("{arch}.{key}"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        };
        let inter = {
            let shared = get("feed_forward_length");
            if shared > 0 {
                shared
            } else {
                get("expert_feed_forward_length")
            }
        };
        let signature = Signature {
            arch: arch.clone(),
            layers: get("block_count"),
            hidden: get("embedding_length"),
            inter,
            heads: get("attention.head_count"),
            kv_heads: get("attention.head_count_kv"),
            experts: get("expert_count"),
        };
        if signature.layers == 0 || signature.hidden == 0 {
            bail!("Ollama did not report a shape");
        }
        Ok(signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_hugging_face_config_gives_the_shape() {
        let config = json!({
            "model_type": "qwen3",
            "num_hidden_layers": 36,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        });
        let shape = Signature::from_hf_config(&config).expect("a shape");
        assert_eq!(shape.key(), "qwen3-36-4096-12288");
        assert_eq!(shape.heads, 32);
        assert_eq!(shape.kv_heads, 8);
    }

    #[test]
    fn a_mixture_of_experts_config_gives_the_expert_size() {
        let config = json!({
            "model_type": "qwen3",
            "num_hidden_layers": 48,
            "hidden_size": 2048,
            "intermediate_size": 0,
            "moe_intermediate_size": 768,
            "num_experts": 128,
        });
        let shape = Signature::from_hf_config(&config).expect("a shape");
        assert_eq!(shape.inter, 768);
        assert_eq!(shape.experts, 128);
    }

    #[test]
    fn ollama_metadata_gives_the_shape() {
        let info = json!({
            "general.architecture": "qwen3",
            "qwen3.block_count": 36,
            "qwen3.embedding_length": 4096,
            "qwen3.feed_forward_length": 12288,
            "qwen3.attention.head_count": 32,
            "qwen3.attention.head_count_kv": 8,
        });
        let shape = Signature::from_ollama_info(&info).expect("a shape");
        assert_eq!(shape.key(), "qwen3-36-4096-12288");
    }

    #[test]
    fn two_shapes_match_only_when_the_numbers_do() {
        let a = Signature {
            arch: "qwen3".into(),
            layers: 36,
            hidden: 4096,
            inter: 12288,
            heads: 32,
            kv_heads: 8,
            experts: 0,
        };
        let mut b = a.clone();
        assert!(a.matches(&b));
        b.layers = 28;
        assert!(!a.matches(&b));
    }
}
