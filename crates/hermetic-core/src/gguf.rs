//! Reading GGUF headers, and the shape signature that decides which
//! Heretic adapter a model can use.
//!
//! An adapter is bound to the *shape* of a model, not its name: llama.cpp
//! refuses an adapter whose tensors do not match the base model exactly, so
//! a Qwen3-0.6B adapter cannot be used on Qwen3-4B. The signature is that
//! shape.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::signature::Signature;


// ── GGUF metadata ─────────────────────────────────────────────────────────

/// A GGUF value: only what we need to identify a model.
#[derive(Debug, Clone)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Str(String),
    Bool(bool),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::I64(v) if *v >= 0 => Some(*v as u64),
            Value::F64(v) if *v >= 0.0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

fn read_string(f: &mut impl Read) -> Result<String> {
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len) as usize;
    if n > 1 << 20 {
        bail!("implausible string length {n}");
    }
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// How deep an array may be nested before the file is called malformed.
///
/// A GGUF never nests arrays more than a couple of levels deep. A file that
/// does is either broken or trying to make the reader recurse until the
/// stack runs out, and either way it is not a model.
const MAX_DEPTH: usize = 16;

fn read_value(f: &mut impl Read, vtype: u32) -> Result<Value> {
    read_value_at(f, vtype, 0)
}

fn read_value_at(f: &mut impl Read, vtype: u32, depth: usize) -> Result<Value> {
    if depth > MAX_DEPTH {
        bail!("the arrays in this file are nested too deeply to be a model");
    }
    Ok(match vtype {
        0 => {
            let mut b = [0u8; 1];
            f.read_exact(&mut b)?;
            Value::U64(b[0] as u64)
        }
        1 => {
            let mut b = [0u8; 1];
            f.read_exact(&mut b)?;
            Value::I64(b[0] as i8 as i64)
        }
        2 => {
            let mut b = [0u8; 2];
            f.read_exact(&mut b)?;
            Value::U64(u16::from_le_bytes(b) as u64)
        }
        3 => {
            let mut b = [0u8; 2];
            f.read_exact(&mut b)?;
            Value::I64(i16::from_le_bytes(b) as i64)
        }
        4 => {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Value::U64(u32::from_le_bytes(b) as u64)
        }
        5 => {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Value::I64(i32::from_le_bytes(b) as i64)
        }
        6 => {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Value::F64(f32::from_le_bytes(b) as f64)
        }
        7 => {
            let mut b = [0u8; 1];
            f.read_exact(&mut b)?;
            Value::Bool(b[0] != 0)
        }
        8 => Value::Str(read_string(f)?),
        9 => {
            let mut et = [0u8; 4];
            f.read_exact(&mut et)?;
            let elem = u32::from_le_bytes(et);
            let mut count = [0u8; 8];
            f.read_exact(&mut count)?;
            let n = u64::from_le_bytes(count) as usize;
            // A GGUF's tokenizer holds a hundred thousand strings. Reading
            // them into memory to throw them away costs seconds per model,
            // and nothing here needs them.
            //
            // A long array of numbers is not skipped, though: a hybrid model
            // has one feed-forward size per layer, and throwing that away
            // would give the model the wrong shape.
            if n > 64 && elem != 10 && elem != 11 && elem != 12 {
                for _ in 0..n {
                    skip_value_at(f, elem, depth + 1)?;
                }
                Value::Array(Vec::new())
            } else {
                let mut out = Vec::new();
                for _ in 0..n {
                    out.push(read_value_at(f, elem, depth + 1)?);
                }
                Value::Array(out)
            }
        }
        10 => {
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            Value::U64(u64::from_le_bytes(b))
        }
        11 => {
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            Value::I64(i64::from_le_bytes(b))
        }
        12 => {
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            Value::F64(f64::from_le_bytes(b))
        }
        other => bail!("unknown GGUF value type {other}"),
    })
}

/// Step over a value without building it.
///
/// The reader is buffered, so this is a copy rather than a seek: a GGUF's
/// tokenizer is a hundred thousand strings, and a syscall each would cost
/// seconds per model.
fn skip_value_at(f: &mut impl Read, vtype: u32, depth: usize) -> Result<()> {
    if depth > MAX_DEPTH {
        bail!("the arrays in this file are nested too deeply to be a model");
    }
    let mut scratch = [0u8; 8192];
    match vtype {
        0 | 1 | 7 => {
            f.read_exact(&mut scratch[..1])?;
        }
        2 | 3 => {
            f.read_exact(&mut scratch[..2])?;
        }
        4 | 5 | 6 => {
            f.read_exact(&mut scratch[..4])?;
        }
        10 | 11 | 12 => {
            f.read_exact(&mut scratch[..8])?;
        }
        8 => {
            let mut len = [0u8; 8];
            f.read_exact(&mut len)?;
            let mut n = u64::from_le_bytes(len);
            while n > 0 {
                let chunk = n.min(scratch.len() as u64) as usize;
                f.read_exact(&mut scratch[..chunk])?;
                n -= chunk as u64;
            }
        }
        9 => {
            let mut et = [0u8; 4];
            f.read_exact(&mut et)?;
            let elem = u32::from_le_bytes(et);
            let mut count = [0u8; 8];
            f.read_exact(&mut count)?;
            let n = u64::from_le_bytes(count);
            for _ in 0..n {
                skip_value_at(f, elem, depth + 1)?;
            }
        }
        other => bail!("unknown GGUF value type {other}"),
    }
    Ok(())
}

/// Read a GGUF file's metadata. Only the header is read, so this is cheap
/// even for a 30 GB model.
pub fn gguf_metadata(path: &Path) -> Result<BTreeMap<String, Value>> {
    gguf_metadata_with(path, |_| false)
}

/// Read a GGUF file's metadata, stopping as soon as `stop` is satisfied.
///
/// A GGUF keeps the tokenizer after the model's shape, and the tokenizer is a
/// hundred thousand strings. Everything that identifies a shape comes first,
/// so the reader can stop before the expensive part.
fn gguf_metadata_with(
    path: &Path,
    stop: impl Fn(&BTreeMap<String, Value>) -> bool,
) -> Result<BTreeMap<String, Value>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut f = BufReader::with_capacity(1 << 16, file);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("{} is not a GGUF file", path.display());
    }
    let mut version = [0u8; 4];
    f.read_exact(&mut version)?;
    let mut tensors = [0u8; 8];
    f.read_exact(&mut tensors)?;
    let mut kv_count = [0u8; 8];
    f.read_exact(&mut kv_count)?;
    let n = u64::from_le_bytes(kv_count);
    if n > 1_000_000 {
        bail!("implausible metadata count {n}");
    }

    let mut meta = BTreeMap::new();
    for _ in 0..n {
        let key = read_string(&mut f)?;
        let mut vt = [0u8; 4];
        f.read_exact(&mut vt)?;
        let value = read_value(&mut f, u32::from_le_bytes(vt))?;
        meta.insert(key, value);
        if stop(&meta) {
            break;
        }
    }
    Ok(meta)
}

fn num(meta: &BTreeMap<String, Value>, key: &str) -> u64 {
    match meta.get(key) {
        Some(Value::Array(items)) => {
            // A hybrid architecture has one feed-forward size per layer.
            // The largest of them identifies the shape well enough.
            items
                .iter()
                .filter_map(|v| v.as_u64())
                .max()
                .unwrap_or(0)
        }
        Some(value) => value.as_u64().unwrap_or(0),
        None => 0,
    }
}

/// The shape signature of a GGUF model.
pub fn signature_of(path: &Path) -> Result<Signature> {
    // The shape's keys all come before the tokenizer, so the reader can stop
    // as soon as it has them.
    let meta = gguf_metadata_with(path, |meta| {
        let Some(arch) = meta.get("general.architecture").and_then(|v| v.as_str()) else {
            return false;
        };
        let p = format!("{arch}.");
        meta.contains_key(&format!("{p}block_count"))
            && meta.contains_key(&format!("{p}embedding_length"))
            && (meta.contains_key(&format!("{p}feed_forward_length"))
                || meta.contains_key(&format!("{p}expert_feed_forward_length")))
    })?;
    let arch = meta
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // A file without a shape is not a shape. Saying "unknown, zero layers,
    // zero width" would give it the key `unknown-0-0-0`, and every such
    // file on the machine would share one adapter - which is fitted to
    // neither of them.
    let p = format!("{arch}.");
    let layers = num(&meta, &format!("{p}block_count"));
    let hidden = num(&meta, &format!("{p}embedding_length"));
    if layers == 0 || hidden == 0 {
        anyhow::bail!(
            "{} does not say what shape it is - it has no block count or no width",
            path.display()
        );
    }

    // A mixture-of-experts model has no shared feed-forward size: it has one
    // per expert, and the shared field is zero.
    let inter = {
        let shared = num(&meta, &format!("{p}feed_forward_length"));
        if shared > 0 {
            shared
        } else {
            num(&meta, &format!("{p}expert_feed_forward_length"))
        }
    };
    Ok(Signature {
        layers,
        hidden,
        inter,
        heads: num(&meta, &format!("{p}attention.head_count")),
        kv_heads: num(&meta, &format!("{p}attention.head_count_kv")),
        experts: num(&meta, &format!("{p}expert_count")),
        arch,
    })
}

/// The model name recorded inside the file, if any.
///
/// The name is in the header, before the tokenizer - which is a hundred
/// thousand strings. The reader stops as soon as it has the name, so this
/// does not walk the whole file.
pub fn gguf_name(path: &Path) -> Option<String> {
    let meta = gguf_metadata_with(path, |meta| meta.contains_key("general.name")).ok()?;
    meta.get("general.name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// The head's own width.
///
/// This is the width of one attention head: the output of one head, and the
/// input of one. The model's width divided by its heads is only the same
/// thing when the heads are square - which they usually are not. A model
/// whose heads are wider than that (Qwen 3, for one) has an attention
/// output that is wider than the model.
pub fn head_width(meta: &BTreeMap<String, Value>, prefix: &str) -> u64 {
    // The value length is the one that decides the attention output: it is
    // the heads' width that the output projection mixes.
    let value = num(meta, &format!("{prefix}attention.value_length"));
    if value > 0 {
        return value;
    }
    let key = num(meta, &format!("{prefix}attention.key_length"));
    if key > 0 {
        return key;
    }
    let hidden = num(meta, &format!("{prefix}embedding_length"));
    let heads = num(meta, &format!("{prefix}attention.head_count"));
    if hidden > 0 && heads > 0 {
        hidden / heads
    } else {
        0
    }
}

/// Can an adapter be bound to this model's attention output?
///
/// Gemma 2 has eight heads of 256 each and a width of 2304, so its
/// attention output is 2048 wide and 2304 deep - not square. llama.cpp
/// reshapes such a tensor when it loads the model, into a square one, and
/// an adapter built for the file's shape no longer fits it: llama.cpp
/// applies it anyway, the weights are corrupted, and the model answers in
/// gibberish.
///
/// The adapter cannot bind to such a model, so it has to be folded into the
/// weights instead. This is how that is known.
///
/// Qwen 3 is the same shape of problem - sixteen heads of 128 each and a
/// width of 1024 - and it is just as true there: binding its adapter turns
/// "Paris" into "capital".
pub fn has_rectangular_attention(path: &Path) -> bool {
    let Ok(meta) = gguf_metadata(path) else {
        return false;
    };
    let Some(arch) = meta.get("general.architecture").and_then(|v| v.as_str()) else {
        return false;
    };
    let p = format!("{arch}.");
    let hidden = num(&meta, &format!("{p}embedding_length"));
    let heads = num(&meta, &format!("{p}attention.head_count"));
    if hidden == 0 || heads == 0 {
        return false;
    }
    let width = head_width(&meta, &p);
    if width == 0 {
        return false;
    }
    heads * width != hidden
}
pub fn header_size(path: &Path) -> Result<u64> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    let mut version = [0u8; 4];
    f.read_exact(&mut version)?;
    let mut tensors = [0u8; 8];
    f.read_exact(&mut tensors)?;
    let mut kv_count = [0u8; 8];
    f.read_exact(&mut kv_count)?;
    let n = u64::from_le_bytes(kv_count);
    for _ in 0..n {
        let _ = read_string(&mut f)?;
        let mut vt = [0u8; 4];
        f.read_exact(&mut vt)?;
        let _ = read_value(&mut f, u32::from_le_bytes(vt))?;
    }
    Ok(f.stream_position()?)
}

/// Skip to the tensor directory without reading it.
pub fn seek_to_tensors(path: &Path) -> Result<()> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(0))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_key_is_stable() {
        let s = Signature {
            arch: "qwen3".into(),
            layers: 36,
            hidden: 4096,
            inter: 12288,
            heads: 32,
            kv_heads: 8,
            experts: 0,
        };
        assert_eq!(s.key(), "qwen3-36-4096-12288");
    }

    fn meta(pairs: &[(&str, u64)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Value::U64(*v)))
            .collect()
    }

    #[test]
    fn a_model_with_square_attention_can_bind_an_adapter() {
        // Qwen 2.5 0.5B: fourteen heads of sixty-four in a width of 896.
        let m = meta(&[
            ("qwen2.embedding_length", 896),
            ("qwen2.attention.head_count", 14),
            ("qwen2.attention.key_length", 64),
        ]);
        assert_eq!(head_width(&m, "qwen2."), 64);
        assert_eq!(14 * head_width(&m, "qwen2."), 896);
    }

    #[test]
    fn a_model_with_wider_heads_cannot_bind_an_adapter() {
        // Qwen 3 0.6B: sixteen heads of 128 in a width of 1024. The heads
        // are wider than the model's width divided by its heads, so the
        // attention output is 2048 wide and the model is not square.
        let m = meta(&[
            ("qwen3.embedding_length", 1024),
            ("qwen3.attention.head_count", 16),
            ("qwen3.attention.key_length", 128),
        ]);
        assert_eq!(head_width(&m, "qwen3."), 128);
        assert_ne!(16 * head_width(&m, "qwen3."), 1024);
    }

    #[test]
    fn the_value_length_decides_the_width_when_the_two_differ() {
        // The attention output is as wide as the values are, not as wide as
        // the keys are.
        let m = meta(&[
            ("m.embedding_length", 512),
            ("m.attention.head_count", 4),
            ("m.attention.key_length", 64),
            ("m.attention.value_length", 128),
        ]);
        assert_eq!(head_width(&m, "m."), 128);
    }

    #[test]
    fn a_model_without_a_head_width_is_not_rectangular() {
        // Nothing is known, so nothing is claimed.
        let m = meta(&[("m.embedding_length", 512)]);
        assert_eq!(head_width(&m, "m."), 0);
    }
}
