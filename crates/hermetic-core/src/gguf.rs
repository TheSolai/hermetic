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

fn read_value(f: &mut impl Read, vtype: u32) -> Result<Value> {
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
            if n > 64 {
                for _ in 0..n {
                    skip_value(f, elem)?;
                }
                Value::Array(Vec::new())
            } else {
                let mut out = Vec::new();
                for _ in 0..n {
                    out.push(read_value(f, elem)?);
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
fn skip_value(f: &mut impl Read, vtype: u32) -> Result<()> {
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
                skip_value(f, elem)?;
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
    let p = format!("{arch}.");
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
        layers: num(&meta, &format!("{p}block_count")),
        hidden: num(&meta, &format!("{p}embedding_length")),
        inter,
        heads: num(&meta, &format!("{p}attention.head_count")),
        kv_heads: num(&meta, &format!("{p}attention.head_count_kv")),
        experts: num(&meta, &format!("{p}expert_count")),
        arch,
    })
}

/// The model name recorded inside the file, if any.
pub fn gguf_name(path: &Path) -> Option<String> {
    let meta = gguf_metadata(path).ok()?;
    meta.get("general.name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Read only the header — used to skip the tensor directory quickly.
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
}
