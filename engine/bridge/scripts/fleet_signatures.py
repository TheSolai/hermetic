#!/usr/bin/env python3
"""Shape signatures for the models actually installed, from Ollama's manifest store.

An adapter is bound to one signature (arch + layer count + hidden size +
intermediate size + head geometry), because llama.cpp rejects an adapter whose
tensor shapes differ from the base model.
"""
import json
import struct
import sys
from pathlib import Path

MANIFESTS = Path.home() / ".ollama/models/manifests"
BLOBS = Path.home() / ".ollama/models" / "blobs"

# ── minimal GGUF metadata reader (header only, no tensor data) ────────────
GGUF_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 8: "s",
              9: "a", 10: "Q", 11: "q", 12: "d"}
_FMT = {"B": "<B", "b": "<b", "H": "<H", "h": "<h", "I": "<I", "i": "<i",
        "f": "<f", "?": "<?", "Q": "<Q", "q": "<q", "d": "<d"}


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, vtype):
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        (elem,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        et = GGUF_TYPES[elem]
        if et == "s":
            return [_read_str(f) for _ in range(count)]
        size = struct.calcsize(_FMT[et])
        raw = f.read(size * count)
        return list(struct.unpack(f"<{count}{et}", raw))
    fmt = _FMT[GGUF_TYPES[vtype]]
    (val,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    return val


def gguf_metadata(path):
    """Return {key: value} from a GGUF file's header."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("GGUF magic invalid")
        struct.unpack("<I", f.read(4))          # version
        struct.unpack("<Q", f.read(8))          # tensor count
        kv_count = struct.unpack("<Q", f.read(8))[0]
        meta = {}
        for _ in range(kv_count):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            meta[key] = _read_value(f, vtype)
        return meta


def resolve(name):
    """ollama model name -> manifest path"""
    name, _, tag = name.partition(":")
    tag = tag or "latest"
    parts = name.split("/")
    if parts[0] == "hf.co":
        cand = MANIFESTS.joinpath(*parts, tag)
    elif len(parts) == 1:
        cand = MANIFESTS / "registry.ollama.ai" / "library" / parts[0] / tag
    else:
        cand = MANIFESTS / "registry.ollama.ai" / parts[0] / parts[-1] / tag
    return cand if cand.exists() else None


def blob_for(manifest_path):
    try:
        data = json.loads(manifest_path.read_text())
    except Exception:
        return None
    layers = data.get("layers", [])
    if isinstance(layers, dict):
        layers = list(layers.values())
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        digest = layer.get("digest", "")
        if not digest:
            continue
        p = BLOBS / ("sha256-" + digest.replace("sha256:", "").replace("sha256-", ""))
        if p.exists():
            return p
    return None


def flat(v):
    """Collapse list-valued GGUF metadata into a hashable summary."""
    if isinstance(v, list):
        uniq = sorted({str(x) for x in v})
        return "[" + ",".join(uniq) + "]" if uniq else "[]"
    return v


def signature(path):
    meta = gguf_metadata(path)
    arch = meta.get("general.architecture", "?")
    a = f"{arch}."
    return {
        "arch": arch,
        "layers": flat(meta.get(a + "block_count")),
        "hidden": flat(meta.get(a + "embedding_length")),
        "heads": flat(meta.get(a + "attention.head_count")),
        "kv": flat(meta.get(a + "attention.head_count_kv")),
        "inter": flat(meta.get(a + "feed_forward_length")),
        "exp": flat(meta.get(a + "expert_count")),
        "name": meta.get("general.name", "?"),
        "gb": round(Path(path).stat().st_size / 1e9, 1),
    }


def main():
    names = [l.strip() for l in open("/tmp/models.txt") if l.strip()]
    rows, errors = [], []
    for name in names:
        mf = resolve(name)
        if not mf:
            errors.append((name, "manifest not found"))
            continue
        blob = blob_for(mf)
        if not blob:
            errors.append((name, "blob not found"))
            continue
        try:
            rows.append((name, signature(blob)))
        except Exception as e:
            errors.append((name, f"{type(e).__name__}: {e}"))

    print(f"{'model':<50} {'arch':<16} {'lay':>4} {'hid':>6} {'hd':>4} {'kv':>4} {'int':>7} {'exp':>5} {'GB':>6}")
    print("-" * 112)
    for name, s in sorted(rows, key=lambda r: r[0]):
        print(f"{name:<50} {str(s['arch']):<16} {str(s['layers']):>4} {str(s['hidden']):>6} "
              f"{str(s['heads']):>4} {str(s['kv']):>4} {str(s['inter']):>7} "
              f"{str(s['exp'] or '-'):>5} {s['gb']:>6}")

    sigs = {}
    for name, s in rows:
        key = (s["arch"], s["layers"], s["hidden"], s["inter"], s["exp"])
        sigs.setdefault(key, []).append(name)
    print(f"\n{len(rows)} models -> {len(sigs)} distinct shape signatures")
    print("\nsignatures shared by more than one installed model (one adapter covers all of them):")
    for key, group in sorted(sigs.items(), key=lambda kv: -len(kv[1])):
        if len(group) > 1:
            print(f"  {key}  <- {', '.join(sorted(group))}")
    if errors:
        print("\nnot scanned:")
        for n, e in errors:
            print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
