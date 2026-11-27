#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Merge a Heretic PEFT adapter into the base weights.

Heretic produces a PEFT adapter. llama.cpp can bind such an adapter to a GGUF
at runtime - that is the fast path - but a LoRA adapter has to match the base
model's shape to the digit, and llama.cpp cannot apply a LoRA to every
architecture it can load.

Merging removes both restrictions: the adapter is folded into the weights and
what comes out is an ordinary model, in any architecture.

The merged weights are written as safetensors; `convert_hf_to_gguf.py` turns
them into a GGUF afterwards.

Usage:
    python merge_adapter.py --base <hf repo or path> --adapter <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="the base model: a Hugging Face repo id or a directory")
    parser.add_argument("--adapter", required=True, help="the directory Heretic wrote the adapter to")
    parser.add_argument("--out", required=True, help="where to write the merged model")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype, torch.bfloat16)

    print(f"[merge] base    {args.base}", flush=True)
    print(f"[merge] adapter {args.adapter}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.device == "mps":
        model = model.to("mps")
    elif args.device == "cuda":
        model = model.to("cuda")

    print("[merge] applying the adapter", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)

    print("[merge] merging into the weights", flush=True)
    model = model.merge_and_unload()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[merge] writing {out}", flush=True)
    model.save_pretrained(out, safe_serialization=True)

    try:
        AutoTokenizer.from_pretrained(
            args.base, trust_remote_code=args.trust_remote_code
        ).save_pretrained(out)
    except Exception as error:  # noqa: BLE001 - the tokenizer is not always needed
        print(f"[merge] no tokenizer saved: {error}", flush=True)

    # The fast tokenizer writes `tokenizer.json` and stops there. Some
    # converters - Gemma's among them - want the SentencePiece file itself,
    # and without it the conversion stops with "File not found:
    # tokenizer.model". So the files the repository carries are copied across.
    for name in TOKENIZER_FILES:
        if (out / name).exists():
            continue
        source = _tokenizer_file(args.base, name, args.trust_remote_code)
        if source is None:
            continue
        shutil.copy(source, out / name)
        print(f"[merge] copied {name}", flush=True)

    print("[merge] done", flush=True)
    return 0


# The files a tokenizer is made of, beyond the ones `save_pretrained` writes.
TOKENIZER_FILES = (
    "tokenizer.model",
    "spiece.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
)


def _tokenizer_file(base: str, name: str, trust_remote_code: bool):
    """Find one tokenizer file, in a local directory or in a repository."""
    local = Path(base)
    if local.is_dir():
        candidate = local / name
        return candidate if candidate.is_file() else None

    try:
        from huggingface_hub import hf_hub_download
    except Exception:  # noqa: BLE001
        return None
    try:
        return Path(
            hf_hub_download(
                repo_id=base,
                filename=name,
            )
        )
    except Exception as error:  # noqa: BLE001 - the file is simply not there
        print(f"[merge] no {name}: {error}", flush=True)
        return None


if __name__ == "__main__":
    raise SystemExit(main())
