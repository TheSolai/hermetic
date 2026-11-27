# Runtimes

How each runtime is supported, and what each one cannot do.

The rule the whole design follows: **every runtime that can be talked to
exposes the OpenAI chat API**, so the terminal has one chat client and the
runtime behind the endpoint is the runtime's business. What the terminal
cares about is one thing only - whether the weights can be changed.

## The two answers

| answer | what it means | when |
|---|---|---|
| `hereticed` | the weights were changed and the model is served with the adapter bound in | the runtime can bind an adapter, or the weights can be converted into something it can |
| `prompt policy` | the weights could not be changed, so a system prompt asks the model not to refuse | the runtime holds the weights and will not change them |
| `as it is` | the model is served untouched | on purpose |

The third is rare and deliberate: the terminal does not quietly serve an
unmodified model when the user asked for a hereticed one. It says so.

## llama.cpp

**The runtime that does the work.** It takes a base GGUF and a LoRA adapter
and applies one to the other.

```
llama-server -m model.gguf --lora-scaled adapter.gguf:1.0 --port 50473
```

The adapter binds only if the shapes match to the digit, which is why
adapters are keyed by shape.

* **Modification:** native. The adapter is applied at full strength, which
  is what Heretic optimised.
* **Readiness:** the log line `model loaded`. Never `/health` - the server
  answers `/health` before the weights are in memory.
* **API:** OpenAI. `/v1/chat/completions`, with server-sent events.

### What needs converting

| source | what has to happen |
|---|---|
| a GGUF file | nothing |
| an Ollama model stored as a GGUF | the blob is hard-linked into the cache, so one adapter serves every quantisation of that shape |
| an Ollama model stored as safetensors (the MLX builds) | the weights are converted to a GGUF - or the adapter is folded into them and the result is served as an ordinary GGUF |
| an MLX model | the adapter is folded into the safetensors |

## Ollama

**Two things at once, and the terminal treats them as two things.**

*As a source*, Ollama's store is just another place weights live. The
terminal reads the blobs, hard-links them into the cache, and serves them
with llama.cpp and the adapter bound in. That is how an Ollama model gets
hereticed - Ollama's own engine is not involved.

*As a runtime*, Ollama's engine is fast and needs no conversion, but it
cannot change weights. When a model cannot be hereticed, Ollama serves it -
through its OpenAI-compatible API, so the terminal's one chat client still
works.

```
ollama run qwen2.5:0.5b   →   HERETICAL resolves it, binds the adapter,
                             and serves it with llama.cpp
```

### What Ollama knows that a GGUF header does not

Ollama's `/api/show` answers with more than the shape:

| field | what it is for |
|---|---|
| `general.architecture` | the family: `qwen3`, `llama`, `gemma2` |
| `general.parameter_count` | the exact number of parameters - the key to finding the original weights of a renamed model |
| `general.size_label` | `8B`, `30B-A3B` |
| `general.basename` | the family it came from |

`general.parameter_count` is what makes the donor search work for a
distilled model: `deepseek-r1:8b` reports `8190735360`, and so does
`Qwen/Qwen3-8B` on the hub. Nothing in the name says so.

## MLX

Apple's framework for models on Apple Silicon. An MLX model is a
*directory*: a `config.json` and one or more `.safetensors` files. There is
no GGUF, and llama.cpp cannot read it.

* **Modification:** native, in the same sense as llama.cpp - safetensors are
  writable, so the adapter can be folded into them.
* **Runtime:** `mlx_lm.server`, which serves the OpenAI API at
  `http://127.0.0.1:8080/v1`.
* **Shape:** from the model's own `config.json`.

## Any OpenAI-compatible server

LM Studio, vLLM, Jan, `text-generation-webui`, `mlx_lm.server`, a cloud
API - anything that answers `POST /v1/chat/completions`.

* **Modification:** none. The server holds the weights and will not change
  them, so the terminal uses a prompt-level policy and says so.
* **This is the floor.** A runtime that does not exist as a specific
  adapter still works: point the terminal at a base URL and a model name.

## The workarounds, and why they exist

| problem | what the terminal does |
|---|---|
| the model is renamed, so its name is no use | the donor search is driven by the shape and the parameter count |
| the repository is gated | the hub's own `gated` field says so, and `HF_TOKEN` unlocks it |
| a download answers `401` | that means "gated" *or* "does not exist", so the repository's record is asked instead of guessing |
| a re-upload of quantised weights | filtered out: Heretic needs the original tensors, and a GGUF repo does not have them |
| the weights are safetensors | converted to a GGUF, or the adapter is folded into them |
| llama.cpp refuses the adapter | the adapter is folded into the weights instead, and the merged model is served |
| the server answers `/health` before it is ready | the log line `model loaded` is the only readiness signal that counts |

## What cannot be done

Only local, open-weight models can have their weights modified. A closed
API cannot, and the terminal says so rather than pretending.

An adapter can only be built for a model whose original weights can be
found - a repository with exactly the same shape. A model whose shape nobody
publishes cannot be abliterated, and the terminal says so rather than
serving it as if it had been.

Some families - Meta's Llama, Google's gated Gemma - will not serve their
weights without a Hugging Face token. Export `HF_TOKEN` and the terminal
uses it, both to read those repositories and to download the weights.
