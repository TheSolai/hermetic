# Design

## The shape of it

Upstream Heretic is a Python program. It is good. It is also a program that
wants to own your terminal: it prints a Rich console full of progress bars,
and it asks you questions with `questionary` and blocks until you answer.

That blocking is the sharp edge. Run Heretic unattended - from a script, from
a CI job, from a scheduler - and the first question it asks ends the run
with `EOFError`, because there is no TTY to read from. The obvious fix is
`--trial-index`, and that works, but it means giving up the one genuinely
useful interaction in the whole program: picking which trial to keep.

So the terminal keeps the interaction and moves it somewhere it can be
answered without a human.

There are three processes, and each one owns the thing it is good at:

```
  hermetic (Rust)                     hermetic-shim (Rust)
  ┌──────────────────────┐         ┌──────────────────────┐
  │ a real zsh in a PTY  │◀───────▶│ resolves the model   │
  │ vt100 grid           │  shell  │ finds the adapter    │
  │ the HERETIC panel    │  hook   │ llama-server + LoRA  │
  └──────────┬───────────┘         │ chats over HTTP      │
             │                     └──────────┬───────────┘
             │ NDJSON over a unix socket        │ spawns
             ▼                                  ▼
      ┌──────────────────────┐        ┌──────────────────────┐
      │ hermetic_bridge.py      │───────▶│ heretic (upstream,   │
      │ runs Heretic         │        │  unmodified)         │
      │ unchanged            │        │ ARA and every other  │
      └──────────────────────┘        │ modifier and scorer  │
                                      └──────────────────────┘
```

The engine is **not forked**. The bridge hooks five seams that upstream
already exposes (see `PROTOCOL.md`), which means every scorer, every
modifier, every shipped profile, the plugin system, the `reproduce.json`
machinery and the Grimoire keep working untouched. Upstream can be updated
with `git pull` in `engine/heretic`.

## Why a terminal with a shell inside it

The obvious first version was a terminal that *ran the engine*: it spawned
Heretic, showed its console, and answered its questions. That is a viewer,
not a terminal. There was nowhere to type, which is the one thing a terminal
has to have.

So the terminal now hosts a **shell** in a pseudo-terminal. The user types at
a real prompt, with their own rc files, their own history, their own line
editing - zsh's ZLE does the editing, because the keys are forwarded to the
PTY raw. The terminal reads the PTY's output through a `vt100::Parser` and
draws the resulting screen, so the colours and the cursor survive.

The shell's `ollama` is routed to the interceptor. Not by wrapping the binary:
by defining a shell function that hands `ollama run <model>` to
`hermetic-shim` and passes everything else through to the real `ollama`. The
user's own `ollama` still works; only `run` is intercepted.

## Why the interceptor is a separate binary

Because it runs **in the foreground of the user's shell**, and it has to be
able to take over the terminal: print what it is doing, serve a model, and
then run a chat loop that reads from the same stdin the user was typing into.

That is also why `llama-cli` cannot do the chat: it has no interactive mode
and no `--jinja`. The interceptor serves the model with `llama-server` and
talks to it over the OpenAI-compatible endpoint, which is where the token
streaming comes from.

## Why adapters are keyed by shape

llama.cpp refuses an adapter whose tensors do not line up with the base
model:

```
E llama_adapter_lora_init: failed to apply lora adapter:
tensor 'blk.0.attn_output.weight' has incorrect shape (hint: maybe wrong base model?)
```

So the adapter belongs to the *shape* of a model - architecture, layers,
hidden size, feed-forward size - and not to its name. One adapter then serves
every checkpoint and every quantisation of that shape, which is what makes
the cache worth having: a build takes minutes, and after that it is free.

Cross-size transplant is impossible and cross-quantisation is fine. The
terminal therefore never guesses: it reads the GGUF header, computes the
signature, and looks that up.

## Finding something to build from

A GGUF cannot be abliterated - Heretic needs the original weights in
safetensors. Ollama does not keep those: it stores a GGUF, or, for the MLX
builds, a pile of per-tensor blobs.

So when a shape has no adapter yet, the terminal looks for a Hugging Face
repo with **exactly** that shape, and runs Heretic on that. The shape has to
match to the digit, because the adapter will not bind otherwise. The search
is the hub's own API, filtered by `config.json`.

When Ollama stores a model as safetensors and no GGUF of that shape is
cached, the terminal says so rather than pretending: llama.cpp cannot read
safetensors.

## Which trial is kept

Heretic optimises an ablation: it tries a number of ways to remove a model's
refusals, scores each one, and keeps the best. "Best" is not obvious.

Upstream sorts its Pareto front by the scorers one after another, so the trial
that refuses **not at all** comes first - whatever it has cost the model. On
gemma-2-2b that is the difference between a KL divergence of 0.08 and one of
8.79, and the second one does not speak any more: it answers in gibberish. The
scorers rate a destroyed model as Pareto-optimal, because it really does
refuse the least.

So the bridge does not take upstream's first trial. It takes the trial with the
**lowest total score** - the sum of the scorers, all of which are minimised.
That is the one that refuses less *and* keeps the model closest to what it
was. The same choice is made for the default answer, so it holds whether the
client picks a choice or the engine has to fall back to one.

Two things make this stick:

* The bridge asks upstream's own question ("Which trial do you want to use?")
  and answers it itself. Upstream's control flow is not forked.
* A study remembers the settings of the run that created it, and restores them
  when it is resumed - including `trial_index`, the trial to keep. A study
  created by a run with the trial pinned therefore pins it for good, and the
  question is never asked again. The bridge drops that pin when a study is
  resumed: the pin belongs to the run that asked for it, not to the study.

## Tabs

Every tab is a shell in a pseudo-terminal and, once a model is loaded in it,
a model of its own. The tab is named after the model, so the strip at the top
of the panel says what is open:

```
 qwen2.5:0.5b   qwen3-0.6b-stock-…   NEW
```

The interceptor needs to know which tab asked for it, so the shell is given
`HERMETIC_TAB` in its environment and the interceptor sends that back with
the model it is serving.

## Why the panel is on the right, and why it is narrow

The shell is the thing the user is working in. It gets the space. The panel
is a dashboard: the model, its shape, its adapter, where it is served, what
the engine is doing.

The panel's width is not arbitrary: the shell's pseudo-terminal is sized to
exactly the pane it is drawn in. When it is not - as in the first version,
where the pseudo-terminal was sized to the whole window - the shell draws
lines that are wider than the pane and they are cut off at the panel. A
terminal that clips its own output is worse than no terminal.

## Why a separate process, not a library

Heretic is AGPL-3.0-or-later. Anything that links it inherits that. This
project calls it as a separate process, so the licence boundary is a process
boundary - the same trick that keeps the Meridian pipeline (Apache-2.0) clean
of the abliteration code it drives.

It also means the terminal can be written in Rust while the engine stays
Python, which is the only way to get a real vt100 grid without reimplementing
Heretic's Rich console.

## Why the terminal listens and the bridge connects

The obvious arrangement is the other way round: the bridge binds the socket
and the terminal connects.

But the bridge has to *answer* a question, and it does that by blocking on
the socket. If the terminal connects to the bridge, then the moment the
terminal dies the engine gets a broken pipe - which is fine - but the moment
the engine is slow to start, the terminal has to retry a connect against a
socket that does not exist yet.

Listening is the forgiving direction. The terminal binds first, the engine
connects when it is ready, and the engine's own retry loop handles the
startup gap. A slow Python import costs nothing.

## The question/answer protocol

The engine asks questions. The terminal answers them. The interesting part is
what an answer *is*.

Upstream's `questionary.select` returns the chosen **object** - an Optuna
`FrozenTrial` in the important case - and Heretic then reads
`trial.user_attrs["scores"]` off it.

The bridge has to send the choices to the terminal over JSON, and a
`FrozenTrial` does not survive that trip. What arrives at the terminal is a
projection: a title, and a string rendering of the value.

So the terminal sends back an **index**, and the bridge resolves it against
the original objects it kept to one side.

Getting this wrong is quiet and then loud. The first version resolved the
index against the JSON payload, so Heretic received a string where it
expected a `FrozenTrial`:

```
AttributeError: 'str' object has no attribute 'user_attrs'
```

The run had already produced good scores (baseline 48/100, trial 4/100) when
it died. The fix is to keep `objects` next to the payload and resolve
against that.

## The loop that never ends

Upstream's outer loop asks the same question again after every finished
action - "Which trial do you want to use?" - because you may want to pick a
different one. A headless run that always answers "the first one" therefore
saves the model over and over, forever.

The exit signal is an **empty** answer. So the terminal answers the first
question with an index, and every repeat with `""`. That is upstream's own
way of saying "stop", and it is the only thing that ends the loop.

## The false success

The most instructive bug in this project was not a crash. It was a run that
reported success and had done nothing at all.

Two things had to go wrong together:

1. The client passed `--config profiles/fast-test.toml` - a **relative**
   path. The engine's working directory is `engine/`, so the path resolved
   to `engine/profiles/fast-test.toml`, which does not exist. Heretic raises
   `SystemExit("heretic: configuration file ... does not exist")`.
2. The bridge mapped the exit code with `e.code if isinstance(e.code, int)
   else 0`. That `SystemExit` carries a **string**, not an int, so the fatal
   error was reported as **exit code 0**.

The terminal dutifully reported "engine finished with exit code 0". Nothing
had run. No model was loaded, no scores were computed, no output directory
was created.

Three fixes, because one would not have been enough:

- The client resolves `--config` to an absolute path before handing it to
  the engine. The engine's cwd is not the client's cwd, and a path that
  crosses that boundary has to be made absolute first.
- The bridge treats a non-integer `SystemExit` as an error: exit code 1 and
  an `error` event. A string exit code is a message, not a status.
- The headless mode echoes the engine's stdout and stderr. Before this, the
  engine's output was discarded, so there was nowhere for the failure to
  show up. A run that cannot fail silently is a run that can be trusted.

The general lesson: **exit code 0 is a claim, not a fact.** A harness that
reports success has to be able to distinguish "the work happened" from "the
process ended".

## Two kinds of output

The engine writes two things, and they need different handling.

**Console output** goes to stdout and is a Rich console with cursor movement,
progress bars and colours. It is not a stream of lines. The terminal feeds
it into a `vt100::Parser`, which interprets the escape sequences and holds a
screen grid. The terminal then draws that grid. This is why the output looks
right: it is not being re-rendered, it is being replayed.

**Structured events** go over the socket. They are what the terminal actually
uses to build its panel: the phase, the scores, the trial history.

The stdout is also mirrored onto the socket as `log` events, so a client can
follow along without parsing terminal escapes. Rich markup is stripped from
those events, because a client that has to understand `[cyan]...[/]` has to
know Rich's dialect.

## Reading a model's shape

The shape signature decides which adapter applies, so it has to be read
quickly and it has to be right.

Two things made it slow. A GGUF's metadata ends with the tokenizer - a
hundred thousand strings - and reading them costs a quarter of a second per
model. The reader now stops as soon as it has the shape: every key that
identifies a model comes before the tokenizer.

And a mixture-of-experts model has no shared feed-forward size: it has one
per expert, and the shared field is zero. The real number is in
`expert_feed_forward_length`. A hybrid architecture has an *array* of
feed-forward sizes, one per layer, and the largest of them identifies it.

Getting this wrong means the wrong adapter is looked up, and the adapter
does not bind.

## Ready means ready

`llama-server` answers `/health` **before** the weights are in memory. A
client that waits for `/health` will send a prompt to a model that is not
there yet.

The only honest signal is the log line the server prints when it is done:

```
I srv  llama_server: model loaded
```

## What is deliberately not here

- **No plugin system.** The terminal is a terminal. Everything interesting
  happens in the engine, where upstream put it.
- **No configuration of Heretic itself.** The engine's own TOML files are
  used as they are, and `profiles/` holds a couple for convenience.
- **No closed models.** Apple Intelligence, Claude and GPT cannot have their
  weights modified. A prompt-level policy for them is possible and is
  labelled as such, but it is not the product.

## Why there is a core crate

The first version of this terminal was written against one runtime. It
resolved a model from Ollama's store, converted it to a GGUF, served it with
llama-server and bound the adapter to it. That works, and it works well -
but the moment a user asks about an MLX model, or a GGUF they already have,
or a server they are running themselves, the code has nowhere to go.

So there is now a crate in the middle, `hermetic-core`, and it knows two
things:

* **where the weights are** - `ModelSource`, and
* **how a model is served** - `ModelRuntime`.

Everything else is written against those. The terminal asks for a list of
models and gets descriptors from every source at once; the interceptor picks
a runtime and asks it to serve; the chat client talks to whatever endpoint
came back.

### Why the chat API is the seam

Every runtime that can be talked to speaks the OpenAI chat API. llama.cpp
does, `mlx_lm.server` does, LM Studio does, vLLM does, Ollama does. So the
terminal has one chat client, `hermetic_core::chat::Chat`, and the runtime
behind the endpoint is the runtime's business.

This is the whole trick. It is why adding a backend is a small job: a
runtime that cannot change weights only has to answer the OpenAI API, and
the terminal labels it `prompt policy` and moves on.

### Why the shape drives everything

An adapter is bound to a *shape*, not to a model. That was true from the
start, and it is what makes one adapter serve every quantisation of a
model.

It is also what makes the search for something to build from work. A model's
name is a lie often enough to be useless: `deepseek-r1:8b` is a `Qwen3-8B`.
The shape is not a lie, and neither is the parameter count - Ollama reports
it exactly, and the hub reports it for every repository. So the donor search
is driven by the shape and the count, and the name is only ever a hint.

### Why a runtime says what it can do

`ModelRuntime::modification` is the answer to "can this model be
hereticed?", and there are three answers:

| answer | what happens |
|---|---|
| `Native` | the adapter is bound at runtime |
| `ByConversion` | the weights are turned into something the runtime can bind one to |
| `PromptOnly` | the weights cannot be changed, and the terminal says so |

A runtime that cannot change weights is not a failure. It is a fact, and the
terminal reports it rather than quietly serving something else.
