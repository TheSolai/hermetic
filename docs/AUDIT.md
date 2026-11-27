# What works, what did not, and what changed

This is the audit of HERMETIC's terminal and engine: every feature, whether it
works, and what was done about the ones that did not.

It was done by reading every line of `hermetic-terminal`, `hermetic-core` and
`hermetic-protocol`, and by running the terminal. Every finding below names the
file and the line it was in, and every one of them has either been fixed or is
listed as still open.

Two rules ran through the whole thing:

1. **A model that says it is hereticed must be hereticed.** If the weights were
   not changed, the terminal says so. This is the rule that found the most
   bugs.
2. **A number on its own says nothing.** A refusal count means nothing next to
   the untouched model's refusal count. Every score is reported next to its
   baseline, and every claim about what happened is backed by what actually
   happened.

---

## The terminal

| Feature | Where | Verdict |
|---|---|---|
| Catalogue discovery (Ollama, GGUF folders, MLX folders, running servers) | `catalogue.rs:54`, `discovery.rs:47` | works |
| The catalogue is read off the UI thread | `main.rs:391` | works |
| The catalogue is read again after a build finishes | `main.rs:414` | **fixed** - it was set but never read |
| Filtering the model list | `app.rs:220` | works |
| Keyboard selection (`Alt-M`, arrows, `PgUp`/`PgDn`, `Enter`) | `main.rs:521` | works |
| Mouse selection (click = tab, alt-click = window) | `ui.rs:306`, `main.rs:671` | works |
| Left and right panel | `ui.rs:50` | works |
| Long names are cut, never wrapped | `ui.rs:487` | **fixed** - the state column could push a row past the panel |
| The shell pane and its PTY | `ui.rs:120`, `shell.rs:28` | works |
| Scrolling the shell pane | `ui.rs:124` | **open** - there is no scrollback; the wheel scrolls the model list |
| Resizing (grid, PTY, layout) | `main.rs:432` | works |
| The panel below 94 columns | `ui.rs:61` | **open** - below that width there is no panel at all |
| Starting a model | `main.rs:707` | works |
| A model that is not on the machine yet is fetched first | `hermetic-shim.rs:302` | **fixed** - it used to be served unchanged |
| Stopping a served model | - | **open** - there is no control for it |
| Two terminals never share a socket or a run directory | `main.rs:260`, `shell.rs:219` | **fixed** - the socket was in `/tmp`, which is world-writable |
| The "serving" event names the right tab | `app.rs:376` | **fixed** - an unknown tab renamed whatever was in front |
| The probe score is put next to the right model | `app.rs:453` | **fixed** - it went to every tab that had none |
| The engine's questions are shown | `ui.rs:216` | **fixed** - they were answered in secret |
| A failed answer is reported as a failure | `main.rs:418` | **fixed** - it said "answered the engine" either way |
| Build progress, scores and trials | `app.rs:403` | works |
| The log of what happened is visible | `ui.rs:232` | **fixed** - the log was collected and never drawn |
| A shell that has ended is noticed | `main.rs:434` | **fixed** - the pane just froze |
| The panel's engine lines keep their state and scores | `app.rs:540` | **fixed** - the whole line was cut, taking the scores with it |
| `--check` | `main.rs:261` | **fixed** - it printed five paths; it is now a real check |
| Clean shutdown, including on a signal | `main.rs:344` | **fixed** - the socket and the run directory were left behind |
| `Alt-0` | `main.rs:512` | **fixed** - it underflowed |
| A model name with an escape sequence in it | `lib.rs:79` | **fixed** - the terminal would have run it |
| Writing to a shell whose reader is not reading | `shell.rs:96` | **fixed** - the event loop could block for ever |
| Writing to an engine that is not reading | `channel.rs:55` | **fixed** - same |
| The status line's title | `ui.rs:508` | **fixed** - a long name pushed the key hints off the screen |

## The engine

| Capability | Where | Verdict |
|---|---|---|
| Finding a model on the hub | `hub.rs:959` | works |
| Telling a busy hub from a model that is not there | `hub.rs:106` | **fixed** - a 429 read as "no model found" |
| Choosing a donor that is the model the weights are | `hub.rs:605` | **fixed** - it chose the first shape match, base or instruct |
| The adapter is fitted to the weights being served | `hub.rs:605` | **fixed** - it was fitted to the base model |
| An adapter is not bound to a model whose attention is not square | `gguf.rs:316` | works - now also true for Qwen 3 |
| The GGUF that holds a folded adapter says so | `registry.rs:502` | **fixed** - it was written in place, and a half-written one was later trusted |
| The study cannot pin the trial or the trial count | `bridge: _install_settings_seam` | works |
| The trial kept is the one with the lowest total score | `bridge:337` | works |
| A finished study is converted, not re-optimised | `registry.rs:405` | works |
| The probe is measured against the untouched model | `bridge: _capture_probe` | works |
| A GGUF that does not say what shape it is | `gguf.rs:257` | **fixed** - it became the key `unknown-0-0-0` |
| A GGUF with more than 64 numbers in an array | `gguf.rs:105` | **fixed** - the array was thrown away |
| A GGUF with arrays nested too deeply | `gguf.rs:67` | **fixed** - the reader could run out of stack |
| A model that cannot bind an adapter | `llamacpp.rs:142` | **fixed** - llama.cpp's own warning was not recognised |
| MLX claiming to have hereticed a model | `mlx.rs:217` | **fixed** - it never applied anything |
| A server that fails | `http.rs:239` | **fixed** - the terminal showed an empty answer |
| A server that only answers `GET` | `openai.rs:49` | **fixed** - it posted, and was reported as not running |
| A question that could not be asked | `chat.rs:88` | **fixed** - it stayed in the history and was asked again |
| A binary that has been copied somewhere else | `config.rs:72` | **fixed** - it looked for its engine on the build machine |
| A GGUF folder with a link back up | `discovery.rs:221` | **fixed** - the walk never ended |
| The adapter state of a model in Ollama's store | `discovery.rs:51` | **fixed** - it always said "unknown" |

---

## What was wrong, and what it was

### The model was served unchanged, and said it was hereticed

The runtime reported what it had done with the weights. llama.cpp binds a LoRA
at runtime, so it said "hereticed". But when the adapter could not be bound -
because the model has an attention output that is not square, and llama.cpp
reshapes such a tensor - the adapter is **folded into the weights** instead.
That is also a change to the weights, but it was reported as "as it is", and
the note underneath it still said the adapter was "bound in".

The mode now says what was actually done:

```
serving   http://127.0.0.1:64736/v1 (hereticed)
          served by llama.cpp on port 64736, with the adapter folded into the weights
```

### The adapter was fitted to the base model, and bound to the instruct one

`qwen2.5:0.5b` is served from weights called *"Qwen2.5 0.5B **Instruct**"*, but
its adapter had been built on `Qwen/Qwen2.5-0.5B` - the **base** model. Same
shape, different weights, different refusal directions. The adapter bound
perfectly well and did nothing at all.

The weights say what they are, in their own `general.name`. The donor says what
it is, in its repository name. When they disagree about the role, they are
different models:

```rust
pub fn donor_matches_weights(donor: &str, weights_name: &str) -> bool
```

The comparison goes both ways now, and `-it` and `-Instruct` are recognised as
the same role, so the same adapter is not rebuilt for ever.

### A model was rebuilt from scratch when the user asked to talk to it

The profile asks for twenty trials. A study with four trials was "not good
enough", so picking the model started an optimiser run that took an hour -
before the model said anything.

The person who asked for the model wants to *talk to it*. The adapter that is
already on disk works. So it is served, and the better one is something they
can ask for:

```
note      this adapter can still be improved
          type /improve to spend the time on a better one
```

### The model with the wrong adapter was served anyway

The merged model - the one with the adapter folded into the weights - was
served without checking which model the adapter had been made for. A model
folded from another model's adapter is that other model with the wrong weights
under it.

### The panel went on saying "build" for a model that was ready

`catalogue_stale` was set when a build finished and read in no place. The
panel kept saying "build" for a model that had an adapter, and a user who
believed it would rebuild one that already existed - hours of the GPU.

### The log was collected and never shown

Every event the engine sent was pushed into `App.log`, and `App.log` was
never drawn. When an engine died, the terminal showed nothing at all. The log
is now in the panel, and so is the question an engine asks.

### The terminal could be told things by other users on the machine

The channel socket was `/tmp/hermetic-<pid>.sock`. `/tmp` is world-writable,
so anybody on the machine could connect to it and tell the terminal that a
model was being served, and which. It is now in this user's own temporary
directory, which is not world-writable.

### A model name could run an escape sequence

A model's name can be a file name on disk, and macOS allows anything in one
except `/` and NUL. A name with an escape sequence in it was drawn as it was,
which means the terminal *ran* it - and a newline in a name moved every row
below it, so a click landed on the wrong model.

---

## Still open

These are known and not fixed. They are listed so that they are not a
surprise.

| | |
|---|---|
| **No scrollback in the shell pane** | Output that has scrolled off is gone. The vt100 parser keeps it; nothing reads it. |
| **No control for a served model** | A model is stopped by closing its tab or quitting. There should be a key for it. |
| **The panel disappears below 94 columns** | Below that width there is a bare shell: no catalogue, no tabs, no engine status. |
| **The shape key does not include the head count** | Two models with the same width and depth but different attention geometry would share an adapter. Changing the key would orphan every adapter on every machine, so it has not been done. |
| **A busy hub during a donor search** | The hub is asked for many repositories, and the search can take a minute. A rate limit in the middle of it is reported, but the search is not resumed. |
| **`HERMETIC_PROFILE` is not validated** | It is joined into a path. A profile name with `../` in it can point outside the profiles directory. |
