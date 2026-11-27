# The protocol

The bridge and the terminal speak newline-delimited JSON over a unix socket.
One socket, both directions.

```
terminal                                   bridge
   |                                          |
   |  ---- connect -------------------------> |   (the bridge connects; the terminal listens)
   |  <--- {"event":"log",...} ------------- |
   |  <--- {"event":"scores",...} ---------- |
   |  <--- {"event":"question",...} -------- |
   |  ---- {"answer": 0} -------------------> |
   |  <--- {"event":"done","exit_code":0} -- |
```

## Connection

The **terminal listens** and the **bridge connects**. The bridge retries its
connect, so a slow Python start is not a failure on the terminal's side.

The socket path is passed to the bridge as `--hermetic-socket` and as
`HERMETIC_CHANNEL_SOCKET`. The terminal binds it before starting the engine, so
the connect cannot race.

The bridge can also write the same events to a file
(`--hermetic-events-file` / `HERMETIC_CHANNEL_FILE`). That is how a headless run
records what happened.

### More than one engine

The terminal accepts **more than one** connection. The user can start a second
build while the first is still running, so the socket is a hub: every
connection is given an id, and the events from it are tagged with that id.
The answer goes back down the connection it came from, which matters because
the engine that is waiting for an answer is the one that asked.

## Events

Every event has an `event` field naming its kind and a `ts` with a Unix
timestamp. The terminal ignores events it does not know, so the engine can
grow without the terminal having to change.

### `log`

```json
{"event": "log", "ts": 1795742930.93, "text": "[cyan]v2.2.0.dev0[/]"}
```

A line of the engine's console output, with Rich markup still in it. This is
the same text the engine prints to stdout; it is sent as well so a client can
follow along without parsing the terminal stream.

### `build_start`

```json
{"event": "build_start", "ts": 1795748200.11, "label": "adapter for Qwen/Qwen2.5-0.5B"}
```

The bridge is starting work on something. `label` says what, so the terminal
can tell two running builds apart. It is passed as `--hermetic-label`.

### `serving`

```json
{"event": "serving", "ts": 1795748200.11, "model": "qwen2.5:0.5b",
 "tab": 2, "shape": "qwen2 · 24 layers · hidden 896 · ffn 4864",
 "adapter": "/Users/amre/.hermetic-ollama/adapters/qwen2-24-896-4864.gguf",
 "endpoint": "http://127.0.0.1:58487"}
```

The interceptor is serving a model. `tab` is the number of the tab that asked
for it, so the terminal can name that tab after the model. `shape`,
`adapter` and `endpoint` are what the panel shows.

This one is not sent by the bridge: the bridge never sees the model the user
asked for. The interceptor sends it over the same socket, on a connection of
its own, and closes it.

### `scores`

```json
{"event": "scores", "ts": 1795743634.04, "phase": "baseline",
 "scores": [{"name": "Refusals", "value": 0.48, "display": "[bold]48[/]/100"}]}
```

`phase` is `baseline` or `trial`. `display` is the formatted value;
`value` is the raw number.

### `trial_start`

```json
{"event": "trial_start", "ts": 1795743647.72, "number": 1, "modifier": "ARA",
 "parameters": {"start_layer_index": 6, "end_layer_index": 28, ...}}
```

The optimiser proposed a set of parameters. `number` is the trial index.

### `trial_modify`

```json
{"event": "trial_modify", "ts": 1795743647.72, "modifier": "ARA",
 "parameters": {...}}
```

The parameters are being applied to the model now.

### `question`

```json
{"event": "question", "ts": 1795743724.88, "kind": "select",
 "message": "Which trial do you want to use?",
 "payload": {"choices": [
   {"title": "[Trial   1] Refusals: 4/100",
    "value": "FrozenTrial(number=0, ...)",
    "trial": {"number": 0, "parameters": {...}, "scores": [...]}}
 ]},
 "default": null}
```

`kind` is one of `select`, `checkbox`, `text`, `path`, `password`, `confirm`.

`payload.choices` is a **projection** for display. Its `value` field is a
string rendering of the underlying Python object, not the object.

### `done`

```json
{"event": "done", "ts": 1795743724.88, "exit_code": 0}
```

The engine finished. `exit_code` is 0 on success and non-zero on failure.

### `error`

```json
{"event": "error", "ts": 1795743724.88, "error": "AttributeError: ..."}
```

The engine crashed.

## Answers

The terminal sends one NDJSON object per answer:

```json
{"answer": 0}
```

| Question kind | What to send |
|---|---|
| `select` | The **index** of the chosen option: `{"answer": 2}` |
| `checkbox` | A list of indices: `{"answer": [0, 3]}` |
| `text`, `path`, `password` | A string: `{"answer": "some/path"}` |
| `confirm` | A boolean: `{"answer": true}` |

The bridge resolves the index against the **original** choice objects and
hands those back to Heretic. It does not resolve against the payload, which
is only a display projection.

This is the one detail worth getting wrong carefully. The payload's `value`
is `repr(trial)` - a string. If the bridge resolved the index against the
payload, Heretic would receive a string where it expects an Optuna
`FrozenTrial`, and fail with:

```
AttributeError: 'str' object has no attribute 'user_attrs'
```

which is exactly what happened before `objects` was added to the bridge.

## Where the seams are

The bridge does not fork upstream Heretic. It patches five things, all of
which are already seams in upstream's own design:

| Upstream | What the bridge does |
|---|---|
| `heretic.utils.print` | Calls the original, then emits a `log` event. |
| `heretic.main.questionary` | Replaces the module with a question server. |
| `heretic.evaluator.Evaluator.get_scores` | Emits `scores`, then returns the original result. |
| `heretic.evaluator.Evaluator.get_baseline_scores` | Same, with `phase: "baseline"`. |
| `heretic.modifier.Modifier.suggest_parameters` / `.modify_model` | Emits `trial_start` / `trial_modify`, then calls the original. |

The question server implements the subset of `questionary` that Heretic uses:
`select`, `checkbox`, `text`, `path`, `password`, `confirm`. Each returns an
object with `ask()` and `unsafe_ask()`, because that is what upstream calls.

## Exit codes

`done.exit_code` is the engine's real exit status. A `SystemExit` carrying a
**string** - Heretic raises `SystemExit("heretic: configuration file ... does
not exist")` for a missing config - is a fatal error, and the bridge reports
it as exit code 1 with an `error` event.

Reporting it as 0 would tell the terminal the run succeeded when the engine
never started. That bug hid a failed run behind a clean exit.
