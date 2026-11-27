#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
HERETICAL bridge.

Runs the vendored Heretic engine **unchanged**, but exposes it to a native
client over a structured channel:

  * stdout / stderr -> Heretic's own rich console output. The terminal renders
    it through its vt100 grid, exactly like any other command it spawns.
  * channel         -> newline-delimited JSON events (phase, score, trial,
                       question, done, error). The client answers questions, so
                       the engine never blocks on a TTY.

Upstream's control flow is not forked. We only use its own extension points:

  * ``heretic.utils.print``      -> console output + "log" events
  * ``heretic.main.questionary`` -> question server (the client answers)
  * ``Evaluator.get_scores`` /
    ``Evaluator.get_baseline_scores`` -> "scores" events
  * ``Modifier.suggest_parameters`` /
    ``Modifier.modify_model``    -> "trial" events

Everything upstream does keeps working untouched: every scorer, every modifier,
every shipped profile, the plugin system, the reproduce.json machinery and the
Grimoire.

Usage:
    python -m heretical_bridge [heretic CLI args...]

Channel selection (either or both):
    HERETICAL_CHANNEL_FILE=/path/events.ndjson    append NDJSON events to a file
    HERMETIC_CHANNEL_SOCKET=/path/heretical.sock     connect to a unix socket (client
                                               listens; bridge connects & retries)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Channel
# ──────────────────────────────────────────────────────────────────────────


class Channel:
    """Emits newline-delimited JSON events to a file and/or a unix socket."""

    def __init__(self, file_path: str | None = None, socket_path: str | None = None):
        self._file = None
        self._sock: socket.socket | None = None
        self._buf = b""
        self._lock = __import__("threading").Lock()
        self.connected = False

        if file_path:
            self._file = open(file_path, "a", buffering=1, encoding="utf-8")

        if socket_path:
            self._sock = self._connect(socket_path)

    def _connect(self, path: str, attempts: int = 100, delay: float = 0.05) -> socket.socket:
        last: Exception | None = None
        for _ in range(attempts):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                self.connected = True
                return s
            except OSError as e:  # client may not be listening yet
                last = e
                time.sleep(delay)
        raise RuntimeError(f"could not connect to channel socket {path}: {last}")

    def send(self, event: dict[str, Any]) -> None:
        line = (json.dumps(event, default=_jsonable) + "\n").encode()
        with self._lock:
            if self._file is not None:
                try:
                    self._file.write(line.decode())
                except Exception:
                    pass
            if self._sock is not None:
                try:
                    self._sock.sendall(line)
                except Exception:
                    self._sock = None

    def read_answer(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Read one NDJSON answer from the socket (blocking)."""
        if self._sock is None:
            return None
        self._sock.settimeout(timeout)
        try:
            while b"\n" not in self._buf:
                chunk = self._sock.recv(65536)
                if not chunk:
                    return None
                self._buf += chunk
            line, self._buf = self._buf.split(b"\n", 1)
        except (socket.timeout, OSError):
            return None
        if not line.strip():
            return None
        try:
            return json.loads(line.decode())
        except json.JSONDecodeError:
            return None

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return repr(obj)


CHANNEL = Channel()


def emit(event: str, **fields: Any) -> None:
    CHANNEL.send({"event": event, "ts": time.time(), **fields})


# ──────────────────────────────────────────────────────────────────────────
# Question server
# ──────────────────────────────────────────────────────────────────────────


def _trial_summary(value: Any) -> dict[str, Any] | None:
    """Extract a compact summary from an Optuna FrozenTrial choice value."""
    user_attrs = getattr(value, "user_attrs", None)
    if not isinstance(user_attrs, dict):
        return None
    params = user_attrs.get("parameters")
    scores = user_attrs.get("scores")
    number = user_attrs.get("index")
    if number is None:
        number = getattr(value, "number", None)
    summary: dict[str, Any] = {"number": number}
    if isinstance(params, dict):
        summary["parameters"] = {k: _jsonable(v) for k, v in params.items()}
    if isinstance(scores, list):
        summary["scores"] = _jsonable(scores)
    return summary


def _choice_payload(choice: Any) -> dict[str, Any]:
    title = getattr(choice, "title", None)
    value = getattr(choice, "value", None)
    description = getattr(choice, "description", None)
    payload: dict[str, Any] = {
        "title": title if isinstance(title, str) else repr(title),
        "value": _jsonable(value),
    }
    if description:
        payload["description"] = description if isinstance(description, str) else repr(description)
    trial = _trial_summary(value)
    if trial:
        payload["trial"] = trial
    return payload


class _Question:
    """Stands in for a questionary question object.

    Upstream calls ``.ask()`` (and ``.unsafe_ask()``) on whatever
    ``questionary.select(...)`` returned, so the shim must return an object
    with those methods rather than the answer itself.

    The *original* choice values are kept in ``objects``: the JSON payload is a
    lossy projection of them, and upstream expects the real Python objects
    (e.g. an Optuna ``FrozenTrial``) back, not their repr.
    """

    def __init__(
        self,
        kind: str,
        message: str,
        payload: dict[str, Any],
        default: Any,
        objects: list[Any] | None = None,
        server: "QuestionServer | None" = None,
    ):
        self.kind = kind
        self.message = message
        self.payload = payload
        self.default = default
        self.objects = objects or []
        self.server = server

    def ask(self) -> Any:
        return self._resolve()

    def unsafe_ask(self) -> Any:
        return self._resolve()

    def _resolve(self) -> Any:
        repeat = self.server.note_asked(self.message) if self.server else False
        emit(
            "question",
            kind=self.kind,
            message=self.message,
            payload=self.payload,
            default=_jsonable(self.default),
            repeat=repeat,
        )
        answer = CHANNEL.read_answer(timeout=None)
        if answer is None or "answer" not in answer:
            # Headless: upstream's outer loop re-asks the same question after
            # every completed action ("Which trial do you want to use?").
            # Upstream treats an empty answer as "stop" and returns, so a
            # repeated question must be answered empty — otherwise the engine
            # loops forever and never finishes.
            if repeat and not CHANNEL.connected:
                return ""
            return self.default
        value = answer["answer"]
        if self.kind == "select" and isinstance(value, int):
            if 0 <= value < len(self.objects):
                return self.objects[value]
            return self.default
        if self.kind == "checkbox" and isinstance(value, list):
            picked = []
            for i in value:
                if isinstance(i, int) and 0 <= i < len(self.objects):
                    picked.append(self.objects[i])
            return picked
        return value


class QuestionServer:
    """Drop-in replacement for the subset of ``questionary`` that Heretic uses."""

    def __init__(self, policy: str = "auto"):
        self.policy = policy
        self._asked: dict[str, int] = {}

    def note_asked(self, message: str) -> bool:
        """Record that a question was asked; True if it is a repeat."""
        n = self._asked.get(message, 0) + 1
        self._asked[message] = n
        return n > 1

    # -- helpers ---------------------------------------------------------
    def _select_question(self, message: str, choices: Any) -> _Question:
        """Build a select/checkbox question, keeping the original values.

        ``objects`` holds the real Python objects (e.g. Optuna FrozenTrials);
        the JSON payload is only a display projection of them.
        """
        choices = list(choices or [])
        objects = [getattr(c, "value", c) for c in choices]
        rendered = [_choice_payload(c) for c in choices]
        payload = {"choices": rendered}

        default: Any = None
        if objects:
            if self.policy == "first":
                default = objects[0]
            else:
                # "auto": prefer a real trial over the "continue" pseudo-choice
                default = objects[0]
                for obj, r in zip(objects, rendered):
                    if r.get("trial") is not None:
                        default = obj
                        break
        return _Question("select", message, payload, default, objects, server=self)

    # -- questionary API -------------------------------------------------
    def select(self, message: str, choices: Any = None, **_: Any) -> _Question:
        return self._select_question(message, choices)

    def checkbox(self, message: str, choices: Any = None, **_: Any) -> _Question:
        q = self._select_question(message, choices)
        q.kind = "checkbox"
        q.default = []
        return q

    def text(self, message: str, **_: Any) -> _Question:
        payload: dict[str, Any] = {}
        return _Question("text", message, payload, self._default_for("text", payload), server=self)

    def path(self, message: str, **_: Any) -> _Question:
        payload: dict[str, Any] = {}
        return _Question("path", message, payload, self._default_for("path", payload), server=self)

    def password(self, message: str, **_: Any) -> _Question:
        payload: dict[str, Any] = {}
        return _Question("password", message, payload, self._default_for("password", payload), server=self)

    def confirm(self, message: str, **_: Any) -> _Question:
        payload: dict[str, Any] = {}
        return _Question("confirm", message, payload, True, server=self)

    def _default_for(self, kind: str, payload: dict[str, Any]) -> Any:
        if kind == "checkbox":
            return []
        if kind == "path":
            return ""
        if kind == "password":
            return ""
        return ""


# ──────────────────────────────────────────────────────────────────────────
# Seams
# ──────────────────────────────────────────────────────────────────────────


def install() -> None:
    """Patch the upstream seams. Must run before ``heretic.main`` is imported."""
    import heretic.utils as hu

    console_print = hu.print

    def _plain(markup: str) -> str:
        """Strip Rich markup so log events carry readable text.

        The console renders markup itself, but the JSON event is plain text —
        a client that shows it in a pane would otherwise display
        "[bold green]Optimization finished![/]".
        """
        try:
            from rich.text import Text

            return Text.from_markup(markup).plain
        except Exception:
            return re.sub(r"\[/?[a-zA-Z0-9 _#(),.\-]*\]", "", markup)

    def bridged_print(*args: Any, **kwargs: Any) -> None:
        console_print(*args, **kwargs)
        text = " ".join(str(a) for a in args)
        emit("log", text=_plain(text))

    hu.print = bridged_print  # type: ignore[assignment]

    import heretic.evaluator as ev

    _get_scores = ev.Evaluator.get_scores
    _get_baseline = ev.Evaluator.get_baseline_scores

    def get_scores(self: Any, *a: Any, **k: Any) -> Any:
        result = _get_scores(self, *a, **k)
        emit(
            "scores",
            phase="trial",
            scores=[{"name": n, "value": s.value, "display": _plain(s.rich_display)} for n, s in result],
        )
        return result

    def get_baseline_scores(self: Any, *a: Any, **k: Any) -> Any:
        result = _get_baseline(self, *a, **k)
        emit(
            "scores",
            phase="baseline",
            scores=[{"name": n, "value": s.value, "display": _plain(s.rich_display)} for n, s in result],
        )
        return result

    ev.Evaluator.get_scores = get_scores  # type: ignore[method-assign]
    ev.Evaluator.get_baseline_scores = get_baseline_scores  # type: ignore[method-assign]

    import heretic.modifier as md

    # NOTE: concrete plugins (ARA, Abliteration) *override* suggest_parameters
    # and modify_model, so wrapping the base class would never fire. We wrap
    # the loaded *instances* instead.
    _load_and_init = md.load_and_init_modifiers

    def _wrap_modifier_instance(mod: Any) -> None:
        _suggest = mod.suggest_parameters
        _modify = mod.modify_model

        def suggest_parameters(ctx: Any, trial: Any) -> Any:
            params = _suggest(ctx, trial)
            number = None
            try:
                number = trial.user_attrs.get("index", getattr(trial, "number", None))
            except Exception:
                pass
            emit(
                "trial_start",
                number=number,
                modifier=type(mod).__name__,
                parameters=_jsonable(params.to_dict() if hasattr(params, "to_dict") else params),
            )
            return params

        def modify_model(ctx: Any, parameters: Any) -> Any:
            emit(
                "trial_modify",
                modifier=type(mod).__name__,
                parameters=_jsonable(
                    parameters.to_dict() if hasattr(parameters, "to_dict") else parameters
                ),
            )
            return _modify(ctx, parameters)

        mod.suggest_parameters = suggest_parameters
        mod.modify_model = modify_model

    def load_and_init_modifiers(settings: Any, model: Any) -> Any:
        entries = _load_and_init(settings, model)
        for entry in entries:
            try:
                _wrap_modifier_instance(entry.modifier)
            except Exception:
                pass
        return entries

    md.load_and_init_modifiers = load_and_init_modifiers  # type: ignore[assignment]

    # questionary is imported at module scope by heretic.main and heretic.reproduce
    import heretic.main as hm

    shim = QuestionServer()
    hm.questionary = shim  # type: ignore[assignment]
    # Patch any other already-imported heretic module that captured questionary.
    for _name, _mod in list(sys.modules.items()):
        if _name.startswith("heretic.") and getattr(_mod, "questionary", None) is not None:
            _mod.questionary = shim  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    # Both spellings are accepted: the client may use the "hermetic" prefix
    # (the engine's own name) or the "heretical" one. The first spelling of
    # each pair wins and the attribute is the same either way, so the two
    # sides can never disagree about the name of a flag.
    parser.add_argument(
        "--hermetic-events-file",
        "--heretical-events-file",
        dest="events_file",
        default=None,
    )
    parser.add_argument(
        "--hermetic-socket",
        "--heretical-socket",
        dest="socket_path",
        default=None,
    )
    parser.add_argument(
        "--hermetic-policy",
        "--heretical-policy",
        dest="policy",
        default="auto",
        choices=["auto", "first"],
    )
    parser.add_argument(
        "--hermetic-label",
        "--heretical-label",
        dest="label",
        default=None,
        help="what the client should call this run, e.g. the model being worked on",
    )
    known, rest = parser.parse_known_args()

    global CHANNEL
    CHANNEL = Channel(
        file_path=known.events_file
        or os.environ.get("HERMETIC_CHANNEL_FILE")
        or os.environ.get("HERETICAL_CHANNEL_FILE"),
        socket_path=known.socket_path
        or os.environ.get("HERMETIC_CHANNEL_SOCKET")
        or os.environ.get("HERETICAL_CHANNEL_SOCKET"),
    )

    if known.label:
        # So the client can tell several running engines apart.
        emit("build_start", label=known.label)

    install()

    import heretic.main as hm

    sys.argv = [sys.argv[0]] + rest
    exit_code = 0
    try:
        hm.main()
    except SystemExit as e:
        if isinstance(e.code, int):
            exit_code = e.code
        else:
            # A non-integer SystemExit is an *error message*, not a status:
            # Heretic raises `SystemExit("heretic: configuration file ...
            # does not exist")` for a missing config. Reporting that as exit
            # code 0 would tell the client the run succeeded when it never
            # started.
            exit_code = 1
            emit("error", error=str(e.code))
    except BaseException as e:  # noqa: BLE001
        emit("error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        # emit BEFORE closing the channel — otherwise the final event is lost
        emit("done", exit_code=exit_code)
        CHANNEL.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
