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


def _trial_total(value: Any) -> float | None:
    """The total score of a trial: the sum of its scorers.

    Every scorer is minimised, so the lowest total is the trial that refuses
    least *and* keeps the model closest to what it was.
    """
    user_attrs = getattr(value, "user_attrs", None)
    if not isinstance(user_attrs, dict):
        return None
    scores = user_attrs.get("scores")
    if not isinstance(scores, list):
        return None
    total = 0.0
    seen = False
    for score in scores:
        if not isinstance(score, dict):
            continue
        inner = score.get("score")
        if not isinstance(inner, dict):
            continue
        number = inner.get("value")
        if isinstance(number, (int, float)):
            total += float(number)
            seen = True
    return total if seen else None


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
        best_trial: int | None = None,
    ):
        self.kind = kind
        self.message = message
        self.payload = payload
        self.default = default
        self.objects = objects or []
        self.server = server
        # Which of the choices is the trial worth keeping, when the choices
        # are trials.
        self.best_trial = best_trial

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
        return self._resolve_with(answer["answer"])

    def _resolve_with(self, value: Any) -> Any:
        """Turn the client's answer into the object upstream expects."""
        if self.kind == "select" and isinstance(value, int) and not isinstance(value, bool):
            # The client's answer is the first choice, which is the trial
            # upstream sorts first: the one that refuses not at all, even
            # when it has moved the model so far that it no longer speaks.
            # The trial worth keeping is the one with the lowest total
            # score, and the engine knows which that is.
            if self.best_trial is not None and value == 0:
                return self.objects[self.best_trial]
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
        best_trial: int | None = None
        if objects:
            if self.policy == "first":
                default = objects[0]
            else:
                # "auto": prefer a real trial over the "continue" pseudo-choice.
                #
                # And among the trials, the one with the *lowest total score*:
                # upstream sorts its Pareto front by the scorers one after
                # another, so the trial that refuses not at all comes first -
                # even when it has moved the model so far that it no longer
                # speaks. The sum is the score that means something: refuse
                # less *and* still be the model.
                default = objects[0]
                best_total: float | None = None
                for index, (obj, r) in enumerate(zip(objects, rendered)):
                    if r.get("trial") is None:
                        continue
                    total = _trial_total(obj)
                    if total is None:
                        if best_trial is None:
                            best_trial = index
                        continue
                    if best_total is None or total < best_total:
                        best_trial = index
                        best_total = total
                if best_trial is not None:
                    default = objects[best_trial]
                    total = _trial_total(objects[best_trial])
                    number = (rendered[best_trial].get("trial") or {}).get("number")
                    emit(
                        "log",
                        text=(
                            f"keeping trial {number} - the trial with the lowest total "
                            f"score"
                            + (f" ({total:.3f})" if total is not None else "")
                        ),
                    )
        return _Question(
            "select", message, payload, default, objects, server=self, best_trial=best_trial
        )

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
        _capture_probe(self)
        emit(
            "scores",
            phase="trial",
            scores=[{"name": n, "value": s.value, "display": _plain(s.rich_display)} for n, s in result],
        )
        return result

    def get_baseline_scores(self: Any, *a: Any, **k: Any) -> Any:
        result = _get_baseline(self, *a, **k)
        _capture_probe(self, result)
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

    _install_trust_seam()
    _install_settings_seam()

    # questionary is imported at module scope by heretic.main and heretic.reproduce
    import heretic.main as hm

    shim = QuestionServer()
    hm.questionary = shim  # type: ignore[assignment]
    # Patch any other already-imported heretic module that captured questionary.
    for _name, _mod in list(sys.modules.items()):
        if _name.startswith("heretic.") and getattr(_mod, "questionary", None) is not None:
            _mod.questionary = shim  # type: ignore[attr-defined]


def _install_settings_seam(settings_cls: Any = None) -> None:
    """A resumed study must not resurrect a pinned trial.

    Upstream stores the settings of the run that created a study *inside the
    study*, and restores them when the study is resumed. That list includes
    ``trial_index``: the index into the sorted Pareto front of the trial to
    keep. A study created by a run with the trial pinned therefore hands that
    pin to every run that resumes it, and the question "which trial do you
    want to use?" is then never asked - upstream takes the first trial of the
    sorted front without asking.

    That first trial is the one that refuses not at all, whatever it has cost
    the model: upstream sorts its Pareto front by the scorers one after
    another, so a trial with no refusals at all comes first, even when it has
    moved the model so far that it no longer speaks. (On gemma-2-2b that is
    the difference between a KL divergence of 0.08 and one of 8.79 - the
    first speaks, the second answers in gibberish.)

    The pin is dropped here, so the question is always asked, and the bridge
    answers it with the trial that keeps the model speaking. A pin that came
    from the command line is honoured: only the fossil inside a study is
    dropped.
    """
    if settings_cls is None:
        try:
            import heretic.config as hc
        except Exception:
            return
        settings_cls = hc.Settings

    original = getattr(settings_cls, "model_validate_init_only", None)
    if original is None or getattr(original, "_hermetic_patched", False):
        return

    def model_validate_init_only(cls: Any, values: Any, current: Any) -> Any:
        if not isinstance(values, dict):
            return original.__func__(cls, values, current)

        values = dict(values)

        # The trial to keep belongs to the run that asked for it, not to the
        # study. Upstream restores the study's settings over the command
        # line, so the pin has to be carried over explicitly.
        if values.get("trial_index") is not None:
            values["trial_index"] = getattr(current, "trial_index", None)

        # How much work to do belongs to the profile, not to the study. A
        # study made when the profile asked for one trial and resumed under a
        # profile that asks for twenty is a study that has one trial and
        # needs nineteen more - not a study that is finished.
        for field in ("n_trials", "n_startup_trials"):
            wanted = getattr(current, field, None)
            if wanted is not None:
                values[field] = wanted

        return original.__func__(cls, values, current)

    model_validate_init_only._hermetic_patched = True  # type: ignore[attr-defined]
    settings_cls.model_validate_init_only = classmethod(  # type: ignore[method-assign]
        model_validate_init_only
    )


#: How many questions the probe asks.
PROBE_SIZE = 16

#: Where the build is being saved, so the probe can be written next to it.
SAVE_DIRECTORY: Path | None = None

#: Whether the probe has already been written.
_PROBE_WRITTEN = False


def _capture_probe(evaluator: Any, scores: Any = None) -> None:
    """Write down the questions the model is judged on, and the markers.

    Upstream judges a model on a set of prompts and a list of markers: for
    refusals, a hundred questions and thirty-three phrases that give a
    refusal away. Those are the questions that decide whether the ablation
    worked, and they are the questions worth asking the *served* model
    afterwards - the one that is actually running, with the adapter bound to
    it.

    So they are written next to the build, and the client asks them.

    The score of the untouched model goes with them. A number on its own
    says little: "6 of 16" is only good news next to "it used to be 99 of
    100".
    """
    global _PROBE_WRITTEN

    if _PROBE_WRITTEN or SAVE_DIRECTORY is None:
        return

    for entry in getattr(evaluator, "_scorer_entries", []) or []:
        scorer = getattr(entry, "scorer", None)
        prompts = getattr(scorer, "prompts", None)
        markers = getattr(getattr(scorer, "settings", None), "keyword_markers", None)
        if not prompts or not markers:
            continue

        name = str(getattr(scorer, "score_name", "Refusals"))
        baseline = None
        for score_name, score in scores or []:
            if str(score_name) == name:
                baseline = float(getattr(score, "value", 0.0))
                break

        probe = {
            "name": name,
            "markers": [str(m) for m in markers],
            "prompts": [
                {
                    "system": str(getattr(p, "system", "") or ""),
                    "user": str(getattr(p, "user", "") or ""),
                }
                for p in list(prompts)[:PROBE_SIZE]
            ],
            "baseline": baseline,
        }
        try:
            path = SAVE_DIRECTORY / "probe.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(probe, indent=1), encoding="utf-8")
            _PROBE_WRITTEN = True
        except Exception:
            pass
        return


def _install_trust_seam() -> None:
    """Answer the "may this repository run its own code?" question with yes.

    Some models ship their own model code: a repository with an ``auto_map``
    in its config is loaded by executing the Python that sits next to the
    weights. Transformers asks the person at the keyboard whether that is
    allowed - and when the answer comes on a terminal that is not there, it
    fails, and the model is reported as unloadable.

    Upstream Heretic has the same question, asked the same way, and answers
    it the same way: it keeps a set of trusted repositories and adds one when
    the user says so. The engine here runs without a conversation, so the
    answer is given in its place - and the repository is named, so it is not
    a silent thing.

    Set ``HERMETIC_TRUST_REMOTE_CODE=0`` to refuse instead, and the engine
    will report the model as unloadable.
    """
    import os

    if os.environ.get("HERMETIC_TRUST_REMOTE_CODE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return

    try:
        import importlib

        from transformers import dynamic_module_utils
    except Exception:
        return

    original = getattr(dynamic_module_utils, "resolve_trust_remote_code", None)
    if original is None or getattr(original, "_hermetic_patched", False):
        return

    def resolve_trust_remote_code(  # type: ignore[no-untyped-def]
        trust_remote_code: Any,
        model_name: Any,
        has_local_code: Any,
        has_remote_code: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Only the repositories that actually ship code are affected, and
        # only when nobody has said otherwise. An explicit "no" from the
        # caller is honoured.
        if (
            trust_remote_code is None
            and has_remote_code
            and not has_local_code
        ):
            emit(
                "log",
                text=(
                    f"{model_name} ships its own model code; it is trusted so the "
                    "model can be loaded"
                ),
            )
            return True
        return original(
            trust_remote_code,
            model_name,
            has_local_code,
            has_remote_code,
            *args,
            **kwargs,
        )

    resolve_trust_remote_code._hermetic_patched = True  # type: ignore[attr-defined]

    # Transformers binds this function into a dozen modules with
    # ``from ... import resolve_trust_remote_code``, so patching one place is
    # not enough: the binding in each of those modules has to be replaced as
    # well. They are imported here first, so that a module imported later
    # binds the patched function in the first place.
    for module_name in (
        "transformers.dynamic_module_utils",
        "transformers.models.auto.auto_factory",
        "transformers.models.auto.configuration_auto",
        "transformers.models.auto.image_processing_auto",
        "transformers.models.auto.video_processing_auto",
        "transformers.models.auto.processing_auto",
        "transformers.models.auto.feature_extraction_auto",
        "transformers.models.auto.tokenization_auto",
        "transformers.generation.utils",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if getattr(module, "resolve_trust_remote_code", None) is not None:
            module.resolve_trust_remote_code = resolve_trust_remote_code  # type: ignore[attr-defined]


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

    global SAVE_DIRECTORY
    for index, item in enumerate(rest):
        if item == "--save-directory" and index + 1 < len(rest):
            SAVE_DIRECTORY = Path(rest[index + 1])
            break

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
