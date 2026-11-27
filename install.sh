#!/usr/bin/env bash
# HERMETIC - install.
#
# One command, from a fresh clone to a double-clickable app:
#
#     ./install.sh
#
# It checks what it needs, builds the engine (the vendored Heretic, in its own
# virtualenv), builds the two binaries, and installs the app bundle into
# ~/Applications with a shortcut on the Desktop.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/engine/.venv"

say()  { printf '\033[1;36m::\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$1"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$1" >&2; exit 1; }

say "HERMETIC - installing into $ROOT"

# ── what it needs ──────────────────────────────────────────────────────────
[ "$(uname -s)" = "Darwin" ] || die "HERMETIC is a macOS terminal. This is $(uname -s)."

command -v cargo >/dev/null 2>&1 || die "cargo is missing. Install Rust: https://rustup.rs"
command -v python3 >/dev/null 2>&1 || die "python3 is missing."

UV=""
if command -v uv >/dev/null 2>&1; then
  UV="uv"
else
  warn "uv is missing - falling back to python3 -m venv (slower, but it works)."
fi

# llama-server and ollama are not needed to install HERMETIC, but they are
# needed to actually run a model, so say so now rather than at 2am.
command -v llama-server >/dev/null 2>&1 || warn "llama-server is not on the PATH - install llama.cpp, or HERMETIC will not be able to serve a model."
command -v ollama >/dev/null 2>&1 || warn "ollama is not on the PATH - HERMETIC will not find any models in Ollama's store."

# ── the engine ─────────────────────────────────────────────────────────────
if [ -x "$VENV/bin/python" ]; then
  say "the engine's virtualenv is already there"
else
  say "building the engine's virtualenv (this takes a minute)"
  if [ -n "$UV" ]; then
    "$UV" venv --python 3.13 "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

say "installing Heretic into it"
if [ -n "$UV" ]; then
  "$UV" pip install --python "$VENV/bin/python" -e "$ROOT/engine/heretic"
  # Heretic's own dependencies do not include these, but several tokenizers
  # need them, and a model that fails to load is a bad first impression.
  "$UV" pip install --python "$VENV/bin/python" tiktoken protobuf sentencepiece
else
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -e "$ROOT/engine/heretic"
  "$VENV/bin/python" -m pip install --quiet tiktoken protobuf sentencepiece
fi

# ── the binaries ───────────────────────────────────────────────────────────
say "building the terminal"
(cd "$ROOT" && cargo build --release)

# ── the app ─────────────────────────────────────────────────────────────────
say "building the app"
"$ROOT/scripts/build_app.sh" "$ROOT" >/dev/null

say "done."
echo
echo "  app:       ~/Applications/HERMETIC.app"
echo "  shortcut:  ~/Desktop/HERMETIC.command"
echo "  run now:   ~/Applications/HERMETIC.app/Contents/MacOS/hermetic"
echo
echo "  Type a model name at the prompt, or click one in the panel."
echo "  `ollama run qwen3:8b` works too: the adapter is bound before the"
echo "  model is loaded."
