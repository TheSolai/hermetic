#!/bin/zsh
# Verified A/B harness for Heretic LoRA adapters.
#
# Unlike an ad-hoc run, this refuses to produce a number unless it can PROVE,
# from the server logs, that:
#   - both servers are serving the intended base model
#   - server B actually loaded a LoRA adapter (and how many tensors)
#   - server A loaded no adapter
# A stale process holding the port used to silently invalidate a run; this
# script kills port holders first and asserts the invariants afterwards.

set -u
BASE=${BASE:?set BASE=<model.gguf>}
ADAPTER=${ADAPTER:?set ADAPTER=<adapter.gguf>}
PA=${PA:-8081}
PB=${PB:-8082}
LOG_A=${LOG_A:-/tmp/ab-A.log}
LOG_B=${LOG_B:-/tmp/ab-B.log}

free_ports() {
  for p in $PA $PB; do
    pids=$(lsof -nP -iTCP:$p -sTCP:LISTEN -t 2>/dev/null)
    [ -n "$pids" ] && kill $pids 2>/dev/null
  done
  sleep 2
  for p in $PA $PB; do
    if lsof -nP -iTCP:$p -sTCP:LISTEN -t >/dev/null 2>&1; then
      echo "FATAL: port $p still held"; return 1
    fi
  done
  return 0
}

free_ports || exit 1

llama-server -m "$BASE" --host 127.0.0.1 --port $PA -c 2048 -np 4 --no-warmup -lv 5 > "$LOG_A" 2>&1 &
PID_A=$!
llama-server -m "$BASE" --host 127.0.0.1 --port $PB -c 2048 -np 4 --no-warmup -lv 5 --lora "$ADAPTER" > "$LOG_B" 2>&1 &
PID_B=$!
trap 'kill $PID_A $PID_B 2>/dev/null' EXIT

for i in $(seq 1 120); do
  a=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PA/health 2>/dev/null)
  b=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PB/health 2>/dev/null)
  # NOTE: /health answers as soon as the port is bound, which happens BEFORE the
  # model is loaded (and before the LoRA adapter is applied). Readiness must be
  # taken from the server log, not from /health.
  a_ready=$(grep -c "model loaded" "$LOG_A" 2>/dev/null || echo 0)
  b_ready=$(grep -c "model loaded" "$LOG_B" 2>/dev/null || echo 0)
  [ "$a" = "200" ] && [ "$b" = "200" ] && [ "$a_ready" -gt 0 ] && [ "$b_ready" -gt 0 ] && break
  sleep 1
done

echo "=== VERIFICATION ==="
ok=1
for f in "$LOG_A" "$LOG_B"; do
  if grep -q "loading model '$BASE'" "$f"; then
    echo "  OK   $f serves $BASE"
  else
    echo "  FAIL $f does NOT serve $BASE"; ok=0
  fi
done
if grep -q "loading lora adapter from" "$LOG_B"; then
  echo "  OK   $LOG_B loaded a LoRA adapter: $(grep -m1 'loaded .* tensors from lora file' "$LOG_B" | sed 's/.*: //')"
else
  echo "  FAIL $LOG_B did NOT load an adapter"; ok=0
fi
if grep -q "loading lora adapter from" "$LOG_A"; then
  echo "  FAIL $LOG_A unexpectedly loaded an adapter"; ok=0
else
  echo "  OK   $LOG_A has no adapter (baseline)"
fi
if grep -qE "incorrect shape|model loading error" "$LOG_B"; then
  echo "  FAIL adapter shape error: $(grep -m1 -E 'incorrect shape' "$LOG_B")"; ok=0
fi

if [ "$ok" != "1" ]; then
  echo "VERIFICATION FAILED — refusing to report numbers"; exit 2
fi

echo "=== verified: running benchmark ==="
PY=${PY:-/Users/amre/Projects/heretic/.venv/bin/python}
"$PY" "$(dirname "$0")/ab_refusal_bench.py"
