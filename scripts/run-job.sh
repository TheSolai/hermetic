#!/bin/zsh
# Generic detached-job runner for the heretic-term project.
#
# Usage: run-job.sh <job-name> <command...>
#
# Uses a LaunchAgent plist with KeepAlive=false. `launchctl submit` must NOT be
# used for this: submitted jobs are KeepAlive, so a script that exits is
# immediately restarted, turning a one-shot run into an infinite loop.
#
# `nohup ... & disown` does not survive the exec task either - the whole
# process group is torn down when the task ends.
#
# Writes:
#   jobs/<name>.log   combined stdout+stderr
#   jobs/<name>.done  "EXIT=<code>", on completion
set -u
NAME="$1"; shift

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$ROOT/jobs"
PLIST="$HOME/Library/LaunchAgents/com.dross.heretical.$NAME.plist"

mkdir -p "$JOBS" "$HOME/Library/LaunchAgents"
rm -f "$JOBS/$NAME.done" "$JOBS/$NAME.log"

# The command goes in its own file so the plist never has to escape arguments.
printf '%s\n' "$*" > "$JOBS/$NAME.cmd"

INNER="$JOBS/$NAME.inner.sh"
cat > "$INNER" <<'INNER_EOF'
#!/bin/zsh
NAME="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
zsh "$ROOT/jobs/$NAME.cmd" > "$ROOT/jobs/$NAME.log" 2>&1
echo "EXIT=$?" > "$ROOT/jobs/$NAME.done"
INNER_EOF
chmod +x "$INNER"

cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dross.heretical.$NAME</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$INNER</string>
    <string>$NAME</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>StandardOutPath</key>
  <string>$JOBS/$NAME.log</string>
  <key>StandardErrorPath</key>
  <string>$JOBS/$NAME.log</string>
</dict>
</plist>
PLISTEOF

launchctl bootout "gui/$(id -u)/com.dross.heretical.$NAME" 2>/dev/null
launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null \
  || launchctl load -w "$PLIST" 2>/dev/null

echo "started: com.dross.heretical.$NAME"
