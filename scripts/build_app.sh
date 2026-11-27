#!/bin/zsh
# Build HERMETIC.app — a standalone, double-clickable macOS app bundle
# around the `hermetic` terminal and the `hermetic-shim` interceptor.
#
# The terminal is a terminal program, so the bundle's launcher opens a real
# Terminal window and runs it there (the standard macOS pattern for CLI apps).
#
# The engine (vendored Heretic, the Python bridge, the virtualenv) is far too
# big to ship inside a bundle, so the bundle carries a small configuration file
# that says where the engine lives.

set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
BUNDLE="$ROOT/dist/HERMETIC.app"
NAME="HERMETIC"

rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"

# ── the binaries ───────────────────────────────────────────────────────────
if [ ! -x "$ROOT/target/release/hermetic" ] || [ ! -x "$ROOT/target/release/hermetic-shim" ]; then
  echo "building..."
  (cd "$ROOT" && cargo build --release)
fi
cp "$ROOT/target/release/hermetic" "$BUNDLE/Contents/MacOS/hermetic"
cp "$ROOT/target/release/hermetic-shim" "$BUNDLE/Contents/MacOS/hermetic-shim"

# ── where the engine is ─────────────────────────────────────────────────────
cat > "$BUNDLE/Contents/Resources/hermetic.conf" <<CONF
# Where HERMETIC finds the engine (vendored Heretic + bridge + venv).
engine_dir=$ROOT/engine
CONF

# ── the launcher ───────────────────────────────────────────────────────────
# A .command file is used because `osascript ... tell application "Terminal"`
# needs Automation permission, which is denied in many environments and pops a
# consent dialog. Terminal runs a .command file natively — no TCC prompt.
cat > "$BUNDLE/Contents/MacOS/launcher" <<'LAUNCHER'
#!/bin/sh
# HERMETIC launcher — hands off to Terminal, which runs the terminal.
#
# The bundle root is two levels up from Contents/MacOS.
BUNDLE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# Terminal opens a .command file natively, with a real TTY. Writing it to a
# temp path avoids the quirks of opening a file that lives inside a bundle.
RUNNER="${TMPDIR:-/tmp}/hermetic-$$.command"
cat > "$RUNNER" <<RUNCMD
#!/bin/sh
exec "$BUNDLE_DIR/Contents/MacOS/hermetic"
RUNCMD
chmod +x "$RUNNER"
open -a Terminal "$RUNNER"
LAUNCHER
chmod +x "$BUNDLE/Contents/MacOS/launcher"

# ── Info.plist ─────────────────────────────────────────────────────────────
cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$NAME</string>
    <key>CFBundleDisplayName</key><string>$NAME</string>
    <key>CFBundleExecutable</key><string>launcher</string>
    <key>CFBundleIdentifier</key><string>org.thesolai.hermetic</string>
    <key>CFBundleIconFile</key><string>icon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>0.1.0</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# ── icon ──────────────────────────────────────────────────────────────────
ICONSET="$ROOT/dist/iconset.iconset"
rm -rf "$ICONSET"; mkdir -p "$ICONSET"
ICONSET="$ICONSET" "$ROOT/engine/.venv/bin/python" - <<'PY'
from PIL import Image, ImageDraw
import os

OUT = os.environ.get("ICONSET")
S = 512

def make(size, path):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = size * 0.08
    # rounded dark panel
    d.rounded_rectangle([m, m, size - m, size - m], radius=size * 0.14,
                        fill=(18, 21, 26, 255), outline=(70, 80, 96, 255), width=max(1, size // 64))
    # prompt caret
    cx, cy = size * 0.30, size * 0.34
    s = size * 0.10
    d.line([(cx, cy), (cx + s, cy + s * 0.9)], fill=(126, 214, 145, 255), width=max(2, size // 48))
    d.line([(cx + s, cy + s * 0.9), (cx, cy + s * 1.8)], fill=(126, 214, 145, 255), width=max(2, size // 48))
    # underscore
    d.line([(cx, cy + s * 2.3), (cx + s * 1.5, cy + s * 2.3)], fill=(126, 214, 145, 255), width=max(2, size // 48))
    # slash mark, evoking HERMETIC
    x0, y0 = size * 0.58, size * 0.30
    for i in range(2):
        x = x0 + i * size * 0.09
        d.line([(x, y0), (x - size * 0.06, y0 + size * 0.40)], fill=(222, 190, 110, 255), width=max(2, size // 56))
    img.save(path)

for s in (16, 32, 64, 128, 256, 512):
    make(s, os.path.join(OUT, f"icon_{s}.png"))
if os.path.exists(os.path.join(OUT, "icon_512.png")):
    make(512, os.path.join(OUT, "icon_512@2x.png"))
PY
iconutil -c icns "$ICONSET" -o "$BUNDLE/Contents/Resources/icon.icns"

# ── codesign (ad-hoc) so Gatekeeper is quiet ──────────────────────────────
codesign --force --deep --sign - "$BUNDLE" 2>/dev/null || true

# ── the double-clickable shortcut on the Desktop ──────────────────────────
# The bundle is installed into ~/Applications first, so the shortcut and the
# bundle can never disagree about which one is the real one.
INSTALLED="$HOME/Applications/HERMETIC.app"
rm -rf "$INSTALLED"
mkdir -p "$HOME/Applications"
cp -R "$BUNDLE" "$INSTALLED"
codesign --force --deep --sign - "$INSTALLED" 2>/dev/null || true

DESKTOP="$HOME/Desktop/HERMETIC.command"
cat > "$DESKTOP" <<DESKCMD
#!/bin/sh
# HERMETIC
exec "$INSTALLED/Contents/MacOS/hermetic"
DESKCMD
chmod +x "$DESKTOP"

echo "built: $BUNDLE"
echo "installed: $INSTALLED"
echo "shortcut:  $DESKTOP"
ls -la "$BUNDLE/Contents/MacOS/"
