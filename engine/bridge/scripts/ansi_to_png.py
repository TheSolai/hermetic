#!/usr/bin/env python3
"""Render a captured ANSI terminal screen to a PNG.

Usage: ansi_to_png.py <ansi-capture> <out.png> [title]
"""
import re
import sys
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Menlo.ttc"
SIZE = 15
PAD = 18
BG = (16, 18, 22)
FG = (208, 214, 222)
TITLEBAR = (34, 38, 46)

BASE = {
    30: (60, 64, 72), 31: (233, 106, 106), 32: (140, 200, 130), 33: (222, 190, 110),
    34: (110, 160, 235), 35: (200, 140, 220), 36: (110, 195, 205), 37: (208, 214, 222),
    90: (110, 118, 130), 91: (245, 140, 140), 92: (165, 220, 150), 93: (240, 210, 130),
    94: (140, 185, 245), 95: (215, 165, 235), 96: (140, 215, 225), 97: (245, 248, 252),
}
DIM = (128, 136, 148)
BOLD = (245, 248, 252)
GREEN = (126, 214, 145)
ITALIC = (160, 168, 180)


def parse(line):
    """-> list of (text, color, bold, dim)"""
    spans = []
    color, bold, dim = FG, False, False
    i = 0
    while i < len(line):
        m = re.match(r"\x1b\[([0-9;]*)m", line[i:])
        if m:
            for code in (m.group(1) or "0").split(";"):
                c = int(code or 0)
                if c == 0:
                    color, bold, dim = FG, False, False
                elif c == 1:
                    bold = True
                elif c == 2:
                    dim = True
                elif c == 22:
                    bold = dim = False
                elif c == 3:
                    color = ITALIC
                elif c == 39:
                    color = FG
                elif c in BASE:
                    color = BASE[c]
            i += m.end()
            continue
        nxt = line.find("\x1b[", i)
        if nxt == -1:
            nxt = len(line)
        text = line[i:nxt]
        if text:
            spans.append((text, color, bold, dim))
        i = nxt
    return spans


def main():
    src, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "HERETICAL"

    raw = open(src, encoding="utf-8", errors="replace").read()
    lines = raw.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()

    font = ImageFont.truetype(FONT, SIZE)
    cw = font.getlength("M")
    ch = SIZE + 6

    cols = max((len(re.sub(r"\x1b\[[0-9;]*m", "", l)) for l in lines), default=80)
    title_h = 34
    W = int(PAD * 2 + cw * cols)
    H = int(PAD * 2 + title_h + ch * len(lines))

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # window chrome
    d.rectangle([0, 0, W, title_h], fill=TITLEBAR)
    for idx, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + idx * 18, 12, 24 + idx * 18, 22], fill=c)
    d.text((80, 9), title, font=ImageFont.truetype(FONT, 13), fill=(150, 158, 170))

    y = PAD + title_h
    for line in lines:
        x = PAD
        for text, color, bold, dim in parse(line):
            f = ImageFont.truetype(FONT, SIZE)
            if dim:
                color = DIM
            if bold:
                color = BOLD
            d.text((x, y), text, font=f, fill=color)
            x += cw * len(text)
        y += ch

    img.save(out)
    print(f"wrote {out} ({W}x{H})")


if __name__ == "__main__":
    main()
