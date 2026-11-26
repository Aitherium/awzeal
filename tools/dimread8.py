#!/usr/bin/env python3
"""Decode the ZealOS VESA 720x400 screen: 90 cols x 25 rows, 8x16 cells at
the 8px pitch (NOT 9 — the earlier failure used 9px and misaligned every
glyph). FontStd.ZC rows are LSB-first; flip for comparison."""
import re
import sys
from PIL import Image

PPM = sys.argv[1]
FONT = "/mnt/c/AitherOS-Fresh/ZealOS-AI/src/Kernel/FontStd.ZC"
text = open(FONT, encoding="cp437").read()
hexes = re.findall(r"0x([0-9A-Fa-f]{16})", text)


def flip_row(r):
    return int(f"{r:08b}"[::-1], 2)


glyphs = [[flip_row((int(hx, 16) >> (8 * i)) & 0xFF) for i in range(16)] for hx in hexes]

im = Image.open(PPM).convert("L")
w, h = im.size
px = list(im.getdata())


def decode(xo, yo, thresh):
    lines = []
    for gy in range(25):
        line = ""
        for gx in range(90):
            bits = []
            for r in range(16):
                rb = 0
                for dx in range(8):
                    xx = gx * 8 + xo + dx
                    yy = gy * 16 + yo + r
                    if xx < w and yy < h and px[yy * w + xx] > thresh:
                        rb |= 0x80 >> dx
                bits.append(rb)
            bestg, berr = 0, 999
            for gi, g in enumerate(glyphs):
                err = sum(1 for a, b in zip(bits, g) if (a & 0xFF) != (b & 0xFF))
                if err < berr:
                    berr, bestg = err, gi
            if berr > 12:
                line += " "
            else:
                line += chr(bestg) if 32 <= bestg < 127 else "?"
        lines.append(line)
    return lines


best = None
for xo in range(8):
    for yo in range(4):
        for th in (45, 90, 140):
            lines = decode(xo, yo, th)
            score = sum(1 for line in lines if line.strip())
            if best is None or score > best[0]:
                best = (score, xo, yo, th)
print("best:", best, flush=True)
for line in decode(best[1], best[2], best[3]):
    if line.strip():
        print(line.rstrip())
