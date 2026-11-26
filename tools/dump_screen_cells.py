#!/usr/bin/env python3
"""Dump glyph cells of a ZealOS VESA PPM as ASCII art for eyeball reading.
Usage: dump_screen_cells.py <screen.ppm> [xoff] [thresh]

Reads the PPM (P6) directly, finds the content row bands, and prints every
glyph cell as 8x16 ASCII art so the rendered text can be read by eye — no
font matching (FontStd matching failed at every geometry on these screens).
"""
import sys

path = sys.argv[1]
TH = int(sys.argv[2]) if len(sys.argv) > 2 else 100

data = open(path, "rb").read()
assert data[:2] == b"P6"
pos = data.find(b"\n", 2) + 1
line = data[pos:data.find(b"\n", pos)]
w, h = map(int, line.split())
pos = data.find(b"\n", pos) + 1
px = data[pos:]


def get(x, y):
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0
    o = (y * w + x) * 3
    return px[o]


rows = [y for y in range(h) if any(get(x, y) > TH for x in range(w))]
bands = []
for y in rows:
    if bands and y - bands[-1][-1] <= 3:
        bands[-1].append(y)
    else:
        bands.append([y])
print(f"image {w}x{h}, content bands: {[(b[0], b[-1], len(b)) for b in bands]}")

for bi, band in enumerate(bands):
    y0 = band[0]
    print(f"\n===== band {bi}: rows {y0}-{band[-1]} ({len(band)} px tall) =====")
    xcols = [x for x in range(w) if any(get(x, y) > TH for y in band)]
    xruns = []
    for x in xcols:
        if xruns and x - xruns[-1][-1] <= 2:
            xruns[-1].append(x)
        else:
            xruns.append([x])
    print("x runs:", [(r[0], r[-1]) for r in xruns])
    for xi, run in enumerate(xruns):
        x0 = run[0]
        gx0 = (x0 // 8) * 8  # align to the 8px glyph grid
        print(f"--- run {xi}: x={x0}-{run[-1]} (cell x0={gx0}) ---")
        for yy in range(16):
            row = ""
            for xx in range(gx0, min(gx0 + 8, w)):
                row += "#" if get(xx, y0 + yy) > TH else "."
            if row.strip("."):
                print(row)
