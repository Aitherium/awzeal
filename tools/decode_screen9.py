#!/usr/bin/env python3
"""Decode the ZealOS freeze screen: 9px horizontal pitch (cells at x=2+9n),
16px vertical cells, glyph content ~10px tall. Prints every cell of every
content band as ASCII art — read the text by eye.
Usage: decode_screen9.py <screen.ppm>
"""
import sys

path = sys.argv[1]
TH = 100
data = open(path, "rb").read()
assert data[:2] == b"P6"
pos = data.find(b"\n", 2) + 1
w, h = map(int, data[pos:data.find(b"\n", pos)].split())
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

# cells at 9px pitch, x0 = 2; each cell 9 wide x 16 tall
for bi, band in enumerate(bands):
    y0 = band[0]
    xcols = [x for x in range(w) if any(get(x, y) > TH for y in band)]
    xmin, xmax = min(xcols), max(xcols)
    print(f"\n===== line {bi}: rows {y0}-{band[-1]}, x {xmin}-{xmax} =====")
    gx0 = (xmin // 9) * 9
    ng = (xmax - gx0) // 9 + 2
    for ci in range(ng):
        cx = gx0 + ci * 9
        art = []
        for yy in range(16):
            row = ""
            for xx in range(cx, cx + 9):
                row += "#" if get(xx, y0 + yy) > TH else "."
            art.append(row)
        if any("#" in r for r in art):
            print(f"-- cell {ci} (x={cx}) --")
            for r in art:
                print(r)
