#!/usr/bin/env python3
"""Extract unique glyph masks from the freeze screen, print each once with its
positions, then render the lines with a user-supplied mapping.
Usage:
  glyph_render.py <screen.ppm> extract   -> print unique glyphs + positions
  glyph_render.py <screen.ppm> render "0:S 1:e 2:r ..." -> render lines using
    comma/space-separated maskidx:letter pairs (letters outside 32-126 -> ?)
"""
import sys

path = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "extract"
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


def cell_mask(y0, cx):
    m = 0
    for yy in range(16):
        for xx in range(9):
            if get(cx + xx, y0 + yy) > TH:
                m |= 1 << (yy * 9 + xx)
    return m


def cell_art(m):
    out = []
    for yy in range(16):
        row = ""
        for xx in range(9):
            row += "#" if m & (1 << (yy * 9 + xx)) else "."
        out.append(row)
    return out


cells = []  # (band, cellidx, mask)
for bi, band in enumerate(bands):
    y0 = band[0]
    xcols = [x for x in range(w) if any(get(x, y) > TH for y in band)]
    xmin, xmax = min(xcols), max(xcols)
    gx0 = (xmin // 9) * 9
    ng = (xmax - gx0) // 9 + 2
    for ci in range(ng):
        cells.append((bi, ci, cell_mask(y0, gx0 + ci * 9)))

if MODE == "extract":
    uniq = {}
    for bi, ci, m in cells:
        if m and m not in uniq:
            uniq[m] = []
        if m:
            uniq[m].append((bi, ci))
    for i, (m, locs) in enumerate(uniq.items()):
        print(f"=== G{i}: used at {locs[:6]}{'...' if len(locs) > 6 else ''} ===")
        for r in cell_art(m):
            print(r)
else:
    mapping = {}
    for tok in sys.argv[3].split():
        if ":" in tok:
            k, v = tok.split(":", 1)
            mapping[int(k)] = v
    for bi in range(len(bands)):
        line = ""
        for ci in range(90):
            m = next((c[2] for c in cells if c[0] == bi and c[1] == ci), 0)
            if not m:
                line += " "
                continue
            gi = None
            for k in mapping:
                pass
            # find which glyph index this mask is
            uniq = {}
            for bb, cc, mm in cells:
                if mm and mm not in uniq:
                    uniq[mm] = len(uniq)
            line += mapping.get(uniq[m], "?")
        print(f"L{bi}: {line.rstrip()}")
