#!/usr/bin/env python3
"""hadnames.py <gguf> -- the prism.hadamard name lists in full + the non-repeating tensor shapes."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_dump import rd, rstr, rval

with open(sys.argv[1], "rb") as f:
    assert f.read(4) == b"GGUF"
    ver, nt, nkv = rd(f, "<I"), rd(f, "<Q"), rd(f, "<Q")
    for _ in range(nkv):
        k = rstr(f)
        v = rval(f, rd(f, "<I"))
        if k.startswith("prism.") and isinstance(v, tuple) and v[1] == 8:
            names = v[3]
            sfx = sorted(set(n.split(".", 2)[-1] if n.startswith("blk.") else n for n in names))
            print(k, len(names), sfx)
            print("  layers per suffix:", {s: sum(1 for n in names if n.endswith(s)) for s in sfx})
    seen = set()
    for _ in range(nt):
        name = rstr(f)
        nd = rd(f, "<I")
        dims = [rd(f, "<Q") for _ in range(nd)]
        typ = rd(f, "<I")
        rd(f, "<Q")
        key = name.split(".", 2)[-1] if name.startswith("blk.") else name
        if (key, typ) not in seen:
            seen.add((key, typ))
            print("T", name, dims, typ)
