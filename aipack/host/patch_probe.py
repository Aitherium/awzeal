#!/usr/bin/env python3
"""patch_probe.py <in.iso> <out.iso> -- same-length tweak of /Home/Once.ZC's prompt text.
Proves whether in-place ISO edits reach the booted T: drive."""
import sys

d = bytearray(open(sys.argv[1], "rb").read())
old, new = b'"Install onto hard drive "', b'"Install ONTO hard drive "'
i = d.find(old)
assert i > 0 and d.find(old, i + 1) > 0  # two copies; the first is /Home/Once.ZC
d[i:i + len(old)] = new
open(sys.argv[2], "wb").write(d)
print("probe patched at", hex(i))
