#!/usr/bin/env python3
"""gguf_meta.py <file.gguf> -- print GGUF KV metadata (arrays summarised) and tensor type census."""
import collections
import struct
import sys

T = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def rstr(f):
    return f.read(rd(f, "<Q")).decode("utf-8", "replace")


def rval(f, t):
    if t in T:
        return rd(f, T[t])
    if t == 8:
        return rstr(f)
    if t == 9:
        et, n = rd(f, "<I"), rd(f, "<Q")
        vals = [rval(f, et) for _ in range(n)]
        return f"[array type={et} n={n}] " + (repr(vals[:6]) if et != 8 or n < 8 else repr(vals[:3]))
    raise ValueError(f"type {t}")


def main(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, nt, nkv = rd(f, "<I"), rd(f, "<Q"), rd(f, "<Q")
        print("version", ver, "tensors", nt)
        for _ in range(nkv):
            k = rstr(f)
            v = rval(f, rd(f, "<I"))
            if "tokenizer.ggml.merges" in k or "tokenizer.ggml.tokens" in k or "token_type" in k:
                v = str(v)[:80]
            print(f"{k} = {str(v)[:200]}")
        census = collections.Counter()
        names = []
        for _ in range(nt):
            name = rstr(f)
            nd = rd(f, "<I")
            dims = [rd(f, "<Q") for _ in range(nd)]
            typ = rd(f, "<I")
            rd(f, "<Q")
            census[typ] += 1
            names.append((name, dims, typ))
        print("tensor types:", dict(census))
        for n, d, t in names[:40]:
            print(" ", n, d, t)


if __name__ == "__main__":
    main(sys.argv[1])
