#!/usr/bin/env python3
"""gguf_dump.py <gguf> -- every key (arrays abbreviated) and every tensor (name dims type offset)."""
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
        return ("arr", et, n, [rval(f, et) for _ in range(n)])
    raise ValueError(t)


def main(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, nt, nkv = rd(f, "<I"), rd(f, "<Q"), rd(f, "<Q")
        print("version", ver, "tensors", nt, "kv", nkv)
        align = 32
        for _ in range(nkv):
            k = rstr(f)
            v = rval(f, rd(f, "<I"))
            if k == "general.alignment":
                align = v
            if isinstance(v, tuple):
                _, et, n, vals = v
                print(f"{k} = [arr et={et} n={n}] {repr(vals[:8])[:300]}")
            else:
                print(f"{k} = {str(v)[:300]}")
        for _ in range(nt):
            name = rstr(f)
            nd = rd(f, "<I")
            dims = [rd(f, "<Q") for _ in range(nd)]
            typ = rd(f, "<I")
            off = rd(f, "<Q")
            print("T", name, dims, typ, off)
        pos = f.tell()
        print("data_start", (pos + align - 1) // align * align, "align", align)


if __name__ == "__main__":
    main(sys.argv[1])
