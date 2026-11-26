#!/usr/bin/env python3
"""split_gguf.py <gguf> [--part-mb N] [--out DIR] [--prefix BZ] [--each CMD]

FAT32 caps a file at 4 GiB - 1 and the Bonsai 2.5 GGUF is 8.3 GB, so the ZealOS
pack carries it as numbered parts (8.3 names BZ000.BIN, BZ001.BIN, ...) that
AIBONSAI.ZC loads into separate allocations and hands to ZlInit as a BZPARTS1
table. Cuts fall only on tensor boundaries, so no tensor straddles two parts and
the engine never needs the parts to be contiguous in memory. Part 0 also holds
the whole GGUF header. Bytes after the last tensor are dropped.

Without --out it prints the plan (cut offsets, part sizes) only. --cuts prints the
comma list the harness takes in BZ_CUTS to prove the part-table path on Linux.
--each runs CMD with the part path after writing it (and deletes the part when CMD
succeeds), so a disk builder can stream 7 GB through 1 GB of scratch.
"""
import argparse
import os
import struct
import subprocess
import sys

TYPE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def skip_val(f, t):
    if t == 8:
        f.seek(rd(f, "<Q"), 1)
    elif t == 9:
        et, n = rd(f, "<I"), rd(f, "<Q")
        if et == 8:
            for _ in range(n):
                f.seek(rd(f, "<Q"), 1)
        else:
            f.seek(TYPE_SIZE[et] * n, 1)
    else:
        f.seek(TYPE_SIZE[t], 1)


def tensor_bytes(dims, typ):
    n = 1
    for d in dims:
        n *= d
    if typ == 0:
        return 4 * n
    if typ == 142:
        return n // 128 * 34
    if typ == 30:
        return 2 * n
    raise ValueError(f"tensor type {typ} has no size rule here")


def plan(path, part_mb):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        rd(f, "<I")
        nt, nkv = rd(f, "<Q"), rd(f, "<Q")
        align = 32
        for _ in range(nkv):
            k = f.read(rd(f, "<Q")).decode()
            t = rd(f, "<I")
            if k == "general.alignment":
                align = rd(f, "<I")
            else:
                skip_val(f, t)
        spans = []
        for _ in range(nt):
            f.seek(rd(f, "<Q"), 1)
            nd = rd(f, "<I")
            dims = [rd(f, "<Q") for _ in range(nd)]
            typ, off = rd(f, "<I"), rd(f, "<Q")
            spans.append((off, off + tensor_bytes(dims, typ)))
        data_start = (f.tell() + align - 1) // align * align
    spans = sorted((data_start + a, data_start + b) for a, b in spans)
    limit = part_mb << 20
    cuts, start = [], 0
    for a, b in spans:
        if b - start > limit and a > start:
            cuts.append(a)
            start = a
        if b - start > limit:
            raise SystemExit(f"tensor at {a} ({b - a} B) is larger than a part")
    end = spans[-1][1]
    bounds = [0] + cuts + [end]
    return data_start, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--part-mb", type=int, default=1024)
    ap.add_argument("--out")
    ap.add_argument("--prefix", default="BZ")
    ap.add_argument("--each")
    ap.add_argument("--cuts", action="store_true")
    a = ap.parse_args()
    data_start, bounds = plan(a.gguf, a.part_mb)
    if a.cuts:
        print(",".join(str(c) for c in bounds[1:-1]))
        return 0
    print(f"data_start {data_start}, model bytes used {bounds[-1]} of {os.path.getsize(a.gguf)}")
    for i in range(len(bounds) - 1):
        print(f"  {a.prefix}{i:03d}.BIN  [{bounds[i]}, {bounds[i + 1]})  {(bounds[i + 1] - bounds[i]) / 2**20:.1f} MiB")
    if not a.out:
        return 0
    os.makedirs(a.out, exist_ok=True)
    with open(a.gguf, "rb") as f:
        for i in range(len(bounds) - 1):
            p = os.path.join(a.out, f"{a.prefix}{i:03d}.BIN")
            f.seek(bounds[i])
            left = bounds[i + 1] - bounds[i]
            with open(p, "wb") as o:
                while left:
                    buf = f.read(min(left, 64 << 20))
                    o.write(buf)
                    left -= len(buf)
            if a.each:
                if subprocess.call(a.each.split() + [p]) != 0:
                    print(f"--each failed on {p}", file=sys.stderr)
                    return 1
                os.remove(p)
            print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
