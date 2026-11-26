#!/usr/bin/env python3
"""pack_engine.py <img_base0.bin> <img_baseD.bin> <delta_hex> <out ENGINE.BIN>

Derive the relocation table by diffing two links of the same objects at base 0
and base DELTA: a field whose value moved by exactly DELTA is absolute.

ENGINE.BIN layout (all little-endian):
  0  "ZENG"   4  u32 image_size   8  u32 n64   12 u32 n32
  16 u32 offsets64[n64]  then u32 offsets32[n32]  then pad to 64
  then the base-0 image (starts with the zl_header, magic "ZLE1").
"""
import struct
import sys


def main(a_path: str, b_path: str, delta_hex: str, out: str) -> int:
    a = open(a_path, "rb").read()
    b = open(b_path, "rb").read()
    delta = int(delta_hex, 16)
    if len(a) != len(b):
        print(f"size mismatch {len(a)} vs {len(b)}", file=sys.stderr)
        return 1
    r64, r32 = [], []
    i = 0
    n = len(a)
    while i < n:
        if a[i] == b[i]:
            i += 1
            continue
        # delta's lowest set byte is byte 3 (0x10000000), so the field starts 3 bytes earlier
        s = i - 3
        if s >= 0 and s + 8 <= n and struct.unpack_from("<Q", b, s)[0] - struct.unpack_from("<Q", a, s)[0] == delta:
            r64.append(s)
            i = s + 8
        elif s >= 0 and s + 4 <= n and struct.unpack_from("<I", b, s)[0] - struct.unpack_from("<I", a, s)[0] == delta:
            r32.append(s)
            i = s + 4
        else:
            print(f"unexplained difference at 0x{i:x}", file=sys.stderr)
            return 1
    if a[:4] != b"ZLE1":
        print("image does not start with ZLE1 header", file=sys.stderr)
        return 1
    hdr = b"ZENG" + struct.pack("<III", len(a), len(r64), len(r32))
    hdr += b"".join(struct.pack("<I", o) for o in r64 + r32)
    hdr += b"\0" * ((-len(hdr)) % 64)
    open(out, "wb").write(hdr + a)
    print(f"ENGINE.BIN image={len(a)} relocs64={len(r64)} relocs32={len(r32)} total={len(hdr) + len(a)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
