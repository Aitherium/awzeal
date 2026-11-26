#!/usr/bin/env python3
"""ppm2png.py <in.ppm> <out.png> -- stdlib-only P6 PPM to PNG (QEMU screendump viewer)."""
import struct
import sys
import zlib


def main(src: str, dst: str) -> int:
    data = open(src, "rb").read()
    parts = data.split(b"\n", 3)
    if parts[0] != b"P6":
        print("not a P6 ppm", file=sys.stderr)
        return 2
    w, h = map(int, parts[1].split())
    px = parts[3]
    raw = b"".join(b"\x00" + px[y * w * 3:(y + 1) * w * 3] for y in range(h))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    open(dst, "wb").write(png)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
