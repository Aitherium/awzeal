#!/usr/bin/env python3
"""patch_once.py <in.iso> <out.iso> <once.zc>

Replace every copy of the live ISO's /Home/Once.ZC body (the file the first
terminal runs at boot) with our autostart, same length, in place. The stock
ISO carries two bodies that both start "//Place this file in /Home" and ask
"Install onto hard drive"; the live T: drive runs the one with "Take Tour",
so both are patched. Space padding keeps every extent and directory record.
"""
import sys

HEAD = b"//Place this file in /Home"


def main(src: str, dst: str, once: str) -> int:
    d = bytearray(open(src, "rb").read())
    new = open(once, "rb").read().replace(b"\r\n", b"\n")
    hits = []
    i = d.find(HEAD)
    while i >= 0:
        e = d.find(b"\x00", i)
        if b"Install onto hard drive" in d[i:e]:
            hits.append((i, e))
        i = d.find(HEAD, i + 1)
    if len(hits) != 2:
        print(f"expected two Once.ZC bodies, found {len(hits)}", file=sys.stderr)
        return 1
    for s, e in hits:
        cap = e - s
        if len(new) > cap:
            print(f"autostart is {len(new)} B, slot at 0x{s:x} is {cap} B", file=sys.stderr)
            return 1
        d[s:e] = new + b" " * (cap - len(new))
        print(f"patched Once.ZC body at 0x{s:x} ({len(new)}/{cap} B)")
    open(dst, "wb").write(d)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
