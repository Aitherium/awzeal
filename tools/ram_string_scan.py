#!/usr/bin/env python3
"""Scan a guest-RAM dump for candidate state strings, printing offsets + context.
Usage: ram_string_scan.py <ram.bin> [--all]

The freeze screen can't be font-decoded reliably; the same strings that got
rendered usually also exist as C strings in RAM. Print each hit's offset and
~96 bytes of printable context so the state can be identified.
"""
import mmap
import sys

path = sys.argv[1]
ALL = "--all" in sys.argv

PATTERNS = [
    b"Press N to skip",
    b"boot-fail-wait",
    b"Once",
    b"AutoFullDistro",
    b"MyDistro.ISO.C",
    b"RunFile",
    b"Reboot",
    b"Exception",
    b"LexExcept",
    b"Error",
    b"KERNEL.ZXE",
    b"Kernel.ZXE",
    b"SyscallInit",
    b"FluxEmit",
    b"MakeMyISO",
    b"MakeSystem",
    b"CC\n",
    b"Sleep(700)",
    b"OnceDrive",
    b"BootMHDIns",
    b"Stack",
    b"IRQ",
    b"reserved",
    b"SCAN",
    b"ZXE",
]

PR = set(bytes(range(32, 127)))


def clean(b):
    return "".join(chr(c) if c in PR else "." for c in b)


with open(path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for pat in PATTERNS:
        hits = []
        start = 0
        while True:
            i = mm.find(pat, start)
            if i < 0:
                break
            hits.append(i)
            start = i + 1
        print(f"\n### {pat!r}: {len(hits)} hit(s)")
        for i in hits[:8]:
            lo = max(0, i - 48)
            hi = min(len(mm), i + len(pat) + 48)
            print(f"  @0x{i:08x}: {clean(mm[lo:hi])}")
        if len(hits) > 8:
            print(f"  ... {len(hits) - 8} more")
    mm.close()
