#!/usr/bin/env python3
"""check_iso_kernel.py <iso> — Stage-1 acceptance: does the built ISO carry
the FORK kernel (the AI files compiled), or the STOCK kernel (the chain's
CC stages never ran — the queued-In() CC killed by the stage Reboot)?

The ZealOS ISO is ISO9660; the kernel lives at /Boot/Kernel.ZXE. This
reads the ISO9660 directory records directly (no 7z/isoinfo dependency),
extracts the kernel, and scans it for the fork-AI string literals (the
compiled CapEngine/FluxIDT/AgentForge/InferQueue/Syscall code carries
them). The stock kernel has NONE of these strings.

Exit 0 = fork markers found; 1 = not found; 2 = could not read the ISO.
"""
import struct
import sys

ISO = sys.argv[1] if len(sys.argv) > 1 else None
if not ISO:
    print("usage: check_iso_kernel.py <ZealOS-MyDistro.iso>")
    sys.exit(2)

# Fork-kernel marker strings (grep'd from the fork's Kernel AI files,
# 2026-08-28). The stock kernel contains none of these.
MARKERS = [
    b"FluxDump: invalid vector",     # FluxIDT.ZC
    b"ForgeDispatch depth exceeded", # AgentForge.ZC
    b"=== Inference Queue",          # InferQueue.ZC
    b"SyscallInit: INT 0x80",        # Syscall.ZC
    b"Models registered",            # InferQueue.ZC
]


def read_sectors(f, start, count):
    f.seek(start * 2048)
    return f.read(count * 2048)


def find_kernel(f):
    """Locate /Boot/Kernel.ZXE via the directory walk."""
    pvd = read_sectors(f, 16, 1)
    if pvd[:6] != b"\x01CD001":
        return None
    root_len = struct.unpack_from("<I", pvd, 156)[0]
    root_start = struct.unpack_from("<I", pvd, 158)[0]
    root = read_sectors(f, root_start, (root_len + 2047) // 2048)[:root_len]
    queue = [("/", root)]
    while queue:
        path, data = queue.pop(0)
        i = 0
        while i < len(data):
            rec_len = data[i]
            if rec_len == 0:
                i += 1
                continue
            rec = data[i:i + rec_len]
            if len(rec) < 34:
                i += rec_len
                continue
            ext_len = rec[32]
            name = rec[33:33 + ext_len].decode("latin-1").rstrip(";1")
            flags = rec[25]
            size = struct.unpack_from("<I", rec, 10)[0]
            start = struct.unpack_from("<I", rec, 2)[0]
            if name in ("\x00", "\x01"):
                i += rec_len
                continue
            full = path + name
            if flags & 0x02:
                d = read_sectors(f, start, (size + 2047) // 2048)[:size]
                queue.append((full + "/", d))
            elif full == "/Boot/Kernel.ZXE":
                return start, size
            i += rec_len
    return None


try:
    with open(ISO, "rb") as f:
        hit = find_kernel(f)
        if not hit:
            print(f"NOT FOUND: /Boot/Kernel.ZXE in {ISO} (ISO9660 walk failed)")
            sys.exit(2)
        start, size = hit
        print(f"kernel: /Boot/Kernel.ZXE at sector {start}, {size} bytes")
        f.seek(start * 2048)
        kernel = f.read(size)
except OSError as e:
    print(f"ERROR: {e}")
    sys.exit(2)

found = [m for m in MARKERS if m in kernel]
print(f"fork-AI markers: {len(found)}/{len(MARKERS)}")
for m in MARKERS:
    print(f"  {'FOUND' if m in kernel else 'absent'}: {m.decode()}")
if found:
    print("RESULT: FORK kernel (the CC stages ran — the AI files compiled)")
    sys.exit(0)
print("RESULT: STOCK kernel (the CC stages never compiled — the queued-In() CC")
print("        was killed by the stage Reboot; the chain advanced without the")
print("        kernel pass)")
sys.exit(1)
