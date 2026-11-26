#!/usr/bin/env python3
"""Patch the E820 scan (linear 0xA64F-0xA678, file 0x2E6F-0x2E98) to skip
entries with base.hi or len.hi != 0 (anything touching >=4GB — the garbage
entry[1] from the QEMU E820 stream). Same byte budget (42 bytes) — no
shifting. Then embed into the image and verify."""
import sys

KERN = "/tmp/kernel_patched2.zxe"
IMG = "/dev/shm/ZealOSH.raw"
LBA = 0x2B3AD

data = bytearray(open(KERN, "rb").read())

# expected current bytes at 0x2E6F (linear 0xA64F): the previous session's BP patch
off = 0x2E6F
cur = bytes(data[off:off + 42])
exp_cur = bytes.fromhex(
    "662e8b4e04"   # mov ecx,[cs:bp+0x4]
    "662e035e08"   # add ebx,[cs:bp+0x8]
    "662e134e0c"   # adc ecx,[cs:bp+0xc]
    "662bd8"       # sub ebx,eax
    "661bca"       # sbb ecx,edx
    "7213"         # jc 0xa679
    "668b4600"     # mov eax,[bp+0x0]
    "662e8b5604"   # mov edx,[cs:bp+0x4]
    "662e034608"   # add eax,[cs:bp+0x8]
    "662e13560c"   # adc edx,[cs:bp+0xc]
)
print("current:", cur.hex())
print("expect :", exp_cur.hex())
if cur != exp_cur:
    print("MISMATCH — aborting")
    sys.exit(1)

# the new bytes: skip entries with base.hi or len.hi nonzero
new = bytes.fromhex(
    "662e837e0400"   # cmp dword [cs:bp+0x4], 0      (base.hi)
    "7522"           # jne 0xa679                    (skip)
    "662e837e0c00"   # cmp dword [cs:bp+0xc], 0      (len.hi)
    "751a"           # jne 0xa679                    (skip)
    "662e035e08"     # add ebx,[cs:bp+0x8]
    "662bd8"         # sub ebx,eax
    "7210"           # jc 0xa679                     (smaller -> skip)
    "668b4600"       # mov eax,[bp+0x0]
    "6633d2"         # xor edx,edx                   (base.hi==0 guaranteed)
    "662e034608"     # add eax,[cs:bp+0x8]
    "90909090"       # nop x4 (pad to 0xA679)
)
assert len(new) == 42, len(new)
data[off:off + 42] = new
open(KERN, "wb").write(data)
print("patched 42 bytes at", hex(off))

# verify with ndisasm
import subprocess
blob = bytes(data[0x2E60:0x2EA0])
open("/tmp/e820_new.bin", "wb").write(blob)
r = subprocess.run(["ndisasm", "-b16", "-o", "0xa640", "/tmp/e820_new.bin"],
                   capture_output=True, text=True)
print(r.stdout)

# embed into the image
img = bytearray(open(IMG, "rb").read())
ioff = LBA * 512
img[ioff:ioff + len(data)] = bytes(data)
open(IMG, "wb").write(img)
back = open(IMG, "rb").read()[ioff:ioff + len(data)]
print("embedded; round-trip:", back == bytes(data))
