#!/usr/bin/env python3
"""FIX the VBE LFB read: file 0x29C6 (linear 0xA5A6).
Current: 66 90 8B 45 28  (nop; movw [di+0x28],ax  — 16-BIT read truncates the
        32-bit LFB 0xFD000000 to 0x0000 -> [0xA0C0]=0 -> the fork's framebuffer
        clear zeroes [0,0x2F43C0) including its own helper -> self-destruction)
Fixed:  90 66 8B 45 28  (nop; mov eax,[di+0x28]  — 32-bit read of the LFB)
Apply to the KERNEL FILE, embed into the image, verify round-trip + disasm."""
import sys, subprocess

KERN = "/tmp/kernel_patched2.zxe"
IMG = "/dev/shm/ZealOSH.raw"
LBA = 0x2B3AD

data = bytearray(open(KERN, "rb").read())
off = 0x29C6
cur = bytes(data[off:off + 5])
print("current:", cur.hex())
assert cur == bytes.fromhex("66908b4528"), f"unexpected bytes: {cur.hex()}"
new = bytes.fromhex("90668b4528")  # nop; mov eax,[di+0x28]
data[off:off + 5] = new
open(KERN, "wb").write(data)
print("patched kernel file:", bytes(data[off:off + 5]).hex())

# verify disasm
blob = bytes(data[0x29C0:0x29F0])
open("/tmp/vbe3.bin", "wb").write(blob)
r = subprocess.run(["ndisasm", "-b16", "-o", "0xa5a0", "/tmp/vbe3.bin"],
                   capture_output=True, text=True)
print(r.stdout)

# embed into the image
img = bytearray(open(IMG, "rb").read())
ioff = LBA * 512
img[ioff:ioff + len(data)] = bytes(data)
open(IMG, "wb").write(img)
back = open(IMG, "rb").read()[ioff:ioff + len(data)]
print("embedded; round-trip:", back == bytes(data))
print("image bytes at LBA:", open(IMG, "rb").read()[ioff + off:ioff + off + 5].hex())
