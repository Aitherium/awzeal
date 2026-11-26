#!/usr/bin/env python3
"""Fix the [0xDA48] deadlock: make the spin's poll exit unconditionally.
Linear 0x1BDE2 (file 0x16202): 48 85 C0 75 07 (test rax,rax; jnz 0x1bdee)
-> 90 90 90 EB 07 (nop nop nop; jmp 0x1bdee) — always exit the wait."""
import subprocess

KERN = "/tmp/kernel_patched2.zxe"
IMG = "/dev/shm/ZealOSH.raw"
LBA = 0x2B3AD
off = 0x16202  # linear 0x1BDE2 in piece e2 (lin 0x17BE0, file 0x10000)

data = bytearray(open(KERN, "rb").read())
cur = bytes(data[off:off + 5])
print("current:", cur.hex())
assert cur == bytes.fromhex("4885c07507"), f"unexpected: {cur.hex()}"
data[off:off + 5] = bytes.fromhex("909090eb07")
open(KERN, "wb").write(data)
print("patched:", bytes(data[off:off + 5]).hex())

# verify disasm
blob = bytes(data[off - 8:off + 20])
open("/tmp/spinpatch.bin", "wb").write(blob)
r = subprocess.run(["ndisasm", "-b64", "-o", "0x1bdda", "/tmp/spinpatch.bin"],
                   capture_output=True, text=True)
print(r.stdout)

# embed
img = bytearray(open(IMG, "rb").read())
ioff = LBA * 512
img[ioff:ioff + len(data)] = bytes(data)
open(IMG, "wb").write(img)
back = open(IMG, "rb").read()[ioff:ioff + len(data)]
print("embedded; round-trip:", back == bytes(data))
print("image at file offset:", open(IMG, "rb").read()[ioff + off:ioff + off + 5].hex())
