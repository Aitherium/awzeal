#!/bin/bash
# inspect_disk.sh -- what the Bonsai guest will actually load: the engine id at header +56 and
# the loader's id check, read back OFF the FAT32 disk image (not from the build dir).
W=/root/zealos-ai
podman run --rm -v "$W:/w" localhost/zealos-bonsai-build:1 bash -c '
  command -v mcopy >/dev/null || { apt-get update -qq >/dev/null && apt-get install -y -qq mtools >/dev/null 2>&1; }
  export MTOOLS_SKIP_CHECK=1
  mcopy -o -i /w/bonsai/bzdisk.img@@1048576 ::/ENGINE2.BIN /tmp/e.bin
  mcopy -o -i /w/bonsai/bzdisk.img@@1048576 ::/AIBONSAI.ZC /tmp/a.zc
  python3 - <<EOF
b = open("/tmp/e.bin", "rb").read()
print("disk ENGINE2", len(b), "id@+56", b[64 + 56:64 + 64], "id@+32", b[64 + 32:64 + 40])
EOF
  grep -n BONSAI25 /tmp/a.zc
  ls -la /w/bonsai/out/ENGINE2.BIN
'
