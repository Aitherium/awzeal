#!/bin/bash
# mkdisk.sh -- build aidisk.img: MBR + one FAT32 partition holding the AI pack.
# ZealOS mounts it as a second SATA drive. Files use 8.3 names.
set -euo pipefail
W=/root/zealos-ai
IMG=$W/aidisk.img
STAGE=$W/stage
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$W/out/ENGINE.BIN" "$STAGE/ENGINE.BIN"
for f in AILLM RUN AIKERNEL AIINFER AITOOLS; do cp "$W/aipack/zealos/$f.ZC" "$STAGE/"; done
[ -s "$STAGE/QWEN3.BIN" ] || ln -f "$W/models/qwen3-0.6b-q8_0.asmbin" "$STAGE/QWEN3.BIN" 2>/dev/null || cp "$W/models/qwen3-0.6b-q8_0.asmbin" "$STAGE/QWEN3.BIN"
rm -f "$IMG"
truncate -s 1200M "$IMG"
podman run --rm -v "$W:/w" docker.io/library/debian:12 bash -c '
  set -e
  apt-get update -qq >/dev/null && apt-get install -y -qq dosfstools mtools fdisk >/dev/null 2>&1
  echo "2048,,c,*" | sfdisk -q /w/aidisk.img
  OFF=$((2048*512))
  SECTORS=$(( $(stat -c %s /w/aidisk.img)/512 - 2048 ))
  mkfs.fat -F 32 -n AIPACK --offset 2048 /w/aidisk.img $((SECTORS/2)) >/dev/null
  export MTOOLS_SKIP_CHECK=1
  for f in /w/stage/*; do mcopy -i /w/aidisk.img@@$OFF "$f" ::/; done
  mdir -i /w/aidisk.img@@$OFF ::/
'
