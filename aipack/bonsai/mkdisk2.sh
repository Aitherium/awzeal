#!/bin/bash
# mkdisk2.sh -- build bzdisk.img: MBR + one FAT32 partition holding the Bonsai pack
#   ENGINE2.BIN, AILLM.ZC (the shared front end), AIBONSAI.ZC, RUN2.ZC, BZ000.BIN ...
# The GGUF is split at tensor boundaries (tools/split_gguf.py) into <= 1 GiB parts,
# each streamed into the image and deleted, so scratch never holds more than one part.
# Boot it like aidisk.img with -m 12G or more (7.3 GiB weights + 640 MiB arena + OS).
set -euo pipefail
W=${ZWORK:-/root/zealos-ai/bonsai}
G=${BZ_GGUF:-/srv/aither-models/bonsai2-27b/Ternary-Bonsai-2.5-27B-PQ2_0.gguf}
IMG=$W/bzdisk.img
STAGE=$W/stage
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$W/out/ENGINE2.BIN" "$STAGE/"
cp "$W/aipack/zealos/AILLM.ZC" "$W/aipack/bonsai/zealos/AIBONSAI.ZC" "$W/aipack/bonsai/zealos/RUN2.ZC" "$STAGE/"
NEED=$(python3 "$W/aipack/bonsai/tools/split_gguf.py" "$G" | awk '/model bytes used/{print $6}')
SIZE_MB=$(( NEED / 1048576 + 512 ))
rm -f "$IMG"; truncate -s ${SIZE_MB}M "$IMG"
podman run --rm -v "$W:/w" -v "$(dirname "$G"):/m:ro" localhost/zealos-bonsai-build:1 bash -c '
  set -e
  apt-get update -qq >/dev/null && apt-get install -y -qq dosfstools mtools fdisk >/dev/null 2>&1
  echo "2048,,c,*" | sfdisk -q /w/bzdisk.img
  OFF=$((2048*512)); SECTORS=$(( $(stat -c %s /w/bzdisk.img)/512 - 2048 ))
  mkfs.fat -F 32 -n BONSAI --offset 2048 /w/bzdisk.img $((SECTORS/2)) >/dev/null
  export MTOOLS_SKIP_CHECK=1
  for f in /w/stage/*; do mcopy -i /w/bzdisk.img@@$OFF "$f" ::/; done
  printf "#!/bin/sh\nmcopy -i /w/bzdisk.img@@%s \"\$1\" ::/\n" $OFF > /tmp/put.sh; chmod +x /tmp/put.sh
  python3 /w/aipack/bonsai/tools/split_gguf.py /m/'"$(basename "$G")"' --out /w/stage/parts --each /tmp/put.sh
  mdir -i /w/bzdisk.img@@$OFF ::/
'
