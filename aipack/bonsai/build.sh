#!/bin/bash
# build.sh <aipack dir> <out dir> -- build the Bonsai engine inside a throwaway container:
#   out/harness       Linux userspace parity harness (mmap the GGUF)
#   out/ENGINE2.BIN   flat HolyC-ABI image for ZealOS (same container format as ENGINE.BIN)
set -euo pipefail
PACK=$1; OUT=$2
mkdir -p "$OUT"
IMG=localhost/zealos-bonsai-build:1
if ! podman image exists $IMG; then
  printf 'FROM docker.io/library/debian:12\nRUN apt-get update -qq && apt-get install -y -qq gcc nasm binutils python3 >/dev/null && rm -rf /var/lib/apt/lists/*\n' \
    | podman build -q -t $IMG -f - /tmp >/dev/null
fi
podman run --rm -v "$PACK:/pack:ro" -v "$OUT:/out" $IMG bash -c '
  set -e
  B=/tmp/bz; mkdir -p $B
  CF="-O2 -mavx2 -mfma -mf16c -fno-math-errno -fno-tree-loop-distribute-patterns -fno-stack-protector -Wall -Wextra -Wno-unused-parameter"
  FREE="-ffreestanding -fPIE -fno-plt -mno-red-zone -fno-jump-tables -fno-asynchronous-unwind-tables -fno-exceptions -fvisibility=hidden -nostdlib"
  # hosted harness (same engine source, hosted ABI)
  gcc $CF -c /pack/bonsai/engine/bonsai.c -o $B/bonsai_host.o
  gcc $CF -c /pack/bonsai/harness/harness.c -o $B/harness.o
  gcc $B/harness.o $B/bonsai_host.o -o /out/harness -lm
  # freestanding image
  gcc $CF $FREE -c /pack/bonsai/engine/bonsai.c -o $B/bonsai.o
  gcc $CF $FREE -c /pack/bonsai/engine/freestd.c -o $B/freestd.o
  nasm -f elf64 /pack/bonsai/engine/glue2.asm -o $B/glue2.o
  OBJS="$B/glue2.o $B/bonsai.o $B/freestd.o"
  for base in 0x0 0x10000000; do
    ld -m elf_x86_64 --defsym=ZL_BASE=$base -T /pack/engine/zeal.ld -z max-page-size=0x40 -z noexecstack $OBJS -o $B/img_$base.elf
    objcopy -O binary $B/img_$base.elf $B/img_$base.bin
  done
  # the loaders refuse a relocated image, so an absolute or GOT relocation fails the build here
  if readelf -r $B/bonsai.o $B/freestd.o $B/glue2.o | grep -q "R_X86_64_64\|R_X86_64_GOT"; then
    echo "FAIL absolute/GOT relocations:"; readelf -rW $B/bonsai.o $B/freestd.o $B/glue2.o | grep -B4 "R_X86_64_64\|R_X86_64_GOT" | head -20; exit 1
  fi
  nm -n $B/img_0x0.elf | grep -E " (zl_header|bz_init|bz_begin)$"
  python3 /pack/engine/pack_engine.py $B/img_0x0.bin $B/img_0x10000000.bin 0x10000000 /out/ENGINE2.BIN
  ls -la /out
'
