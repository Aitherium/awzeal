#!/bin/bash
# Build ENGINE.BIN (llm.asm Qwen3 engine + ZealOS glue) inside a throwaway container.
# usage: build.sh <llm.asm checkout> <aipack dir> <out dir>
set -euo pipefail
LLM=$1; PACK=$2; OUT=$3
mkdir -p "$OUT"
podman run --rm -v "$LLM:/llm:ro" -v "$PACK:/pack:ro" -v "$OUT:/out" docker.io/library/python:3.12-slim bash -c '
  set -e
  command -v nasm >/dev/null || { apt-get update -qq >/dev/null && apt-get install -y -qq nasm binutils >/dev/null; }
  B=/tmp/zb; mkdir -p $B
  F="-f elf64 -DBARE_METAL -DFAST_Q8=1 -I/llm/shared/"
  for m in math matmul tokenizer transformer; do nasm $F /llm/shared/$m.asm -o $B/$m.o; done
  nasm $F /pack/engine/zeal_glue.asm -o $B/glue.o
  OBJS="$B/glue.o $B/math.o $B/matmul.o $B/tokenizer.o $B/transformer.o"
  for base in 0x0 0x10000000; do
    ld -m elf_x86_64 --defsym=ZL_BASE=$base -T /pack/engine/zeal.ld -z max-page-size=0x40 -z noexecstack $OBJS -o $B/img_$base.elf
    objcopy -O binary $B/img_$base.elf $B/img_$base.bin
  done
  nm -n $B/img_0x0.elf | grep -E " (zl_|model_init|g_ctx)" | head
  python3 /pack/engine/pack_engine.py $B/img_0x0.bin $B/img_0x10000000.bin 0x10000000 /out/ENGINE.BIN
'
