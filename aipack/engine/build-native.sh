#!/bin/bash
# build-native.sh <llm.asm checkout> <aipack dir> <out dir>
# Build ENGINE.BIN (llm.asm Qwen3 engine + ZealOS glue). Needs nasm, binutils, python3.
# Used by the awzeal Containerfile and by build.sh (which wraps it in a container).
set -euo pipefail
LLM=$1; PACK=$2; OUT=$3
B=$(mktemp -d)
mkdir -p "$OUT"
F="-f elf64 -DBARE_METAL -DFAST_Q8=1 -I$LLM/shared/"
for m in math matmul tokenizer transformer; do nasm $F "$LLM/shared/$m.asm" -o "$B/$m.o"; done
nasm $F "$PACK/engine/zeal_glue.asm" -o "$B/glue.o"
OBJS="$B/glue.o $B/math.o $B/matmul.o $B/tokenizer.o $B/transformer.o"
for base in 0x0 0x10000000; do
  ld -m elf_x86_64 --defsym=ZL_BASE=$base -T "$PACK/engine/zeal.ld" -z max-page-size=0x40 -z noexecstack \
     $OBJS -o "$B/img_$base.elf" 2>/dev/null
  objcopy -O binary "$B/img_$base.elf" "$B/img_$base.bin"
done
python3 "$PACK/engine/pack_engine.py" "$B/img_0x0.bin" "$B/img_0x10000000.bin" 0x10000000 "$OUT/ENGINE.BIN"
rm -rf "$B"
