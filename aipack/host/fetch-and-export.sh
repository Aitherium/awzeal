#!/bin/bash
# Fetch Qwen3-0.6B, export it to llm.asm's .asmbin v2 (Q8_0, fp16 scales),
# and build the llm.asm reference engine + bare-metal OS (the parity baseline).
# Runs on the awnix host; python/nasm work happens in a throwaway container.
set -euo pipefail
W=/root/zealos-ai
M=$W/models/qwen3-0.6b-bf16
mkdir -p "$M"
[ -d "$W/llm.asm" ] || git clone -q --depth 1 https://github.com/alesha-pro/llm.asm "$W/llm.asm"
for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors; do
  if [ ! -s "$M/$f" ]; then
    echo "fetch $f"
    curl -sSfL --retry 3 -o "$M/$f.part" "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/$f"
    mv "$M/$f.part" "$M/$f"
  fi
done
ls -la "$M"
podman run --rm -v "$W:/w" -w /w/llm.asm docker.io/library/python:3.12-slim bash -c '
  set -e
  apt-get update -qq >/dev/null && apt-get install -y -qq nasm binutils make >/dev/null
  pip install -q numpy
  [ -s /w/models/qwen3-0.6b-q8_0.asmbin ] || python3 scripts/export_weights.py \
      --model-dir /w/models/qwen3-0.6b-bf16 --out /w/models/qwen3-0.6b-q8_0.asmbin --scale-dtype fp16
  make inference os FAST_Q8=1
  ls -la build /w/models
'
