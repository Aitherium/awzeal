#!/bin/bash
# q1proof.sh <gguf> [ref only|run only] -- dense qwen3 (Bonsai Q1_0) parity.
#   1. reference: a throwaway CPU llama-server from the PrismML fork image (prism-b12128, -ngl 0,
#      one pinned core, loopback port), tools/ref_greedy.py for prompt ids + 8 greedy tokens and
#      their top-5 logprobs; the container is removed afterwards.
#   2. the harness on 1 core and on 4 cores (BZ_THREADS=4) against those ids/tokens.
# The model is read from an ext4 copy under /root/zealos-ai/models (9p /mnt/d mmaps are I/O bound).
set -uo pipefail
SRC=$1; MODE=${2:-all}
T=/mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai
M=/root/zealos-ai/models
mkdir -p $M
G=$M/$(basename "$SRC")
[ -s "$G" ] && [ "$(stat -c %s "$G")" = "$(stat -c %s "$SRC")" ] || cp "$SRC" "$G"
REF=$T/tools/ref_france_$(basename "$G" .gguf).json
for f in $T/*.sh; do sed -i 's/\r$//' "$f"; done
if [ "$MODE" != run ]; then
  N=bz-refq1-$$
  podman run -d --rm --pull=never --name $N --cpuset-cpus=15 -p 127.0.0.1:18298:8080 -v $M:/m:ro \
    ghcr.io/aitherium/llamacpp-prism:prism-b12128 -m /m/$(basename "$G") -ngl 0 -t 1 -c 512 -np 1 \
    --host 0.0.0.0 --port 8080 --no-webui >/dev/null
  for i in $(seq 1 120); do curl -s --max-time 2 http://127.0.0.1:18298/health | grep -q ok && break; sleep 2; done
  python3 $T/tools/ref_greedy.py http://127.0.0.1:18298 "$REF"
  podman logs $N 2>&1 | grep -iE "system_info|CPU_REPACK|type +q1_0|rope|yarn" | head -8
  podman stop -t 2 $N >/dev/null 2>&1
fi
[ "$MODE" = ref ] && exit 0
bash $T/bz.sh build 2>&1 | grep -E "KERNEL-PARITY|relocs|error|FAIL"
IDS=$(python3 -c "import json;print(','.join(map(str,json.load(open('$REF'))['prompt_ids'])))")
WANT=$(python3 -c "import json;print(','.join(map(str,json.load(open('$REF'))['tokens'])))")
cat $G > /dev/null
for spec in 15:1 12-15:4; do
  cpus=${spec%%:*}; n=${spec##*:}
  echo "== $(basename $G) cores $n (cpuset $cpus)"
  BZ_GGUF=$G BZ_CPU=$cpus BZ_THREADS=$n bash $T/bz.sh chat "What is the capital of France?" 8 $IDS 2>&1 \
    | grep -E "rc=|PARITY|prefill|greedy|tok/s|top5"
done
echo "want greedy $WANT"
