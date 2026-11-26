#!/bin/bash
# proof_mc.sh -- multi-core proof on Linux: the same bz_worker split ZealOS runs, on pthreads.
# Prompt parity + the first 8 greedy tokens must still equal the llama.cpp reference, and
# 4 cores should decode faster than 1. Both runs read the GGUF mmapped (warm the cache first).
set -euo pipefail
B=/mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh
REF=248045,846,198,3710,369,279,6511,314,9338,30,248046,198,248045,74455,198,248068,271,248069,271
WANT="760,6511,314,9338,369,2972,57590,159034"
bash $B build | tail -2
cat /srv/aither-models/bonsai2-27b/Ternary-Bonsai-2.5-27B-PQ2_0.gguf > /dev/null   # warm the page cache
for spec in "15:1" "12-15:4"; do
  cpus=${spec%%:*}; n=${spec##*:}
  echo "== cores $n (cpuset $cpus)"
  BZ_CPU=$cpus BZ_THREADS=$n bash $B chat "What is the capital of France?" 8 $REF 2>&1 \
    | grep -E "threads=|PARITY|greedy|tok/s|prefill" | sed "s/^/  /"
done
echo "want greedy $WANT"
