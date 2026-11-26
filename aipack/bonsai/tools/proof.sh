#!/bin/bash
# proof.sh -- build, then the two remaining proofs on Linux:
#   1. the ZlBegin/ZlNext/ZlEnd session path (what the ZealOS glue calls), two turns
#   2. the BZPARTS1 part-table path, model mmapped as separate parts cut where split_gguf.py cuts
set -euo pipefail
B=/mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh
G=/srv/aither-models/bonsai2-27b/Ternary-Bonsai-2.5-27B-PQ2_0.gguf
bash $B build | tail -6
python3 /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/tools/split_gguf.py $G
CUTS=$(python3 /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/tools/split_gguf.py $G --cuts)
echo "== session"
bash $B session "What is the capital of France?"
echo "== parts BZ_CUTS=$CUTS"
podman run --rm --cpuset-cpus=15 -e BZ_CUTS="$CUTS" -v /srv/aither-models/bonsai2-27b:/m:ro -v /root/zealos-ai/bonsai/out:/o:ro \
  localhost/zealos-bonsai-build:1 /o/harness /m/$(basename $G) chat "What is the capital of France?" 8 \
  248045,846,198,3710,369,279,6511,314,9338,30,248046,198,248045,74455,198,248068,271,248069,271 | grep -E "part|rc=|PARITY|greedy|tok/s"
