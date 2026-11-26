#!/bin/bash
# b.sh <log> <bz.sh args...> -- like bg.sh, env passthrough (BZ_GGUF, BZ_CPU, BZ_THREADS, BZ_THINK)
LOG=$1; shift
mkdir -p "$(dirname "$LOG")"
for f in /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/*.sh; do sed -i 's/\r$//' "$f"; done
bash /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh "$@" > "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
tail -60 "$LOG"
