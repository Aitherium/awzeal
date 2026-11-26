#!/bin/bash
# bg.sh <log> <bz.sh args...> -- run a bz.sh verb with all output in <log> (wsl.exe re-splits ';')
LOG=$1; shift
mkdir -p "$(dirname "$LOG")"
sed -i 's/\r$//' /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh
bash /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh "$@" > "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
tail -60 "$LOG"
