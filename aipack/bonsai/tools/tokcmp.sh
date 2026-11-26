#!/bin/bash
for f in /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/*.sh; do sed -i 's/\r$//' "$f"; done
bash /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh build 2>&1 | grep -E "PARITY|relocs|FAIL|error"
python3 -u /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/tools/tokcmp.py /root/zealos-ai/bonsai/out/harness /root/zealos-ai/models/Ternary-Bonsai-2-4B-PTQ1_0.gguf 2>&1
