#!/bin/bash
# dump.sh -- GGUF header of the Bonsai 2.5 model into tools/meta.txt (reads the header only)
G=/srv/aither-models/bonsai2-27b/Ternary-Bonsai-2.5-27B-PQ2_0.gguf
D=/mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/tools
mkdir -p /root/zealos-ai/bonsai
python3 $D/gguf_dump.py $G > /root/zealos-ai/bonsai/meta.txt
cp /root/zealos-ai/bonsai/meta.txt $D/meta.txt
ls -la /srv/aither-models/bonsai2-27b/
podman ps --format '{{.Names}} {{.Status}}' | grep -i bonsai
