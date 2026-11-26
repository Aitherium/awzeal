#!/bin/bash
# reftok.sh [gguf name] -- tokenizer parity against the PrismML fork itself (llama-server prism-b12128,
# CPU, one pinned core, throwaway container on a loopback port, removed afterwards). The GGUF must sit
# in /root/zealos-ai/models (run4b.sh / q1proof.sh make the copies). Default: the qwen35 4B; a qwen3
# Q1_0 file (Bonsai-1.7B-Q1_0.gguf, Bonsai-4B-Q1_0.gguf) checks the qwen2 pre-tokenizer.
GG=${1:-Ternary-Bonsai-2-4B-PTQ1_0.gguf}
for f in /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/*.sh; do sed -i 's/\r$//' "$f"; done
bash /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh build 2>&1 | grep -E "PARITY|relocs|FAIL|error"
N=bz-reftok-$$
podman run -d --rm --pull=never --name $N --cpuset-cpus=15 -p 127.0.0.1:18297:8080 -v /root/zealos-ai/models:/m:ro \
  ghcr.io/aitherium/llamacpp-prism:prism-b12128 -m /m/$GG -ngl 0 -t 1 -c 256 -np 1 \
  --host 0.0.0.0 --port 8080 --no-webui >/dev/null
for i in $(seq 1 90); do curl -s --max-time 2 http://127.0.0.1:18297/health | grep -q ok && break; sleep 2; done
curl -s --max-time 2 http://127.0.0.1:18297/health; echo
BZ_REF_URL=http://127.0.0.1:18297 python3 -u /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/tools/tokcmp.py /root/zealos-ai/bonsai/out/harness /root/zealos-ai/models/$GG 2>&1
rc=$?
podman stop -t 2 $N >/dev/null 2>&1
echo "tokcmp rc=$rc ($GG)"
