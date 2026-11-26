#!/bin/bash
# run4b.sh -- Ternary-Bonsai-2-4B (PTQ1_0) parity vs the live llama-server, 1 and 4 cores.
# The 4B lives on D: (/mnt/d, 9p drvfs): mmap page faults over 9p dominate there, so the
# harness reads an ext4 copy under /root/zealos-ai/models (made once, ~0.95 GB).
SRC=/mnt/d/models/bonsai2-4b/Ternary-Bonsai-2-4B-PTQ1_0.gguf
G=/root/zealos-ai/models/Ternary-Bonsai-2-4B-PTQ1_0.gguf
mkdir -p /root/zealos-ai/models
[ -s "$G" ] && [ "$(stat -c %s "$G")" = "$(stat -c %s "$SRC")" ] || cp "$SRC" "$G"
REF=248045,846,198,3710,369,279,6511,314,9338,30,248046,198,248045,74455,198,248068,271,248069,271
cat $G > /dev/null
export BZ_GGUF=$G
awk '/MemAvailable/{print "MemAvailable GiB", int($2/1048576)}' /proc/meminfo
SPECS=${*:-"15:1 12-15:4"}
for spec in $SPECS; do
  cpus=${spec%%:*}; n=${spec##*:}
  echo "== cores $n (cpuset $cpus)"
  BZ_CPU=$cpus BZ_THREADS=$n bash /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh chat "What is the capital of France?" 8 $REF 2>&1 | grep -E "rc=|threads=|PARITY|prefill|greedy|tok/s"
done
echo "want greedy 760,6511,314,9338,369,2972,57590,159034"
