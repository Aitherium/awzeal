#!/bin/bash
# bench.sh -- build, then the in-RAM single-core matvec bench (no model mmap) on CPU 15
for f in /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/*.sh; do sed -i 's/\r$//' "$f"; done
bash /mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh build 2>&1 | grep -E "PARITY|relocs|FAIL|error" 
podman run --rm --cpuset-cpus=15 -v /root/zealos-ai/bonsai/out:/o:ro localhost/zealos-bonsai-build:1 /o/kernel_test --bench
awk '/MemAvailable/{print "MemAvailable GiB", int($2/1048576)}' /proc/meminfo
