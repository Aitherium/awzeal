#!/bin/bash
# golden4b.sh -- the 4B against the PrismML fork's CPU golden (llama.cpp prism-b12128, -ngl 0;
# AitherVeil selftest/bonsai2-4b-golden.fork.json): think-mode prompt ids + 16 greedy tokens.
export BZ_GGUF=/root/zealos-ai/models/Ternary-Bonsai-2-4B-PTQ1_0.gguf BZ_THINK=1 BZ_CPU=12-15 BZ_THREADS=4
B=/mnt/c/AitherOS-Worktrees/zealos-ai/aipack/bonsai/bz.sh
run() { echo "== $1"; bash $B chat "$1" 16 "$2" 2>&1 | grep -E "PARITY|greedy|tok/s"; echo "   want   $3"; }
run "What is the capital of France? Answer in one short sentence." 248045,846,198,3710,369,279,6511,314,9338,30,21134,303,799,2716,11316,13,248046,198,248045,74455,198,248068,198 90700,8340,25,271,16,13,220,2972,2014,53983,279,5952,64700,198,262,348
run "Count from 1 to 5." 248045,846,198,2427,494,220,16,310,220,20,13,248046,198,248045,74455,198,248068,198 90700,8340,25,271,16,13,220,2972,2014,53983,279,5952,64700,561,1156,369
run "Name the three primary colors of light." 248045,846,198,657,279,2250,5839,7736,314,3000,13,248046,198,248045,74455,198,248068,198 90700,8340,25,271,16,13,220,2972,2014,53983,279,5952,64700,561,1156,369
