#!/bin/bash
# bz.sh <verb> [args] -- run inside awnix: sync aipack, build, and drive the parity harness.
#   build                      out/harness + out/ENGINE2.BIN
#   ref [question]             reference ids/greedy tokens from the live server (read-only)
#   tok "<text>" | chat "<q>" [n] [ref_ids] | session "<q>"   harness runs (one pinned core)
#   split <out dir> [part_mb]  split the GGUF at tensor boundaries for FAT32
set -euo pipefail
SRC=${ZPACK_SRC:-/mnt/c/AitherOS-Worktrees/zealos-ai/aipack}
W=${ZWORK:-/root/zealos-ai/bonsai}
G=${BZ_GGUF:-/srv/aither-models/bonsai2-27b/Ternary-Bonsai-2.5-27B-PQ2_0.gguf}
CPU=${BZ_CPU:-15}
mkdir -p "$W"
rm -rf "$W/aipack" && cp -r "$SRC" "$W/aipack"
find "$W/aipack" -name '*.sh' -exec sed -i 's/\r$//' {} +
verb=${1:-build}; shift || true
run() {   # the harness runs in the build image with the model dir mounted read-only, one core
  podman run --rm --cpuset-cpus=$CPU -e BZ_THREADS="${BZ_THREADS:-1}" -e BZ_CTX="${BZ_CTX:-512}" -e BZ_ARENA_MB="${BZ_ARENA_MB:-448}" \
    -v "$(dirname "$G"):/m:ro" -v "$W/out:/o:ro" localhost/zealos-bonsai-build:1 /o/harness "/m/$(basename "$G")" "$@"
}
case $verb in
  build) bash "$W/aipack/bonsai/build.sh" "$W/aipack" "$W/out" ;;
  ref) python3 "$W/aipack/bonsai/tools/ref_greedy.py" https://127.0.0.1:8092 "$W/ref.json" "$@" ;;
  tok|chat|session) free -m | head -2; run "$verb" "$@" ;;
  split) python3 "$W/aipack/bonsai/tools/split_gguf.py" "$G" "$@" ;;
  *) echo "unknown verb $verb"; exit 2 ;;
esac
