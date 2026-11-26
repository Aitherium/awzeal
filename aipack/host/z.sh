#!/bin/bash
# z.sh <verb> [args] -- sync the aipack from the worktree into awnix and run a step.
#   sync | engine | disk | run <zctl actions...>
set -euo pipefail
SRC=${ZPACK_SRC:-/mnt/c/AitherOS-Worktrees/zealos-ai/aipack}
W=${ZWORK:-/root/zealos-ai}
rm -rf "$W/aipack" && cp -r "$SRC" "$W/aipack"
find "$W/aipack" -name '*.sh' -exec sed -i 's/\r$//' {} +
verb=${1:-sync}; shift || true
case $verb in
  sync) ;;
  engine) bash "$W/aipack/engine/build.sh" "$W/llm.asm" "$W/aipack" "$W/out" 2>&1 | grep -v debconf ;;
  iso) python3 "$W/aipack/host/patch_once.py" "$W/zealos-stock.iso" "$W/zealos-ai-live.iso" "$W/aipack/zealos/ONCE-AUTOSTART.ZC" ;;
  disk) bash "$W/aipack/host/mkdisk.sh" 2>&1 | grep -v debconf ;;
  run) cd "$W" && python3 "$W/aipack/host/zctl.py" "${ZISO:-$W/zealos-ai-live.iso}" "$@" ;;
  *) echo "unknown verb $verb"; exit 2 ;;
esac
