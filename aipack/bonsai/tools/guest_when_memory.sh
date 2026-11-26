#!/bin/bash
# guest_when_memory.sh -- refresh the Bonsai disk's code files, then boot the 12 GB ZealOS
# guest only once awnix has the memory for it (never squeeze the fleet into OOM).
#   NEED_GB (default 11)  MemAvailable required before boot
#   MAX_WAIT_MIN (default 360)
set -uo pipefail
W=/root/zealos-ai
NEED_GB=${NEED_GB:-11}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-360}
LOG=$W/bonsai-mc.log
echo "$(date -Is) refresh disk files" > "$LOG"
OFF=$((2048 * 512))
podman run --rm -v "$W:/w" localhost/zealos-bonsai-build:1 bash -c "
  set -e
  command -v mcopy >/dev/null || { apt-get update -qq >/dev/null && apt-get install -y -qq mtools >/dev/null 2>&1; }
  export MTOOLS_SKIP_CHECK=1
  for f in /w/bonsai/out/ENGINE2.BIN /w/bonsai/aipack/bonsai/zealos/AIBONSAI.ZC /w/bonsai/aipack/zealos/AILLM.ZC; do
    mcopy -o -i /w/bonsai/bzdisk.img@@$OFF \"\$f\" ::/
  done
  mdir -i /w/bonsai/bzdisk.img@@$OFF ::/ | grep -E 'ENGINE2|AIBONSAI|AILLM'
" >> "$LOG" 2>&1 || { echo "disk refresh failed" >> "$LOG"; exit 1; }
for i in $(seq 1 "$MAX_WAIT_MIN"); do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -ge "$NEED_GB" ]; then
    echo "$(date -Is) MemAvailable ${avail} GiB >= ${NEED_GB}: booting" >> "$LOG"
    cd "$W" && ZRAM=12G ZSMP=4 ZDISK=$W/bonsai/bzdisk.img \
      python3 -u aipack/host/zctl.py zealos-ai-live.iso @aipack/steps/bonsai.txt >> "$LOG" 2>&1
    echo "$(date -Is) guest run exit $?" >> "$LOG"
    exit 0
  fi
  [ $((i % 10)) -eq 1 ] && echo "$(date -Is) waiting: MemAvailable ${avail} GiB < ${NEED_GB}" >> "$LOG"
  sleep 60
done
echo "$(date -Is) gave up after ${MAX_WAIT_MIN} min: memory never freed" >> "$LOG"
exit 2
