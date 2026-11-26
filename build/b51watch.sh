#!/bin/bash
# systemd-owned watcher for the b51 build: polls the build unit's state
# + the vdisk partition every 2 min, writes a timestamped line to the
# watch file; when the build unit exits, records the final state (unit
# status + log tail). Survives the session (runs as a systemd transient
# unit), so the session just reads the watch file.
WATCH=/mnt/c/AitherOS-Fresh/ZealOS-AI/build/b51-watch.txt
LOG=/mnt/c/AitherOS-Fresh/ZealOS-AI/build/b51-build.log
BASE=/mnt/c/AitherOS-Fresh/ZealOS-AI/build/zealos-base.raw
# each build run's temp dir is fresh — discover the newest vdisk. The
# build unit's first ~10s are the zealbooter make; the format that creates
# the vdisk comes after, so retry (a start-race once left the run unwatched).
VDISK=""
for _ in $(seq 1 20); do
  VDISK=$(ls -t /tmp/tmp.*/ZealOS.raw 2>/dev/null | head -1)
  [ -n "$VDISK" ] && break
  sleep 3
done
if [ -z "$VDISK" ]; then
  echo "watch: no vdisk found after retries" > "$WATCH"
  exit 1
fi
echo "watch started $(date +%H:%M:%S)" > "$WATCH"
while systemctl is-active --quiet zealos-b51; do
  P=$(fdisk -l "$VDISK" 2>/dev/null | grep -cE "^$VDISK[0-9]")
  # The leg-1's stall signature (measured 2026-08-29 22:33: the vdisk
  # mtime sat at CREATION for 2h50m while the guest churned the compiler
  # with a static screen — no partition, no install, no disk write). The
  # FL census cannot see it (the leg-1 emits no FL records) and the
  # register churn looks alive. mtime staleness is the detector.
  MT=$(stat -c %Y "$VDISK" 2>/dev/null || echo 0)
  NOW=$(date +%s)
  AGE=$(( (NOW - MT) / 60 ))
  echo "$(date +%H:%M:%S) active partitions=$P vdisk_mtime_age_min=$AGE" >> "$WATCH"
  # At the leg-2's start the vdisk holds the installed system + the staged
  # current-src tree (pre-chain) — the ideal --from-vdisk base. Copy once.
  if [ ! -f "$BASE" ] && grep -q "Rebuilding kernel headers" "$LOG" 2>/dev/null; then
    cp "$VDISK" "$BASE" 2>/dev/null && echo "$(date +%H:%M:%S) BASE PRESERVED -> $BASE" >> "$WATCH"
  fi
  sleep 120
done
echo "=== $(date +%H:%M:%S) BUILD UNIT EXITED ===" >> "$WATCH"
systemctl status zealos-b51 --no-pager -n 4 >> "$WATCH" 2>&1
echo "--- build log tail ---" >> "$WATCH"
tail -8 /mnt/c/AitherOS-Fresh/ZealOS-AI/build/b51-build.log >> "$WATCH" 2>&1
echo "watch done $(date +%H:%M:%S)" >> "$WATCH"
