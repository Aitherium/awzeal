#!/bin/bash
# Watch the b51 leg-1: emit one line when the FL count moves off 6927,
# the partition appears, or the qemu dies (leg-1 done -> build moved on).
VDISK="/tmp/tmp.CZfY2P5zZT/ZealOS.raw"
LAST=""
while true; do
  OUT=$(wsl -d Debian -u root -e bash -lc 'python3 - << "EOF"
import socket, os
def qcmd(cmd, timeout=300):
    s = socket.create_connection(("127.0.0.1", 7101), timeout=10)
    s.settimeout(timeout)
    buf = b""
    while b"(qemu)" not in buf:
        buf += s.recv(65536)
    s.sendall(cmd.encode() + b"\n")
    buf = b""
    while b"(qemu)" not in buf:
        buf += s.recv(65536)
    s.close()
    return buf.decode(errors="replace")
try:
    qcmd("pmemsave 1073741824 1073741824 \"/tmp/flmon.bin\"", 300)
    data = open("/tmp/flmon.bin", "rb").read()
    print(data.count(b"FL:"))
except Exception as e:
    print("DEAD")
EOF
rm -f /tmp/flmon.bin' 2>/dev/null)
  P=$(wsl -d Debian -u root -e bash -lc "fdisk -l $VDISK 2>/dev/null | grep -cE '^$VDISK[0-9]'")
  if [ "$P" != "0" ] && [ -n "$P" ]; then
    echo "PARTITION appeared on vdisk"
    exit 0
  fi
  if [ "$OUT" = "DEAD" ]; then
    if [ "$LAST" = "DEAD" ]; then
      echo "QEMU-DEAD (two consecutive) — leg-1 qemu exited"
      exit 0
    fi
    LAST="DEAD"
  else
    LAST=""
  fi
  # The raw "FL:" count is phase-dependent (boot = 0, compile = source
  # markers) — NOT a progress signal. Only the partition and qemu-exit
  # arms are meaningful; health is regmon.sh's job.
  sleep 90
done
