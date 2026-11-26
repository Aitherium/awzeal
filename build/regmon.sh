#!/bin/bash
# Watch the b51 leg-1 guest health: every 5 min take 3 register samples
# 10s apart; if all 3 show the SAME RIP+RCX, the compile has wedged
# (the healthy churn = RIP alternating the memcpy loop and the compiler
# code, RCX varying). Emit one line and exit. Also emit when the qemu
# dies (leg-1 done). The partition is watched by flmon.sh.
STREAK=0
while true; do
  OUT=$(wsl -d Debian -u root -e bash -lc 'python3 - << "EOF"
import socket, time
def qcmd(cmd, timeout=15):
    try:
        s = socket.create_connection(("127.0.0.1", 7101), timeout=10)
    except OSError:
        return "DEAD"
    s.settimeout(timeout)
    buf = b""
    while b"(qemu)" not in buf:
        try:
            buf += s.recv(65536)
        except socket.timeout:
            s.close()
            return "DEAD"
    s.sendall(cmd.encode() + b"\n")
    buf = b""
    while b"(qemu)" not in buf:
        try:
            buf += s.recv(65536)
        except socket.timeout:
            s.close()
            return "DEAD"
    s.close()
    return buf.decode(errors="replace")
def regs():
    out = qcmd("info registers")
    r = {}
    for l in out.splitlines():
        for tok in l.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                if k in ("RCX", "RIP"):
                    r[k] = v
    return r
samples = []
for i in range(3):
    samples.append(regs())
    if i < 2:
        time.sleep(10)
same = all(s.get("RIP") == samples[0].get("RIP") and s.get("RCX") == samples[0].get("RCX") for s in samples)
if "DEAD" in samples or samples[0].get("RIP") == "?":
    print("DEAD")
elif same:
    print("SUSPECT-FREEZE: %s %s x3" % (samples[0].get("RIP"), samples[0].get("RCX")))
else:
    print("OK %s %s" % (samples[0].get("RIP"), samples[0].get("RCX")))
EOF' 2>/dev/null)
  case "$OUT" in
    DEAD*)
      if [ "$STREAK" = "DEAD" ]; then
        echo "QEMU-DEAD twice — leg-1 qemu exited"
        exit 0
      fi
      STREAK="DEAD"
      ;;
    SUSPECT*)
      echo "$OUT"
      exit 0
      ;;
    OK*)
      STREAK=""
      ;;
  esac
  sleep 240
done
