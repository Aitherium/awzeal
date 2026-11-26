#!/usr/bin/env python3
"""Poll the qemu telnet monitor's `info registers`; log EIP/RIP stability.
Usage: monitor_eip_poll.py <port> <outfile> [duration_s] [interval_s]
Detects a fixed EIP for >= 4 min and flags it as a freeze-like state.
"""
import socket
import sys
import time

PORT = int(sys.argv[1])
OUT = sys.argv[2]
DURATION = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
INTERVAL = int(sys.argv[4]) if len(sys.argv) > 4 else 10

s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
s.settimeout(2)
buf = b""


def drain():
    global buf
    try:
        while True:
            d = s.recv(8192)
            if not d:
                break
            buf += d
    except socket.timeout:
        pass
    return buf


drain()  # greeting
log = open(OUT, "w")
prev = None
stable = 0
start = time.time()
while time.time() - start < DURATION:
    s.sendall(b"info registers\n")
    time.sleep(0.4)
    resp = drain()
    text = resp.decode("latin-1", "replace")
    eip = None
    for line in text.splitlines():
        ls = line.strip()
        if ls.startswith(("EIP=", "RIP=", "CS=", "FLAGS=")):
            eip = ls
            break
    if eip is None:
        eip = "?(%db)" % len(resp)
    now = int(time.time() - start)
    log.write("[%4ds] %s\n" % (now, eip))
    log.flush()
    if eip == prev and "?" not in eip:
        stable += 1
    else:
        stable = 0
    prev = eip
    print("[%4ds] %s stable=%d" % (now, eip, stable), flush=True)
    if stable == 24:
        print("FREEZE-LIKE: EIP fixed for 4 min — screendump", flush=True)
        s.sendall(b"screendump /tmp/freeze-screen2.ppm\n")
        time.sleep(1)
        drain()
    time.sleep(INTERVAL)
log.close()
print("poller done")
