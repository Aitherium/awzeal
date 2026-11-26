#!/usr/bin/env python3
"""freeze-probe.py <monitor-port> <ram-out> [kill-pid]

Poll the leg-2 guest's RAM via the qemu monitor and detect the ZealC
debugger freeze: the compile's FL records stop GROWING when the compile
dies (the debugger loop spins CPU but writes nothing). A healthy compile
keeps appending FL:C:/path,line records.

Loop: every 45s, pmemsave (top 1GB of the 2G guest) -> count FL records. TWO kill arms:

1. STALLED COMPILE (the original arm): the count stalls at >0 for 12
   consecutive probes (9 min) with NO reboot and NO ISO stage ->
   FREEZE, kill qemu, exit 1. b49: the distro chain (AutoFullDistro0-5)
   is a staged reboot ladder — every stage's compile ends in a plateau,
   then the Once chain REBOOTS (~30s) and the next boot's compile resets
   FL. A real freeze (ZealC debugger) never recovers. The FINAL stage
   (AutoFullDistro5/MakeMyISO) is a long NON-compiling plateau — the
   ISO build writes no FL records, so the pre-b49 two-probe arm killed
   the guest mid-ISO (the b48 kill). Guards: wait 12 probes (9 min) and
   treat a RAM dump carrying the MakeMyISO code ('MyDistro.ISO.C' —
   resident only once AFD5 has been loaded) as a HEALTHY ISO stage.
2. BOOT-STAGE PANIC (b41, added after the QEMU-10 empty-CD panic): the
   count is 0 for 20 consecutive probes (15 min). A healthy chain starts
   compiling within minutes of leg-2 boot, so a guest still at FL=0 is
   sitting in the ZealC debugger after a kernel panic (measured 2026-08-28:
   AHCI Port 2 empty-tray ATAPI failure at BlkDevsInitAll — the kernel
   PANICS before StartOS ever runs, FL never rises, and the old probe
   burned the FULL 150-min budget on it). Without this arm a boot-stage
   panic is indistinguishable from a slow boot until the deadline.
   no-verdict (-1) never counts toward the streak; a confirmed >0 resets it.

Exit 0 = the probe was killed externally (build finished).

FAULT-INTOLERANT COUNTING (b34, fixed after build 32b's FALSE freeze):
the 1GB pmemsave under host load outlasts the monitor read window, so the
old code timed out, counted the PARTIAL/STALE dump anyway, got the same
count twice, and killed a HEALTHY compiling guest ~90s into leg 2 (the
'freeze at 4651' that was actually a healthy kernel/system pass). A count
is only a verdict when the dump is COMPLETE: the monitor confirmed the
pmemsave finished, the file is exactly 1GB, and it is deleted after
counting so a stale file can never be re-counted. Anything else is -1 =
no verdict (printed, never judged).
"""
import os
import re
import signal
import socket
import sys
import time

PORT = int(sys.argv[1])
RAM = sys.argv[2]
KILL_PID = int(sys.argv[3]) if len(sys.argv) > 3 else None
SCREEN = str(sys.argv[5]) if len(sys.argv) > 5 else (RAM + ".ppm")
# b48: the guest RAM total comes from the BUILD SCRIPT (argv[4],
# ZEALOS_RAM), never a hardcode here — the two drifted once already
# (b44: hardcoded 1G dump while the guest ran 2G, so the STALL arm
# killed every healthy compile at 15 min because the FL dictionary
# lives in the TOP half of RAM, ZERO in the dumped bottom half —
# measured 2026-08-28: 4650 records in 1G-2G, 0 in 0-1G). The dump
# targets the TOP 1GB of whatever the guest has.
_GUEST_G = float(str(sys.argv[4]).rstrip("Gg")) if len(sys.argv) > 4 else 2.0
RAM_TOTAL = int(_GUEST_G * 1073741824)
RAM_OFFSET = max(0, RAM_TOTAL - 1073741824)  # top 1GB (or all of a smaller guest)
RAM_SIZE = min(RAM_TOTAL, 1073741824)

def monitor(cmd: str, timeout: float = 300) -> bool:
    """Send one monitor command; return True only if the completion
    prompt came back. False = no verdict (connection refused while the
    guest boots, or the command outlasted the window)."""
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=10)
    except OSError:
        return False
    s.settimeout(timeout)
    try:
        buf = b""
        while b"(qemu)" not in buf:
            try:
                buf += s.recv(65536)
            except socket.timeout:
                return False
        s.sendall(cmd.encode() + b"\n")
        buf = b""
        while b"(qemu)" not in buf:
            try:
                buf += s.recv(65536)
            except socket.timeout:
                return False
    finally:
        s.close()
    return True

def save_screen():
    """Screendump the guest BEFORE the kill — every failed build keeps its
    error screen (measured 2026-08-28: every freeze needed a manual vdisk
    reboot + re-freeze to see the error; the evidence was lost at the kill)."""
    try:
        monitor(f'screendump "{SCREEN}"', 60)
        print(f"screen saved to {SCREEN}", flush=True)
    except Exception as e:
        print(f"screendump failed: {e}", flush=True)

def fl_count() -> tuple:
    """Count FL records from a COMPLETE pmemsave. (count, data); count -1
    = no verdict. Keeps the dump as RAM+'.last' — the plateau/ISO-stage
    scans read it, and the next pmemsave writes a fresh RAM file, so a
    stale dump can never be counted twice."""
    if not monitor(f'pmemsave {RAM_OFFSET} {RAM_SIZE} "{RAM}"'):
        return -1, None  # pmemsave did not complete (monitor busy / guest booting)
    try:
        size = os.path.getsize(RAM)
    except OSError:
        return -1, None
    if size != RAM_SIZE:
        return -1, None  # partial write — never count a half-written dump
    with open(RAM, "rb") as f:
        data = f.read()  # FL records live at 850-1000MB — read it all
    os.rename(RAM, RAM + ".last")
    return len(re.findall(rb"FL:C:[^,]+,[0-9]+", data)), data

prev = None
zero_streak = 0
plateau_streak = 0
for i in range(1, 350):
    time.sleep(45)
    cur, data = fl_count()
    if cur < 0:
        print(f"[freeze-probe {i}] FL=no-verdict (incomplete dump)", flush=True)
        continue  # no verdict is never a freeze
    print(f"[freeze-probe {i}] FL={cur}", flush=True)
    if cur == 0:
        zero_streak += 1
        if zero_streak >= 20:  # 20 x 45s = 15 min with NO compile activity
            print(f"STALL DETECTED: FL stayed 0 for {zero_streak} consecutive probes — boot-stage panic (kernel Panic before StartOS), not a slow boot", flush=True)
            save_screen()
            if KILL_PID:
                os.kill(KILL_PID, signal.SIGKILL)
                print(f"killed qemu pid {KILL_PID}", flush=True)
            sys.exit(1)
    else:
        zero_streak = 0
    if prev is not None and cur == prev and cur > 0:
        # b49: see the header — the chain's stage boundaries and the ISO
        # stage are legitimate plateaus; only a plateau with no reboot and
        # no ISO stage (9 min) is a debugger freeze.
        iso_stage = data is not None and b"MyDistro.ISO.C" in data
        if iso_stage:
            print(f"[freeze-probe {i}] ISO STAGE: FL plateau at {cur} with MakeMyISO resident — healthy, waiting for the debug-exit beacon", flush=True)
            plateau_streak = 0
        else:
            plateau_streak += 1
            print(f"[freeze-probe {i}] plateau {plateau_streak}/40 at FL={cur}", flush=True)
            if plateau_streak >= 40:  # 40 x 45s = 30 min
                # b51: raised 12->40 (30 min). Measured 2026-08-29 on a
                # LIVE healthy leg-1 compile: the FL count sat static at
                # 6927 past the old 9-min kill window in TWO consecutive
                # runs (5+ min and 6+ min at the externally-cut points,
                # the file boundary never observed) while the registers
                # showed the healthy churn (RIP alternating the memcpy
                # loop and the compiler hash code, RCX varying
                # 0x400->0x2d3f40) — the FL records jump at FILE
                # BOUNDARIES, and a big file under the loaded nested-KVM
                # box clears 9 min. The 12-probe arm also killed the b49
                # leg-2 round 1 (rc=137 at plateau 18,083) whose
                # KConfig-hang verdict is unproven against the
                # healthy-cadence alternative. 30 min still fails fast
                # on a REAL freeze (the ZealC debugger never emits), and
                # the 350-probe deadline + the round's timeout + the
                # extraction check remain the backstops.
                print(f"FREEZE DETECTED: FL records stalled at {cur} for {plateau_streak} probes (30 min: no reboot, no ISO stage — ZealC debugger)", flush=True)
                try:
                    import shutil
                    shutil.copy(RAM + ".last", RAM + ".plateau")
                    print(f"plateau RAM preserved to {RAM}.plateau", flush=True)
                except Exception as e:
                    print(f"plateau RAM copy failed: {e}", flush=True)
                save_screen()
                if KILL_PID:
                    os.kill(KILL_PID, signal.SIGKILL)
                    print(f"killed qemu pid {KILL_PID}", flush=True)
                sys.exit(1)
    else:
        plateau_streak = 0
    prev = cur
# deadline: 350 probes x 45s = 262 min cap (b49: the distro chain is 5+
# boots, each with a boot compile + the stage's CC — hours of chain)
if KILL_PID:
    print("deadline reached — killing qemu", flush=True)
    os.kill(KILL_PID, signal.SIGKILL)
sys.exit(0)
