#!/usr/bin/env python3
"""bench_all_cores.py -- the "give it every core" experiment, run from the Windows host.

  1. .wslconfig processors=16 -> --processors (backup kept), owner ruling noted inline
  2. fleet_verbs `fleet sleep --execute`: the fleet is stopped AND held, the WSL VM released,
     so the next boot comes up with the new core count and NOTHING else running
  3. bench_sweep.sh inside awnix for each model over --cores
  4. ALWAYS (finally): processors back to the original line, `fleet sleep` (VM restart applies it)
     then `fleet wake` -- the fleet and Garg come back exactly as the sleep record holds

Garg (tenant API) and every bot are dark from step 2 to the final wake. Dry run by default.
usage: bench_all_cores.py [--execute] [--processors 32] [--cores "4 8 16 24 32"] [--models qwen3 bonsai]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

REPO = r"C:\AitherOS-Fresh"
VERBS = [sys.executable, rf"{REPO}\AitherOS\dev\tools\fleet_verbs.py"]
WSLCONFIG = os.path.join(os.environ["USERPROFILE"], ".wslconfig")
SYNC = "/mnt/c/AitherOS-Worktrees/zealos-ai/aipack/host/z.sh"
SWEEP = "/root/zealos-ai/aipack/host/bench_sweep.sh"   # the synced copy: CRLF stripped
ENV = dict(os.environ, AW_PLATFORM_FIRST_OK="1", MSYS_NO_PATHCONV="1")


def run(argv, timeout, execute, what):
    print(f"-> {what}: {' '.join(argv)}", flush=True)
    if not execute:
        return 0
    r = subprocess.run(argv, timeout=timeout, env=ENV)
    print(f"   rc={r.returncode}", flush=True)
    return r.returncode


def set_processors(line: str, execute: bool) -> str:
    text = open(WSLCONFIG, encoding="utf-8").read()
    old = re.search(r"^processors=\d+\s*$", text, re.M)
    if not old:
        raise SystemExit("no processors= line in .wslconfig")
    print(f"-> .wslconfig {old.group(0).strip()} -> {line}", flush=True)
    if execute:
        open(WSLCONFIG, "w", encoding="utf-8").write(text[:old.start()] + line + text[old.end():])
    return old.group(0).strip()


def guest_nproc() -> int:
    r = subprocess.run(["wsl.exe", "-d", "awnix", "-u", "root", "--", "nproc"], capture_output=True,
                       text=True, timeout=300, env=ENV)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--processors", type=int, default=32)
    ap.add_argument("--cores", default="4 8 16 24 32")
    ap.add_argument("--models", nargs="+", default=["qwen3", "bonsai"])
    a = ap.parse_args()
    if a.execute:
        shutil.copy(WSLCONFIG, WSLCONFIG + ".bak-bench-" + time.strftime("%Y%m%d%H%M"))
    original = set_processors(
        f"# 2026-11-26 awzeal all-cores benchmark (temporary; restored by bench_all_cores.py)\n"
        f"processors={a.processors}", a.execute)
    try:
        # The new core count applies only on a real VM restart. `fleet sleep` skips `wsl --shutdown`
        # when anything re-boots the distro between its terminate and its check (measured
        # 2026-11-26: a 30 s wsl polling monitor did, and the sweep ran on 16 vCPUs). So the
        # guest's nproc is the proof, retried, and a wrong count never benchmarks.
        for attempt in range(3):
            run(VERBS + ["fleet", "sleep", "--execute", "--json"], 1800, a.execute,
                f"park the fleet, release the VM (attempt {attempt + 1})")
            if not a.execute:
                break
            got = guest_nproc()
            print(f"   guest nproc={got} (want {a.processors})", flush=True)
            if got == a.processors:
                break
        else:
            print("VM never came back with the new core count: no sweep", flush=True)
            return 1
        run(["wsl.exe", "-d", "awnix", "-u", "root", "--", "sh", "-c", "nproc; free -g | head -2; podman ps -q | wc -l"],
            300, a.execute, "boot awnix (held: fleet stays down)")
        run(["wsl.exe", "-d", "awnix", "-u", "root", "--", "bash", SYNC, "sync"], 300, a.execute,
            "sync the aipack into awnix")
        cores = ",".join(a.cores.split())                 # wsl.exe re-splits spaces: pass one word
        for m in a.models:
            run(["wsl.exe", "-d", "awnix", "-u", "root", "--", "bash", SWEEP, m, cores], 6 * 3600, a.execute,
                f"sweep {m} over cores [{a.cores}]")
    finally:
        text = open(WSLCONFIG, encoding="utf-8").read() if a.execute else ""
        if a.execute:
            text = re.sub(r"# 2026-11-26 awzeal all-cores benchmark.*\nprocessors=\d+", original, text)
            open(WSLCONFIG, "w", encoding="utf-8").write(text)
        print(f"-> .wslconfig restored to {original}", flush=True)
        run(VERBS + ["fleet", "sleep", "--execute", "--json"], 1800, a.execute, "VM restart to apply the restore")
        run(VERBS + ["fleet", "wake", "--execute", "--json"], 2400, a.execute, "wake the fleet (Garg back)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
