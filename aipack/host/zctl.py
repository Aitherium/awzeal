#!/usr/bin/env python3
"""zctl.py -- drive a ZealOS QEMU guest: boot, type, screenshot, wait on COM1.

usage: zctl.py <iso> [action ...]
actions (run in order):
  wait:<sec>                  sleep
  type:<text>                 sendkey each char (\\n = Enter)
  shot:<name>                 screendump -> evidence/<name>.png
  serial:<marker>:<timeout>   wait until COM1 log contains marker (fails the run on timeout)
Env: ZRAM (default 4G), ZSMP (default 4), ZDISK (default /root/zealos-ai/aidisk.img),
     ZEVID (evidence dir), ZQEMU_EXTRA.
The guest is always stopped at the end; exit code 0 only if every serial wait matched.
"""
import os
import socket
import subprocess
import sys
import time

W = "/root/zealos-ai"
RUN = f"{W}/run"
EVID = os.environ.get("ZEVID", "/mnt/c/AitherOS-Worktrees/zealos-ai/aipack/evidence")
KEYS = {" ": "spc", "\n": "ret", ";": "semicolon", ":": "shift-semicolon", '"': "shift-apostrophe",
        "'": "apostrophe", "/": "slash", ".": "dot", ",": "comma", "(": "shift-9", ")": "shift-0",
        "_": "shift-minus", "-": "minus", "=": "equal", "+": "shift-equal", "?": "shift-slash",
        "!": "shift-1", "*": "shift-8", "&": "shift-7", "<": "shift-comma", ">": "shift-dot",
        "[": "bracket_left", "]": "bracket_right", "{": "shift-bracket_left", "}": "shift-bracket_right",
        "\\": "backslash", "#": "shift-3", "%": "shift-5", "$": "shift-4"}


def key(c: str) -> str:
    if c in KEYS:
        return KEYS[c]
    if c.isalpha():
        return f"shift-{c.lower()}" if c.isupper() else c
    if c.isdigit():
        return c
    raise ValueError(f"no key for {c!r}")


class Mon:
    def __init__(self, path: str):
        for _ in range(100):
            try:
                self.s = socket.socket(socket.AF_UNIX)
                self.s.connect(path)
                break
            except OSError:
                time.sleep(0.1)
        self.s.settimeout(5)
        self._drain()

    def _drain(self):
        try:
            time.sleep(0.2)
            self.s.recv(65536)
        except OSError:
            pass

    def cmd(self, c: str) -> None:
        self.s.sendall((c + "\n").encode())
        self._drain()


def ppm_to_png(src: str, dst: str) -> None:
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "ppm2png.py"), src, dst], check=True)


def main(argv):
    iso = argv[0]
    acts = []
    for a in argv[1:]:
        if a.startswith("@"):
            acts += [ln.rstrip("\r\n") for ln in open(a[1:]) if ln.strip() and not ln.startswith("#")]
        else:
            acts.append(a)
    os.makedirs(RUN, exist_ok=True)
    os.makedirs(EVID, exist_ok=True)
    mon_path, serial = f"{RUN}/mon.sock", f"{RUN}/serial.log"
    for p in (mon_path, serial):
        if os.path.exists(p):
            os.remove(p)
    disk = os.environ.get("ZDISK", f"{W}/aidisk.img")
    cmd = ["qemu-system-x86_64", "-machine", os.environ.get("ZMACHINE", "pc"), "-enable-kvm", "-cpu", "host",
           "-m", os.environ.get("ZRAM", "4G"), "-smp", os.environ.get("ZSMP", "4"),
           "-device", "ahci,id=ahci0",
           "-drive", f"id=cd,file={iso},media=cdrom,if=none", "-device", "ide-cd,drive=cd,bus=ahci0.0",
           "-boot", "d", "-display", "vnc=127.0.0.1:41", "-vga", "std",
           "-serial", f"file:{serial}", "-monitor", f"unix:{mon_path},server,nowait",
           "-device", "isa-debug-exit,iobase=0x501,iosize=1", "-no-reboot"]
    if disk and os.path.exists(disk):
        cmd += ["-drive", f"id=ai,file={disk},format=raw,if=none,cache=writethrough",
                "-device", "ide-hd,drive=ai,bus=ahci0.1"]
    cmd += os.environ.get("ZQEMU_EXTRA", "").split()
    q = subprocess.Popen(cmd)
    ok = True
    try:
        mon = Mon(mon_path)
        for a in acts:
            verb, _, arg = a.partition(":")
            if q.poll() is not None:
                print(f"qemu exited ({q.returncode}) before {a}")
                ok = False
                break
            if verb == "mon":
                mon.s.sendall((arg + "\n").encode())
                time.sleep(0.5)
                try:
                    print("mon>", arg, "=>", mon.s.recv(65536).decode("utf-8", "replace")[-400:])
                except OSError as e:
                    print("mon>", arg, "=> (no reply)", e)
            elif verb == "key":
                mon.cmd(f"sendkey {arg}")
            elif verb == "wait":
                time.sleep(float(arg))
            elif verb == "type":
                for c in arg.replace("\\n", "\n"):
                    mon.cmd(f"sendkey {key(c)}")
                    time.sleep(0.03)
            elif verb == "shot":
                ppm = f"{RUN}/{arg}.ppm"
                mon.cmd(f'screendump "{ppm}"')
                time.sleep(1.0)
                ppm_to_png(ppm, f"{EVID}/{arg}.png")
                print(f"shot {EVID}/{arg}.png")
            elif verb == "serial":
                marker, _, tmo = arg.rpartition(":")
                deadline = time.time() + float(tmo)
                hit = False
                while time.time() < deadline and q.poll() is None:
                    if os.path.exists(serial) and marker.encode() in open(serial, "rb").read():
                        hit = True
                        break
                    time.sleep(1)
                print(f"serial {marker!r}: {'HIT' if hit else 'TIMEOUT'}")
                ok = ok and hit
                if not hit:
                    if q.poll() is None:
                        ppm = f"{RUN}/fail.ppm"
                        mon.cmd(f'screendump "{ppm}"')
                        time.sleep(1.0)
                        ppm_to_png(ppm, f"{EVID}/fail.png")
                        print(f"shot {EVID}/fail.png")
                    break
            else:
                raise SystemExit(f"unknown action {a}")
    finally:
        if q.poll() is None:
            q.terminate()
            try:
                q.wait(10)
            except subprocess.TimeoutExpired:
                q.kill()
    data = open(serial, "rb").read() if os.path.exists(serial) else b""
    print(f"--- COM1 ({len(data)} bytes)")
    print(data[-4000:].decode("utf-8", "replace"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
