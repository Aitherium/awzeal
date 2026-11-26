#!/usr/bin/env python3
"""awzeal-run.py -- container entrypoint: boot awzeal under QEMU and wire COM1 to the terminal.

Env:
  AWZEAL_RAM     guest RAM (default 4G)
  AWZEAL_SMP     guest cores, 0 = min(host cores, 8)
  AWZEAL_PROMPT  one-shot: ask this once the chat is up, print the answer, exit 0
  AWZEAL_HOME    where the image keeps the ISO + AI pack (default /opt/awzeal)
Without /dev/kvm it falls back to TCG (works, but slow).
"""
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time

HOME = os.environ.get("AWZEAL_HOME", "/opt/awzeal")
RUN = "/tmp/awzeal"
PACK_FILES = ["ENGINE.BIN", "QWEN3.BIN", "AILLM.ZC", "AIKERNEL.ZC", "AIINFER.ZC", "AITOOLS.ZC", "RUN.ZC"]


def log(msg: str) -> None:
    sys.stderr.write(f"[awzeal] {msg}\n")
    sys.stderr.flush()


def make_disk(img: str) -> None:
    """MBR + one FAT32 partition carrying the AI pack (ZealOS mounts it at boot)."""
    size = sum(os.path.getsize(f"{HOME}/{f}") for f in PACK_FILES) + (256 << 20)
    with open(img, "wb") as f:
        f.truncate(size)
    subprocess.run(["sfdisk", "-q", img], input=b"2048,,c,*\n", check=True)
    sectors = size // 512 - 2048
    subprocess.run(["mkfs.fat", "-F", "32", "-n", "AWZEAL", "--offset", "2048", img, str(sectors // 2)],
                   check=True, stdout=subprocess.DEVNULL)
    env = dict(os.environ, MTOOLS_SKIP_CHECK="1")
    for f in PACK_FILES:
        subprocess.run(["mcopy", "-i", f"{img}@@{2048 * 512}", f"{HOME}/{f}", "::/"], check=True, env=env)


def connect(path: str) -> socket.socket:
    for _ in range(200):
        try:
            s = socket.socket(socket.AF_UNIX)
            s.connect(path)
            return s
        except OSError:
            time.sleep(0.05)
    raise SystemExit(f"cannot connect {path}")


def main() -> int:
    os.makedirs(RUN, exist_ok=True)
    disk = f"{RUN}/aidisk.img"
    log("building the AI pack disk")
    make_disk(disk)
    kvm = os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.R_OK | os.W_OK)
    smp = int(os.environ.get("AWZEAL_SMP", "0")) or min(os.cpu_count() or 1, 8)
    accel = ["-enable-kvm", "-cpu", "host"] if kvm else ["-cpu", "max"]
    if not kvm:
        log("no /dev/kvm: falling back to TCG, expect minutes per answer (run with --device /dev/kvm)")
    com, mon = f"{RUN}/com1.sock", f"{RUN}/mon.sock"
    cmd = ["qemu-system-x86_64", "-machine", "pc", *accel,
           "-m", os.environ.get("AWZEAL_RAM", "4G"), "-smp", str(smp),
           "-device", "ahci,id=ahci0",
           "-drive", f"id=cd,file={HOME}/awzeal-live.iso,media=cdrom,if=none,readonly=on",
           "-device", "ide-cd,drive=cd,bus=ahci0.0",
           "-drive", f"id=ai,file={disk},format=raw,if=none",
           "-device", "ide-hd,drive=ai,bus=ahci0.1",
           "-boot", "d", "-vga", "std", "-display", "vnc=127.0.0.1:0",
           "-chardev", f"socket,id=s0,path={com},server=on,wait=off", "-serial", "chardev:s0",
           "-monitor", f"unix:{mon},server,nowait", "-no-reboot"]
    log(f"booting ZealOS ({'KVM' if kvm else 'TCG'}, {smp} cores, {os.environ.get('AWZEAL_RAM', '4G')})")
    q = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
    signal.signal(signal.SIGTERM, lambda *_: q.terminate())
    serial = connect(com)
    monitor = connect(mon)
    seen = bytearray()
    done = threading.Event()

    def pump():
        while not done.is_set():
            r, _, _ = select.select([serial], [], [], 0.5)
            if r:
                data = serial.recv(4096)
                if not data:
                    break
                seen.extend(data)
                sys.stdout.buffer.write(data.replace(b"ZLOUT-BEGIN\r\n", b"").replace(b"ZLOUT-BEGIN\n", b""))
                sys.stdout.flush()
        done.set()

    threading.Thread(target=pump, daemon=True).start()
    # the boot splash waits for a key before the autostart runs
    t0 = time.time()
    while b"ZAI autostart" not in seen and time.time() - t0 < 300 and q.poll() is None:
        time.sleep(8)
        monitor.sendall(b"sendkey spc\n")
    prompt = os.environ.get("AWZEAL_PROMPT")
    while b"awzeal>" not in seen and q.poll() is None and not done.is_set():
        time.sleep(0.5)
    if prompt:
        mark = len(seen)
        serial.sendall(prompt.encode() + b"\r")
        while b"ZLOUT-END" not in seen[mark:] and q.poll() is None:
            time.sleep(0.2)
        q.terminate()
        q.wait(10)
        return 0 if b"ZLOUT-END" in seen[mark:] else 1
    try:
        while q.poll() is None:
            line = sys.stdin.readline()
            if not line:
                q.wait()
                break
            serial.sendall(line.rstrip("\n").encode() + b"\r")
    except KeyboardInterrupt:
        pass
    finally:
        if q.poll() is None:
            q.terminate()
            q.wait(10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
