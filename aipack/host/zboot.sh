#!/bin/bash
# zboot.sh <iso> <seconds> [extra qemu args...]
# Boot a ZealOS ISO under KVM with a VNC display (so the QEMU input layer is
# wired and `sendkey` reaches the guest's i8042), the HMP monitor on a unix
# socket, COM1 to a log file, and the AI data disk (if present) on AHCI port 1.
# Leaves qemu running for <seconds>, then dumps the screen and stops it.
set -uo pipefail
W=/root/zealos-ai
ISO=$1; SECS=$2; shift 2
RUN=$W/run; mkdir -p "$RUN"
rm -f "$RUN/mon.sock" "$RUN/serial.log"
DISK_ARGS=()
if [ -f "$W/aidisk.img" ]; then
  DISK_ARGS=(-drive "id=ai,file=$W/aidisk.img,format=raw,if=none,cache=writethrough" -device ide-hd,drive=ai,bus=ahci0.1)
fi
qemu-system-x86_64 -machine q35 -enable-kvm -cpu host -m "${ZRAM:-4G}" -smp "${ZSMP:-4}" \
  -device ahci,id=ahci0 \
  -drive "id=cd,file=$ISO,media=cdrom,if=none" -device ide-cd,drive=cd,bus=ahci0.0 \
  "${DISK_ARGS[@]}" \
  -boot d -display vnc=127.0.0.1:41 -vga std \
  -serial "file:$RUN/serial.log" \
  -monitor "unix:$RUN/mon.sock,server,nowait" \
  -device isa-debug-exit,iobase=0x501,iosize=1 \
  -no-reboot "$@" &
QPID=$!
echo "$QPID" > "$RUN/qemu.pid"
for i in $(seq 1 "$SECS"); do
  kill -0 "$QPID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$QPID" 2>/dev/null; then
  echo "screendump $RUN/screen.ppm" | socat - "unix-connect:$RUN/mon.sock" >/dev/null 2>&1
  sleep 1
fi
[ "${ZKEEP:-0}" = 1 ] || { kill "$QPID" 2>/dev/null; wait "$QPID" 2>/dev/null; }
echo "--- serial ($(wc -c < "$RUN/serial.log") bytes)"; tail -c 3000 "$RUN/serial.log"
