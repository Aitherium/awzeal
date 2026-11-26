#!/bin/bash
# kbdtest.sh -- does QEMU's sendkey reach a guest i8042 at all? Uses the llm.asm OS
# (polls port 0x60 directly, echoes the line to COM1).
W=/root/zealos-ai; R=$W/run; mkdir -p $R; rm -f $R/k.sock $R/k.log
qemu-system-x86_64 -machine "${ZMACHINE:-pc}" -enable-kvm -cpu host -m 2G -smp 2 \
  -kernel $W/llm.asm/build/asm-llm-os.elf -initrd $W/models/qwen3-0.6b-q8_0.asmbin \
  -display vnc=127.0.0.1:42 -vga std -serial file:$R/k.log \
  -monitor unix:$R/k.sock,server,nowait -no-reboot &
P=$!
sleep 6
for k in slash h e l p ret; do echo "sendkey $k" | socat - unix-connect:$R/k.sock >/dev/null; sleep 0.2; done
sleep 3
kill $P; wait $P 2>/dev/null
echo "--- COM1"; tail -c 1200 $R/k.log
