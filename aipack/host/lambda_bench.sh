#!/bin/bash
# lambda_bench.sh -- runs ON a rented cloud box (via script_box.py). Writes everything to ./out.
#  1. probe: cores (physical vs SMT), RAM, /dev/kvm + nested virt, NVIDIA PCI ids, IOMMU groups
#     (the last two are what the awzeal GPU driver testbed needs: vfio-pci, noiommu or not)
#  2. no /dev/kvm -> stop (exit 3): QEMU without KVM is ~50x slower, the numbers would be noise
#  3. else: the public ghcr.io/aitherium/awzeal image, Qwen3-0.6B decode at each core count,
#     3 boots each, one thread per PHYSICAL core first, then every SMT thread
set -u
O=out; mkdir -p $O
PHYS=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
THREADS=$(nproc)
{
  echo "threads=$THREADS physical_cores=$PHYS"
  lscpu | grep -E 'Model name|Thread|Core|Socket|NUMA node\(s\)|Hypervisor|Virtualization'
  free -g | head -2
  ls -la /dev/kvm 2>&1
  lsmod | grep -E '^kvm' 2>&1
  for f in /sys/module/kvm_intel/parameters/nested /sys/module/kvm_amd/parameters/nested; do
    [ -f $f ] && echo "$f=$(cat $f)"; done
  echo "--- nvidia pci"; lspci -nn | grep -i nvidia
  echo "--- iommu groups: $(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)"
  sudo dmesg 2>/dev/null | grep -iE 'iommu|dmar|amd-vi' | head -8
  nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv 2>&1 | head -3
} > $O/probe.txt 2>&1
cat $O/probe.txt
[ -e /dev/kvm ] || { echo "NO_KVM: nested virtualization is not exposed on this instance"; exit 3; }
sudo chmod 666 /dev/kvm
IMG=ghcr.io/aitherium/awzeal:latest
sudo docker pull -q $IMG || exit 4
PROMPT="Write a detailed paragraph about how operating system schedulers share CPU time."
CORES="1 2 4 8"
for n in 16 24 32 48 64 96 128; do [ $n -lt $PHYS ] && CORES="$CORES $n"; done
CORES="$CORES $PHYS $THREADS"
echo "model,cores,phys,threads,run,decode_tok_s,prefill_s" > $O/bench.csv
for n in $(echo $CORES | tr ' ' '\n' | sort -nu); do
  for r in 1 2 3; do
    sudo docker run --rm --device /dev/kvm -e AWZEAL_SMP=$n -e AWZEAL_RAM=8G \
      -e AWZEAL_PROMPT="$PROMPT" $IMG > $O/run-$n-$r.txt 2>&1
    line=$(grep -a 'ZLOUT-END' $O/run-$n-$r.txt | tail -1)
    tok=$(echo "$line" | sed -n 's/.*(\([0-9.]*\) tok\/s).*/\1/p')
    pre=$(echo "$line" | sed -n 's/.*prefill=[0-9]* tok \([0-9.]*\)s.*/\1/p')
    echo "qwen3-0.6b,$n,$PHYS,$THREADS,$r,${tok:-FAIL},${pre:-}" | tee -a $O/bench.csv
  done
done
