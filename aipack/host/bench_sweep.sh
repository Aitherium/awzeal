#!/bin/bash
# bench_sweep.sh <qwen3|bonsai> "<cores...>" [ram]
# Boots the awzeal guest once per core count with the benchmark as RUN.ZC, records the best
# of three decode tok/s (plus prefill) per count to bench-<model>.csv, then puts the disk's
# proof RUN.ZC back. Runs inside awnix. Disk edits go through mtools in a throwaway container.
set -uo pipefail
W=${ZWORK:-/root/zealos-ai}
MODEL=$1; CORES=${2//,/ }; RAM=${3:-}   # "4,8,16" (wsl.exe re-splits spaces) or "4 8 16"
case $MODEL in
  qwen3)  DISK=$W/aidisk.img;         BENCH=$W/aipack/zealos/BENCHQ.ZC;          PROOF=$W/aipack/zealos/RUN.ZC;          RAM=${RAM:-4G} ;;
  bonsai) DISK=$W/bonsai/bzdisk.img;  BENCH=$W/aipack/bonsai/zealos/BENCHBZ.ZC;  PROOF=$W/aipack/bonsai/zealos/RUN2.ZC;  RAM=${RAM:-12G} ;;
  *) echo "model must be qwen3 or bonsai"; exit 2 ;;
esac
CSV=$W/bench-$MODEL.csv
put_run() {   # put_run <file>: copy it onto the disk as RUN.ZC
  podman run --rm -v "$W:/w" -v "$(dirname "$1"):/src:ro" localhost/zealos-bonsai-build:1 bash -c "
    command -v mcopy >/dev/null || { apt-get update -qq >/dev/null && apt-get install -y -qq mtools >/dev/null 2>&1; }
    MTOOLS_SKIP_CHECK=1 mcopy -o -i /w/${DISK#$W/}@@1048576 /src/$(basename "$1") ::/RUN.ZC"
}
[ -f "$CSV" ] || echo "model,cores,host_nproc,decode_tok_s_best,decode_tok_s_all,prefill_tok_s_best,when" > "$CSV"
put_run "$BENCH" || { echo "could not stage the benchmark"; exit 1; }
for n in $CORES; do
  echo "== $MODEL cores=$n ram=$RAM"
  cd "$W" && ZRUN=$W/run-bench ZVNC=44 ZRAM=$RAM ZSMP=$n ZDISK=$DISK \
    python3 -u aipack/host/zctl.py zealos-ai-live.iso @aipack/steps/bench.txt > "$W/bench-run.log" 2>&1
  python3 - "$W/run-bench/serial.log" "$MODEL" "$n" "$(nproc)" "$CSV" <<'EOF'
import re, sys, time
log, model, n, nproc, csv = sys.argv[1:6]
text = open(log, "rb").read().decode("latin-1")
dec = [float(x) for x in re.findall(r"decode=\d+ tok [\d.]+s \(([\d.]+) tok/s\)", text)]
pre = [int(a) / float(b) for a, b in re.findall(r"prefill=(\d+) tok ([\d.]+)s", text) if float(b) > 0]
if not dec:
    print(f"   no ZLOUT-END lines for cores={n} (see {log})")
    sys.exit(0)
row = [model, n, nproc, f"{max(dec):.2f}", "/".join(f"{d:.2f}" for d in dec),
       f"{max(pre):.2f}" if pre else "", time.strftime("%Y-%m-%dT%H:%M:%S")]
open(csv, "a").write(",".join(row) + "\n")
print(f"   decode best {max(dec):.2f} tok/s (runs {row[4]})  prefill best {row[5]} tok/s")
EOF
done
put_run "$PROOF" && echo "proof RUN.ZC restored"
column -s, -t < "$CSV"
