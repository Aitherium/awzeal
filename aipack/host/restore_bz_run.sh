#!/bin/bash
# put the Bonsai proof RUN2.ZC back on bzdisk.img as RUN.ZC (after an interrupted bench sweep)
W=${ZWORK:-/root/zealos-ai}
podman run --rm -v "$W:/w" localhost/zealos-bonsai-build:1 bash -c "
  command -v mcopy >/dev/null || { apt-get update -qq >/dev/null && apt-get install -y -qq mtools >/dev/null 2>&1; }
  MTOOLS_SKIP_CHECK=1 mcopy -o -i /w/bonsai/bzdisk.img@@1048576 /w/aipack/bonsai/zealos/RUN2.ZC ::/RUN.ZC &&
  MTOOLS_SKIP_CHECK=1 mtype -i /w/bonsai/bzdisk.img@@1048576 ::/RUN.ZC | head -3"
