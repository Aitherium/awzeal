#!/usr/bin/env python3
"""tokcmp.py <harness> <gguf> -- raw-text tokenization: our engine vs the live 4B llama-server /tokenize.

Read-only against the server (POST /tokenize on its loopback port). Exit 0 when every case matches.
"""
import json
import os
import subprocess
import sys
import urllib.request

CASES = ["? Answer", "France? Answer", "a Answer", " Answer", "x? y", "Hi! There", "1. Two", "ok, Answer",
         "What is the capital of France? Answer in one short sentence.", "a.b c", "end.\n Next", "(x) Answer",
         "Hello, world! How's it going? I'm fine.", "x = 12345 + 6.78;", "line1\n\n\nline2\r\n  indented",
         "tabs\there  and   spaces   ", "naïve café — déjà vu", "日本語のテキスト。",
         "emoji \U0001F642 ok!?", "  leading", "a...b!!! c??", "It'S THE'LL WE'VE",
         "#include <stdio.h>\nint main(){return 0;}", "1,000,000.50", "Q: Why?\nA: Because.", "--flag=value --x",
         "?\n\nNext", "end!\r\n\r\nx", "a ? b", "x ?? y"]
URL = os.environ.get("BZ_REF_URL", "http://127.0.0.1:8267") + "/tokenize"


def ref(t):
    r = urllib.request.Request(URL, json.dumps({"content": t, "add_special": False}).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=30))["tokens"]


def main():
    gguf = sys.argv[2].rsplit("/", 1)[1]
    bad = 0
    for t in CASES:
        out = subprocess.run(["podman", "run", "--rm", "--cpuset-cpus=15", "-v", "/root/zealos-ai/models:/m:ro", "-v",
                              "/root/zealos-ai/bonsai/out:/o:ro", "localhost/zealos-bonsai-build:1", "/o/harness",
                              "/m/" + gguf, "tok", t], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        ours = [int(x) for x in out.split(",") if x]
        r = ref(t)
        ok = ours == r
        bad += not ok
        print("OK  " if ok else "DIFF", repr(t)[:60], "" if ok else f"ours {ours} ref {r}")
    print(f"TOKENIZER {'PASS' if not bad else 'FAIL'}: {len(CASES) - bad}/{len(CASES)} cases identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
