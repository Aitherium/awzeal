#!/usr/bin/env python3
"""Verify the b49 patch needles against the real source bytes (dry-run).
The fork AFD4 == the stock's for the popup line — the chain runs the
installed copies for stages 0-3 and the OSBuild copy for AFD4+."""
tests = [
    ("src/Misc/Auto/AutoFullDistro4.ZC",
     b"PopUp(\"#include \\\"/Misc/Auto/AutoFullDistro5\\\"\");",
     b"#include \"/Misc/Auto/AutoFullDistro5\";;Auto;;"),
    # The registry needle: AutoInstall writes OnceDrive('C', "PopUp(...)")
    # — the ZC literal parses to backslash-quotes in the stored entry.
    ("src/Misc/Auto/AutoInstall.ZC",
     b"PopUp(\"#include \\\"/Misc/Auto/AutoFullDistro0\\\";\");",
     b"#include \"/Misc/Auto/AutoFullDistro0\";;Auto;;"),
]
base = "/mnt/c/AitherOS-Fresh/ZealOS-AI/"
for path, needle, repl in tests:
    data = open(base + path, "rb").read()
    if needle in data:
        print(f"{path}: needle FOUND ({len(needle)}B); repl {len(repl)}B <= {len(needle)}B OK")
        patched = data.replace(needle, repl + b" " * (len(needle) - len(repl)))
        for ln in patched.splitlines():
            if b"AutoFullDistro" in ln and b"PopUp" not in ln:
                print(f"  patched line: {ln[:80]!r}")
    else:
        print(f"{path}: needle MISSING — context around the target:")
        i = data.find(b"AutoFullDistro0" if b"0" in needle else b"PopUp")
        print(" ", data[max(0, i - 40):i + 60])
