#!/usr/bin/env python3
"""holyc_lint.py <file.ZC>... -- flag constructs the ZealOS compiler rejects or mis-parses.

Exit 1 when any finding, 0 when clean. Measured traps (ZealOS-AI build loop):
  struct-assign   a[i] = b[j] / *p = *q style whole-class copies (use MemCopy)
  ternary         ?: does not exist in ZealC
  continue        no `continue` keyword in ZealC (use goto)
  inner-decl      a typed declaration nested deeper than the function body's first level
  sci-literal     1.0e-5 style float literals
Comments and string/char literals are ignored.
"""
import re
import sys

TYPES = r"(?:I8|U8|I16|U16|I32|U32|I64|U64|F64|Bool|U0|C[A-Z][A-Za-z0-9]+)"
RX = {
    "continue": re.compile(r"\bcontinue\b"),
    "ternary": re.compile(r"\?[^:;]*:"),
    "sci-literal": re.compile(r"\b\d+\.\d*[eE][-+]?\d+"),
    "inner-decl": re.compile(rf"^\t\t+{TYPES}\s+\**[A-Za-z_]\w*\s*(=|;|,|\[)"),
    "struct-assign": re.compile(r"(\w+(?:->|\.)\w+\[[^\]]+\])\s*=\s*\w+(?:->|\.)\w+\[[^\]]+\]\s*;"),
}


def strip(line: str) -> str:
    out, q = [], None
    i = 0
    while i < len(line):
        c = line[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == q:
                q = None
            i += 1
            continue
        if c in "\"'":
            q = c
        elif line.startswith("//", i):
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def main(paths):
    bad = 0
    for p in paths:
        in_block = False
        for n, raw in enumerate(open(p, "rb").read().decode("latin-1").splitlines(), 1):
            line = raw
            if in_block:
                if "*/" in line:
                    in_block = False
                    line = line.split("*/", 1)[1]
                else:
                    continue
            if "/*" in line and "*/" not in line.split("/*", 1)[1]:
                in_block = True
                line = line.split("/*", 1)[0]
            code = strip(line)
            for name, rx in RX.items():
                if rx.search(code if name != "inner-decl" else line):
                    print(f"{p}:{n}: {name}: {raw.strip()}")
                    bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
