import sys

path = sys.argv[1]
data = open(path, "rb").read()
lines = data.split(b"\n")
for i, ln in enumerate(lines):
    low = ln.lower()
    if b"border_chars" in ln or b"aux_font" in ln:
        print(i + 1, ln[:100].hex())
