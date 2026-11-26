#!/usr/bin/env python3
"""fix-cp437-code.py — replace cp437 non-ASCII bytes in CODE contexts.

The boot-time MakeSystem compile (which recompiles the installed system
on every boot) rejects cp437 bytes (0xE3=pi, 0xE9=theta, ...) appearing
in code positions: identifiers, expressions, default args. Comments are
fine (the lexer skips them). Apps are NOT in the boot compile, so only
src/System/ (and Kernel/) need the sweep.

Usage: python3 tools/fix-cp437-code.py [paths...]   (default: src/System)
"""
import sys

# cp437 -> ascii replacements for code contexts
MAP = {
    0xE0: b"gamma",   # Γ
    0xE1: b"delta",   # Δ
    0xE2: b"epsilon", # Θ (0xE2 is actually Θ in cp437)
    0xE3: b"pi",      # π
    0xE4: b"sigma",   # Σ
    0xE5: b"phi",     # Φ
    0xE6: b"psi",     # Ψ
    0xE7: b"omega",   # Ω
    0xE8: b"alpha",   # α
    0xE9: b"theta",   # θ
    0xEA: b"beta",    # β
    0xEB: b"xi",      # ξ
    0xEC: b"eta",     # η
    0xED: b"zeta",    # ζ
    0xEE: b"mu",      # μ
    0xEF: b"nu",      # ν
    0xF0: b"sigma2",  # ς
    0xF1: b"+-",      # ±
    0xF2: b"delta2",  # δ
    0xF3: b"infinity",# ∞
    0xF4: b"phi2",    # φ
    0xF5: b"epsilon2",# ε
    0xF6: b"intersection",  # ∩
    0xF7: b"approx",  # ≈
    0xF8: b"not_equal",      # ≠
    0xF9: b"less_equal",     # ≤
    0xFA: b"greater_equal",  # ≥
    0xFB: b"sqrt",   # √
    0xFC: b"degree", # °
}


def fix_file(path: str) -> int:
    try:
        d = open(path, "rb").read()
    except OSError:
        return 0
    lines = d.split(b"\n")
    fixed = 0
    for i, l in enumerate(lines):
        code = l.split(b"//")[0]
        for byte, repl in MAP.items():
            if code.count(bytes([byte])):
                rest = b"//".join(l.split(b"//")[1:])
                newcode = code.replace(bytes([byte]), repl)
                lines[i] = newcode + (b"//" + rest if rest else b"")
                fixed += 1
                break
    if fixed:
        open(path, "wb").write(b"\n".join(lines))
        print(f"{path}: {fixed} lines fixed")
    return fixed


def main() -> int:
    import glob

    roots = sys.argv[1:] or ["src/System"]
    total = 0
    for root in roots:
        for p in glob.glob(f"{root}/**/*.ZC", recursive=True):
            total += fix_file(p)
    print(f"total: {total} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
