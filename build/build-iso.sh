#!/bin/sh

# b35: CPU count for the guest compile is env-configurable
# (parallel-compile race during the compiler self-compile is the suspect
# for the app-pass freeze — ZEALOS_SMP=1 serialises it)
ZEALOS_SMP=${ZEALOS_SMP:-4}
ZEALOS_MACHINE=${ZEALOS_MACHINE:-q35}
ZEALOS_CPU=${ZEALOS_CPU:-""}

set -e

# Build OS using AUTO.ISO minimal auto-install as bootstrap to merge codebase, recompile system, attempt build limine UEFI hybrid ISO

# make sure we are in the correct directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
SCRIPT_NAME="$(basename "$0")"
EXPECTED_DIR="$(pwd -P)"

if test "${EXPECTED_DIR}" != "${SCRIPT_DIR}"
then
	( cd "$SCRIPT_DIR" || exit ; "./$SCRIPT_NAME" "$@" );
	exit
fi

[ "$1" = "--headless" ] && QEMU_HEADLESS='-display none'
# b51: --from-vdisk <base.raw> skips the leg-1 auto-install — the base
# vdisk (preserved as ./zealos-base.raw by a previous successful build)
# already carries the installed system; the staging + the leg-2 chain
# rebuild from the CURRENT src tree. The kernel-update path (~1-2h
# instead of the full 2-4h leg-1 + chain).
FROM_VDISK=""
if [ "$1" = "--from-vdisk" ]; then
    FROM_VDISK="$2"
    if [ ! -f "$FROM_VDISK" ]; then
        echo "ERROR: --from-vdisk base not found: $FROM_VDISK" >&2
        exit 1
    fi
    shift 2
fi

KVM=''
# WSL2: no kvm module in lsmod — /dev/kvm is the truth (nested virt).
[ -e /dev/kvm ] && KVM=' -accel kvm'
[ -z "$KVM" ] && echo "NOTE: no /dev/kvm — running QEMU legs on TCG (slow)"

# Guest RAM (b39+): the distro app pass froze at ~957MB with -m 1G — the
# compile's dictionary grows into the top of RAM and the module-hash alloc
# at each app file's first construct fails. ZEALOS_RAM=2G is the fix.
# b44: default = 2G (the freeze-probe now dumps the TOP 1GB to match —
# a 0-based dump of a 2G guest sees FL=0 forever and the stall arm
# killed every healthy compile at 15 min, measured 2026-08-28).
ZEALOS_RAM="${ZEALOS_RAM:-2G}"

# Set this true if you want to test ISOs in QEMU after building.
TESTING=false

TMPDIR="$(mktemp -d)"
TMPISODIR="$TMPDIR/iso"
TMPDISK="$TMPDIR/ZealOS.raw"
TMPMOUNT="$TMPDIR/mnt"

# Change this if your default QEMU version does not work and you have installed a different version elsewhere.
QEMU_BIN_PATH="$(dirname "$(which qemu-system-x86_64)")"

mount_tempdisk() {
	# nbd first (can wedge after repeated builds); fall back to a raw
	# loop mount with the partition offset (start sector 63 = 32256).
	if sudo modprobe nbd 2>/dev/null && \
	   sudo "$QEMU_BIN_PATH/qemu-nbd" -c /dev/nbd0 -f raw "$TMPDISK" 2>/dev/null && \
	   sudo partprobe /dev/nbd0 2>/dev/null && \
	   sudo mount /dev/nbd0p1 "$TMPMOUNT" 2>/dev/null; then
		return
	fi
	echo "NOTE: nbd path failed - using offset loop mount"
	sudo mount -o loop,offset=32256 "$TMPDISK" "$TMPMOUNT"
}

umount_tempdisk() {
	sync
	sudo umount "$TMPMOUNT" 2>/dev/null || true
	sudo "$QEMU_BIN_PATH/qemu-nbd" -d /dev/nbd0 2>/dev/null || true
}

script_cleanup() {
    sync

    # The drvfs mount can be "busy" at cleanup (the 9p client keeps it alive
    # after qemu exits); plain umount then fails and the rm -rf below would
    # CRAWL THE WHOLE C: DRIVE through the still-attached mount. Lazy umount
    # detaches it first (measured 2026-08-26: builds 25/27 cleanup logged
    # Permission-denied walking into src-cache-none/...).
    sudo umount "$SRC_MNT" >/dev/null 2>&1 || sudo umount -l "$SRC_MNT" >/dev/null 2>&1 || true
    sudo umount "$TMPMOUNT" >/dev/null 2>&1 || true
    sudo "$QEMU_BIN_PATH/qemu-nbd" -d /dev/nbd0 >/dev/null 2>&1 || true

    # On a failed build, KEEP the evidence: the serial log names the guest's
    # last compile position (the debugger-freeze tell) and the vdisk still
    # holds the healed copy + partial compile for forensics. The rm is
    # skipped; the caller removes it once the failure is understood.
    if [ -n "$PRESERVE_TMPDIR" ] && [ -d "$TMPDIR" ]; then
        echo "Preserving failed-build evidence: $TMPDIR (serial: serial-leg2.log, vdisk: ZealOS.raw)"
        return
    fi
    echo "Deleting temp folder ..."
    sudo rm -rf "$TMPDIR"
    sudo rm -rf "$TMPISODIR"
}

trap 'script_cleanup' EXIT

mkdir -p "$TMPMOUNT"
mkdir -p "$TMPISODIR"

# The WSL 9p client cache has served STALE copies of this source tree to
# the copy step (measured 2026-08-26: builds 24-25 heals fired on files
# that are byte-clean in the working tree; build 25 compiled a stale
# FluxIDT.ZC and froze the debugger at "Invalid lval at fv"). Copy from a
# FRESH cache=none drvfs mount of C: instead — verified 2026-08-26: sha1
# of a cache=none read of BackLib.ZC (83433f3c) == the Windows-side
# certutil hash exactly. Fall back to ../src (the old, possibly-stale
# path) only if the mount is impossible (e.g. non-WSL Linux).
SRC_MNT="$TMPDIR/src-cache-none"
SRC_ROOT=""
if sudo mkdir -p "$SRC_MNT" && sudo mount -t drvfs 'C:' "$SRC_MNT" -o cache=none 2>/dev/null; then
    SRC_ROOT="$SRC_MNT/AitherOS-Fresh/ZealOS-AI/src"
    echo "NOTE: source read through fresh cache=none drvfs mount (immune to 9p stale reads)"
else
    echo "WARNING: cache=none drvfs mount failed - falling back to ../src (9p cache CAN serve stale copies)" >&2
    SRC_ROOT="$(cd .. && pwd -P)/src"
fi

echo "Building ZealBooter..."
make -C ../zealbooter distclean all || ( echo "ERROR: ZealBooter build failed !" && false )

if [ -n "$FROM_VDISK" ]; then
    echo "Using preserved vdisk base: $FROM_VDISK (skipping the leg-1 auto-install)"
    cp "$FROM_VDISK" "$TMPDISK"
else
    echo "Making temp vdisk, running auto-install ..."
    "$QEMU_BIN_PATH/qemu-img" create -f raw "$TMPDISK" 1024M
    # b51: leg-1 timeout raised 2700 -> 14400 — the patched (KConfig headless)
    # install compile measured 75 min of healthy compiling without finishing
    # (2026-08-29 diag runs), and the SMP=1 serialized run churned 45+ min
    # through the compiler self-compile alone (registers healthy, no partition
    # yet, 2026-08-29 10:00) — 7200 would abort a slow-but-healthy install
    # mid-compile. The cap only fires if the install never completes.
    timeout 14400 "$QEMU_BIN_PATH/qemu-system-x86_64" -machine "$ZEALOS_MACHINE" $ZEALOS_CPU $KVM -drive format=raw,file="$TMPDISK" -m "$ZEALOS_RAM" -rtc base=localtime -smp "$ZEALOS_SMP" -cdrom AUTO.ISO -device isa-debug-exit $QEMU_HEADLESS -serial file:"$TMPDIR/serial-leg1.log" -monitor telnet:127.0.0.1:7101,server,nowait || true

    # The leg above ends `|| true` historically, which swallowed e.g. a KVM
    # misconfiguration and left an EMPTY vdisk — the failure then surfaced later
    # as a confusing nbd/mount error. Fail loudly here instead.
    if ! fdisk -l "$TMPDISK" 2>/dev/null | grep -qE "^$TMPDISK[0-9]"; then
        echo "ERROR: auto-install leg produced no partition (KVM/accel flag broken?) — aborting" >&2
        exit 1
    fi
fi

# Patch the boot loader to auto-select Drive C at every reboot of the chain.
# The AUTO.ISO-installed loader waits at "0. Old Boot Record / 1. Drive C"
# for a keypress, and QEMU's sendkey does NOT reach the guest's i8042 with
# -display none (measured 2026-08-26: ps2-kbd exists in qtree, sendkey
# returns OK, the i8042 status bit never sets — the input layer is dead
# without a display). Every distro-chain reboot would stall there forever.
# The getkey subroutine is `xor ah,ah; int 0x16` (32 E4 CD 16) — patched to
# `mov al,'1'; nop; nop` (B0 31 90 90) it returns '1' unconditionally and
# the menu boots Drive C. Two code copies exist (the loader blob is
# duplicated in the boot area); patch every match, assert before writing.
python3 - "$TMPDISK" << 'PYPATCH'
import sys
path = sys.argv[1]
# The getkey idiom is `xor ah,ah; int 0x16` (32 E4 CD 16). Patch every
# occurrence to `mov al,'1'; nop; nop` (B0 31 90 90 — same length). The
# 4-byte core is unambiguous and covers ALL copies: the boot-area blob
# (two copies 0x2A00 apart) AND /Boot/BootMHD2.BIN in the filesystem
# (a longer needle once missed the FS copy and the guest loaded the
# unpatched one).
needle = bytes.fromhex("32e4cd16")
patch  = bytes.fromhex("b0319090")
data = open(path, "rb").read()
hits, i = 0, 0
while True:
    i = data.find(needle, i)
    if i < 0:
        break
    hits += 1
    with open(path, "r+b") as f:
        f.seek(i)
        f.write(patch)
    i += 1
if hits == 0:
    print("WARN: boot-loader getkey not found — the chain will stall at the drive menu")
else:
    print(f"patched boot loader: {hits} getkey site(s) -> auto-select Drive C")
PYPATCH

echo "Copying all src/ code into vdisk Tmp/OSBuild/ ..."
rm -f "$SRC_ROOT/Home/Registry.ZC"
rm -f "$SRC_ROOT/Home/MakeHome.ZC"
rm -f "$SRC_ROOT/Boot/Kernel.ZXE"
mount_tempdisk
sudo mkdir -p "$TMPMOUNT/Tmp/OSBuild"
sudo cp -r "$SRC_ROOT"/* "$TMPMOUNT/Tmp/OSBuild/"
# FAIL LOUDLY on a silent staging failure: build 29's Tmp/OSBuild came out
# EMPTY on the vdisk while the System/Kernel/Compiler/Home overlays landed
# (census 2026-08-26) — a broken copy must abort here, not 90 minutes
# later at the extraction cp. src/ has 787 files; under 600 is a broken copy.
OSB_COUNT=$(find "$TMPMOUNT/Tmp/OSBuild" -type f 2>/dev/null | wc -l)
echo "staged $OSB_COUNT files into Tmp/OSBuild"
if [ "$OSB_COUNT" -lt 600 ]; then
    echo "ERROR: Tmp/OSBuild staging copy produced only $OSB_COUNT files (src has 787) — aborting" >&2
    exit 1
fi
# Overlay the CURRENT System + Kernel trees over the ISO-installed ones:
# the AUTO.ISO install ships the 2026-05 snapshot (old MakeSystem without
# the AI warning suppression, old AgentDemo with ternaries, and a STOCK
# KernelA.HH with no CKernelState). The boot compile (StartOS ->
# /System/MakeSystem) must compile the CURRENT tree or it crawls through
# thousands of re-include warnings, misses every fix, and dies with
# "Expecting type at CKernelState" when the extern block includes the
# stock kernel header.
sudo cp -r "$SRC_ROOT"/System/* "$TMPMOUNT/System/"
sudo cp -r "$SRC_ROOT"/Kernel/* "$TMPMOUNT/Kernel/"
# Overlay the CURRENT Compiler tree too: the AUTO.ISO-installed one is
# cp437-encoded (π = 0xE3) while the fork source is UTF-8 (π = 0xCF 0x80)
# — the MakeAll compiler self-compile reads /Compiler and died "Invalid
# lval at" on a stale 0xE3 copy (measured 2026-08-26: the WSL 9p cache
# served an old BackLib.ZC to the copy step; the heal+check below covers
# any survivor of any cause).
sudo cp -r "$SRC_ROOT"/Compiler/* "$TMPMOUNT/Compiler/"
# Home too: AI/AIInfer's AINetEnsure reads ipv4_globals.local_ip, but the
# installed (2026-05) /Home/Net/Protocols/IPV4.ZC predates that member —
# "Undefined identifier at ';'" otherwise. (cp -r adds/updates; the
# installed Registry.ZC, which the build deletes from src/Home, survives.)
sudo cp -r "$SRC_ROOT"/Home/* "$TMPMOUNT/Home/"
# The owner's 2026-05-03 commit was authored from a Windows checkout and
# its editor replaced UTF-8 bytes (∞/π/box-drawing) with U+FFFD
# replacement chars (0xEF 0xBF 0xBD) in KernelA.HH and KMain.ZC. Every
# compile the chain runs — the boot's StartOS -> /Kernel/KernelA.HH
# include, and MakeAll's CompComp + Comp(/Kernel) — dies "Invalid lval
# at" on those lines, and MakeAll's try/catch swallows it (PutExcept +
# Beep), so the fork kernel silently never installs and the chain runs
# on the stock kernel. Fail loudly instead.
# The pattern is $(printf octal) DELIBERATELY: the script runs under
# /bin/sh = dash, which has no $'...' ANSI-C quoting — a bare
# $'\xef\xbf\xbd' would pass the literal 8-byte text and silently
# never match (measured 2026-08-26: dash emits 24 5c 78 65 66 ...).
if sudo grep -rl --binary-files=text "$(printf '\357\277\275')" "$TMPMOUNT/Kernel/" "$TMPMOUNT/Compiler/" >/dev/null 2>&1; then
    echo "ERROR: U+FFFD mojibake in /Kernel or /Compiler overlay — restore the UTF-8 chars (∞/π/box-drawing)" >&2
    exit 1
fi
# The AUTO.ISO-installed system recompiles ITSELF on every boot
# (StartOS -> /System/MakeSystem). Its embedded source carries cp437
# bytes (pi/theta/...) in code positions that the ZealC parser rejects,
# so the installed copies must be healed BEFORE boot 1's compile, or the
# distro chain never reaches the /Tmp/OSBuild merge. Same sweep the tree
# fixer runs; binary assets (cursor bitmaps) are skipped.
if [ -f "$TMPMOUNT/System/Math/Math.ZC" ]; then
    # fail loudly: a silent heal miss costs a full 10-min build cycle.
    # The fixer must ALSO be verified: a line carrying two cp437 byte
    # kinds (theta+pi) once lost the later one while the heal reported
    # success (mtimes changed, files rewritten, same parse error). The
    # --check gate exits non-zero if any parser-fatal byte survived.
    sudo python3 ../tools/fix-cp437-code.py "$TMPMOUNT/System"
    sudo python3 ../tools/fix-cp437-code.py --check "$TMPMOUNT/System"
    # b42: the sweep splits each line on "//" and treats the rest as CODE —
    # but Mouse.ZC/WallPaper.ZC/ZSplash.ZC embed their sprite bitmaps INLINE
    # (raw bytes in the 0xF3-0xFC range) and the sweep replaces those DATA
    # bytes like cp437 chars, corrupting the bins: the boot compile then died
    # at Mouse.ZC:5-6 with 6x "Bin Not Found" -> "Missing expression at ';'"
    # (measured 2026-08-28: Mouse +32B, WallPaper +5B, ZSplash +72B growth).
    # Their TEXT regions are pure ASCII (verified), so they need no heal —
    # restore the fork originals AFTER the --check (the check must see the
    # healed tree, or the restored data bytes trip it).
    for f in Mouse.ZC WallPaper.ZC ZSplash.ZC; do
        sudo cp "$SRC_ROOT/System/$f" "$TMPMOUNT/System/$f"
        echo "restored System/$f — embedded sprite data intact"
    done
fi
# /Compiler is in the SAME class: the installed tree is cp437 while the
# fork source is UTF-8, and the compiler's self-compile (MakeAll's
# CompComp) reads it. A stale cp437 byte in a CODE position (measured:
# BackLib.ZC's `d == π` as 0xE3) is an undefined identifier and dies
# "Invalid lval at" in a debugger freeze — the heal converts it to ascii
# ("pi", which the kernel defines), and --check fails the build loudly.
if [ -d "$TMPMOUNT/Compiler" ]; then
    sudo python3 ../tools/fix-cp437-code.py "$TMPMOUNT/Compiler"
    sudo python3 ../tools/fix-cp437-code.py --check "$TMPMOUNT/Compiler"
fi
# /Kernel is the same class, read by the MakeAll kernel rebuild — the
# fork kernel's FIRST compile. Measured 2026-08-26: the fork's working
# tree had been corrupted to cp437 in stock Kernel files (KMathB.ZC's
# `return ∞;` as 0xEC, KMisc.ZC `Busy(I64 µS)`, StrPrint/StrScan's
# `d == ∞`), each an undefined identifier freezing the debugger at the
# first kernel compile. Restored from stock; this heal covers any stale
# copy the same way it covers /Compiler.
if [ -d "$TMPMOUNT/Kernel" ]; then
    sudo python3 ../tools/fix-cp437-code.py "$TMPMOUNT/Kernel"
    sudo python3 ../tools/fix-cp437-code.py --check "$TMPMOUNT/Kernel"
    # b42: same class — KMain.ZC's border_chars literals are raw cp437 STRING
    # bytes (the fork's box-drawing chars); the sweep ASCII-fies them.
    # Restore the fork original after the check.
    sudo cp "$SRC_ROOT/Kernel/KMain.ZC" "$TMPMOUNT/Kernel/KMain.ZC"
    echo "restored Kernel/KMain.ZC — border_chars bytes intact"
fi
# /Misc/Auto = the distro chain's OWN sources (AutoFullDistro0-5). The
# chain runs the INSTALLED /Misc (copied from AUTO.ISO at auto-install),
# not the /Tmp/OSBuild overlay, and once-compiles against the STOCK
# kernel's symbol table — which has no SerialStr (fork-only,
# Kernel/SerialUART.ZC). The chain files must therefore stay
# stock-compatible: measured 2026-08-26, AUTO.ISO contains zero
# "SerialStr" bytes, and the € once blamed on AutoFullDistro4.ZC:3 was a
# debugger render artifact (LexExcept prints no char). The heal below is
# a safety net for any cp437 byte in the snapshot's CODE positions; it
# currently fixes nothing (0 lines) and must keep doing so.
if [ -d "$TMPMOUNT/Misc/Auto" ]; then
    sudo python3 ../tools/fix-cp437-code.py "$TMPMOUNT/Misc"
    sudo python3 ../tools/fix-cp437-code.py --check "$TMPMOUNT/Misc"
fi
# b49: the distro chain is interactive at TWO popups, and headless QEMU
# cannot press Enter (the i8042 input is dead with -display none, measured
# 2026-08-26 — sendkey returns OK, the status bit never sets). AutoInstall
# writes Once/User = PopUp("#include /Misc/Auto/AutoFullDistro0;") — the
# chain's FIRST wall (measured b46-b48: the boot compile completes at
# FL=18,081, then the plateau = the popup waiting forever, and the probe
# killed the guest). AutoFullDistro4's PopUp is the chain's LAST wall
# (the ISO stage's gate). Patch BOTH popups out — byte-level, in-place,
# pad with spaces (ZealC is whitespace-insensitive) — replacing the
# popup with the direct #include;;Auto;; run of the next stage. The
# AFD4 patch must hit BOTH copies: the chain runs the INSTALLED /Misc
# for stages 0-3, then AFD3's CopyTree("C:/Tmp/OSBuild/", "C:/")
# overwrites /Misc with the OSBuild (fork) copy — so both must carry it.
sudo python3 - "$TMPMOUNT" << 'PYPATCH'
import sys
tmp = sys.argv[1]
# b49: AutoFullDistro4's PopUp is the chain's LAST interactive wall (the
# ISO stage's gate). Headless QEMU cannot press Enter (the i8042 input
# is dead with -display none, measured 2026-08-26). Patch it OUT in BOTH
# copies — the chain runs the INSTALLED /Misc for stages 0-3, then
# AFD3's CopyTree("C:/Tmp/OSBuild/", "C:/") overwrites /Misc with the
# OSBuild (fork) copy — so both must carry the fix. RunFile = the
# top-level run (an #include inside Auto()'s body would nest
# MakeMyISO's definition). Replacement is SHORTER than the needle
# (padded with spaces; ZealC is whitespace-insensitive).
# NOTE: the Once/User registry popup is NOT patched here — the leg-1
# chain already consumed it (the AFD0 ran during leg-1; measured
# 2026-08-28: the installed registry carries AFD0's OnceDrive entries,
# no popup).
patches = [
    (tmp + "/Misc/Auto/AutoFullDistro4.ZC",
     b'PopUp("#include \\"/Misc/Auto/AutoFullDistro5\\"");',
     b'RunFile("/Misc/Auto/AutoFullDistro5")'),
    (tmp + "/Tmp/OSBuild/Misc/Auto/AutoFullDistro4.ZC",
     b'PopUp("#include \\"/Misc/Auto/AutoFullDistro5\\"");',
     b'RunFile("/Misc/Auto/AutoFullDistro5")'),
]
for path, needle, repl in patches:
    try:
        data = open(path, "rb").read()
    except OSError as e:
        print(f"PATCH FAILED: {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if needle not in data:
        print(f"PATCH FAILED: needle not found in {path}", file=sys.stderr)
        sys.exit(1)
    assert len(repl) <= len(needle), f"{path}: replacement longer than needle"
    data = data.replace(needle, repl + b" " * (len(needle) - len(repl)))
    open(path, "wb").write(data)
    print(f"patched {path}: {len(needle)}-byte popup -> {repl.decode()} ({len(repl)}B)")
PYPATCH
umount_tempdisk

echo "Rebuilding kernel headers, kernel, OS, and building Distro ISO ..."
# pcnet NIC REQUIRED: the installed tree's /Home/Net/Drivers/Run.ZC runs
# NetDriverInit at module load (MakeHome) and throws 'NODRIVER' - crashing
# the boot into the debugger (ClassRep(NULL) -> "bad Pointer") - when no
# NIC exists. qemu's default NIC (e1000e) has no ZealOS driver; only
# PCNet-II/RTL8139/VIRTIO/E1000 are registered, so use pcnet.
# Cap the leg-2 qemu: a guest frozen at the ZealC debugger (a compile
# error, any fault) runs forever with no output — measured 2026-08-26,
# builds 22-23 hung ~40 min each. BUT the box's nested KVM makes the
# HEALTHY chain slow, not fast: RAM forensics of build 28's replay
# (pmemsave + FL-record census) showed 233 distinct files compiled in
# 25 min with NO rendered error and NO loop — a live, slow compile the
# 30-min cap was killing mid-run. timeout 5400 lets a healthy chain
# finish; a real freeze still fails loudly at the extraction cp below,
# and the evidence-preservation (serial tail + kept TMPDIR) makes the
# freeze position recoverable either way.
# b41: -cdrom AUTO.ISO REQUIRED on leg 2. QEMU 10 (Debian trixie,
# reinstalled after the 2026-08-26 WSL disaster) creates a DEFAULT EMPTY
# ide-cd on AHCI port 2 of the q35 machine. The installed stock kernel
# probes every AHCI port at BlkDevsInitAll; the empty tray fails the
# ATAPI capacity command and PANICS (KMain+0x022C, Except at
# AHCIPortCmdWait) before StartOS ever runs. Media present -> the probe
# succeeds exactly as leg-1's install probe did.
# Freeze probe for leg 2 (build 31+): the guest's serial is silent (0
# bytes across builds 22-29) and the distro compile is RAM-only until
# MakeMyISO, so neither serial nor vdisk mtimes can tell a healthy
# compile from the ZealC debugger freeze — measured 2026-08-26 on build
# 29's replay: the chain FREEZES at ~3 min with a rendered-empty error
# window and the debugger loop spins the CPU for the rest of the budget
# (builds 27-29 all died this way; build 28's "233 files in 25 min"
# reading sampled an already-frozen guest). The FL-record census moves
# only while the compiler runs: pmemsave the live guest every 45s and
# count FL:C:/path,line records; two identical consecutive counts =
# frozen -> the probe kills the leg-2 qemu and the build aborts fast.
# timeout 7200 (2h): build 29's census + raw-file mtime show the chain
# compiling RAM-only (the distro builds the new system in memory) and
# starting the MakeMyISO write ~90 min in — the 5400 cap killed it
# mid-ISO-write with no FAT entry yet (hence "MyDistro.ISO.C missing").
# cache=none: with writeback, qemu buffers guest writes and a partial
# ISO leaves no trace; write-through keeps the watchdog honest.
# b43: the leg-2 monitor port must be FREE before launch — a prior run's
# orphaned qemu (wrapper killed, child survived — the b32 class) holds it and
# the leg-2 qemu dies instantly ("Failed to find an available port"), the
# freeze-probe then watches the WRONG guest, and the build burns to the
# extraction failure (measured 2026-08-28, twice in one day).
# b49: the chain is a staged reboot ladder (AutoFullDistro0-5) with TWO
# external-stage boundaries: AutoFullDistro0 and AutoFullDistro5 end in
# OutU8(0x501,0x31) — the isa-debug-exit beacon — making the leg-2 qemu
# exit 99 mid-chain (AFD0's beacon = "boot code + next-stage Once
# installed, relaunch me"; AFD5's = "ISO written, done"). The script
# must relaunch the guest per beacon until /Tmp/MyDistro.ISO.C exists
# or the freeze-probe killed the guest (a compile error / chain stall).
# No timeout wrapper: the freeze-probe owns each leg-2 qemu's lifetime
# (it kills the qemu DIRECTLY on a detected freeze; a SIGKILLed timeout
# wrapper would orphan its qemu child, which then holds monitor port
# 7102 and burns CPU — measured build 32: "Address already in use").
# The probe's own deadline per round = 350 probes x 45s = 262 min.
DISTRO_DONE=""
for ROUND in 1 2 3 4 5 6; do
    echo "=== leg-2 round $ROUND (distro chain stage) ==="
    fuser -k 7102/tcp 2>/dev/null || true
    "$QEMU_BIN_PATH/qemu-system-x86_64" -machine "$ZEALOS_MACHINE" $ZEALOS_CPU $KVM -drive format=raw,file="$TMPDISK",cache=none -m "$ZEALOS_RAM" -rtc base=localtime -smp "$ZEALOS_SMP" -device isa-debug-exit -netdev user,id=n0 -device pcnet,netdev=n0 -cdrom AUTO.ISO $QEMU_HEADLESS -serial file:"$TMPDIR/serial-leg2.log" -monitor telnet:127.0.0.1:7102,server,nowait &
    LEG2_PID=$!
    python3 ../tools/freeze-probe.py 7102 "$TMPDIR/probe-ram.bin" "$LEG2_PID" "$ZEALOS_RAM" > "$TMPDIR/freeze-probe.log" 2>&1 &
    PROBE_PID=$!
    if wait $LEG2_PID; then LEG2_RC=0; else LEG2_RC=$?; fi
    kill $PROBE_PID 2>/dev/null || true
    echo "[round $ROUND] leg-2 qemu exited rc=$LEG2_RC"
    echo "[probe] freeze-probe log tail:" >> "$TMPDIR/freeze-probe.log" 2>/dev/null || true
    tail -6 "$TMPDIR/freeze-probe.log" 2>/dev/null || true
    if grep -qE "FREEZE DETECTED|STALL DETECTED" "$TMPDIR/freeze-probe.log" 2>/dev/null; then
        echo "ERROR: freeze-probe killed the guest (compile error / chain stall) — screendump + plateau RAM preserved" >&2
        break
    fi
    # The guest exits 99 on the isa-debug-exit beacon (a stage boundary,
    # NOT the end). 137 = the probe's SIGKILL (a freeze — caught above).
    # Either way the next truth lives on the vdisk: does the ISO exist?
    if mount_tempdisk; then
        if [ -e "$TMPMOUNT/Tmp/MyDistro.ISO.C" ]; then
            echo "=== MyDistro.ISO.C present after round $ROUND — the chain reached MakeMyISO ==="
            umount_tempdisk
            DISTRO_DONE=1
            break
        fi
        umount_tempdisk
    fi
    echo "[round $ROUND] no ISO yet — relaunching for the next chain stage"
done
if [ -z "$DISTRO_DONE" ]; then
    echo "ERROR: the distro chain did not produce /Tmp/MyDistro.ISO.C (probe log above)" >&2
    PRESERVE_TMPDIR=1
    exit 1
fi

LIMINE_BINARY_BRANCH="v10.x-binary"

if [ -d "limine" ]
then
	cd limine
	git remote set-branches origin $LIMINE_BINARY_BRANCH
	git fetch
	git remote set-head origin $LIMINE_BINARY_BRANCH
	git switch $LIMINE_BINARY_BRANCH
	git config --local pull.ff true
	git config --local pull.rebase true
	git pull
	rm limine

	cd ..
else
    git clone https://github.com/limine-bootloader/limine.git --branch=$LIMINE_BINARY_BRANCH --depth=1
fi
make -C limine

touch limine/Limine-BIOS-HDD.HH
echo "/*\$WW,1\$" > limine/Limine-BIOS-HDD.HH
cat limine/LICENSE >> limine/Limine-BIOS-HDD.HH
echo "*/\$WW,0\$" >> limine/Limine-BIOS-HDD.HH
cat limine/limine-bios-hdd.h >> limine/Limine-BIOS-HDD.HH
sed -i 's/const uint8_t/U8/g' limine/Limine-BIOS-HDD.HH
sed -i "s/\[\]/\[$(grep -o "0x" ./limine/limine-bios-hdd.h | wc -l)\]/g" limine/Limine-BIOS-HDD.HH

mount_tempdisk
echo "Extracting MyDistro ISO from vdisk ..."
if ! cp "$TMPMOUNT/Tmp/MyDistro.ISO.C" ./ZealOS-MyDistro.iso; then
    echo "ERROR: MyDistro.ISO.C missing - the guest chain did not reach MakeMyISO" >&2
    echo "       serial-leg2.log tail (the freeze position is in the last lines):" >&2
    tail -60 "$TMPDIR/serial-leg2.log" 2>/dev/null >&2 || true
    PRESERVE_TMPDIR=1
    exit 1
fi
sudo rm -f "$TMPMOUNT/Tmp/MyDistro.ISO.C"
echo "Setting up temp ISO directory contents for use with limine xorriso command ..."
sudo cp -rf "$TMPMOUNT"/* "$TMPISODIR/"
sudo rm -f "$TMPISODIR/Boot/OldMBR.BIN"
sudo rm -f "$TMPISODIR/Boot/BootMHD2.BIN"
sudo mkdir -p "$TMPISODIR/EFI/BOOT"
sudo cp limine/Limine-BIOS-HDD.HH "$TMPISODIR/Boot/Limine-BIOS-HDD.HH"
sudo cp limine/BOOTX64.EFI "$TMPISODIR/EFI/BOOT/BOOTX64.EFI"
sudo cp limine/limine-uefi-cd.bin "$TMPISODIR/Boot/Limine-UEFI-CD.BIN"
sudo cp limine/limine-bios-cd.bin "$TMPISODIR/Boot/Limine-BIOS-CD.BIN"
sudo cp limine/limine-bios.sys "$TMPISODIR/Boot/Limine-BIOS.SYS"
sudo cp ../zealbooter/bin/kernel "$TMPISODIR/Boot/ZealBooter.ELF"
sudo cp ../zealbooter/limine.conf "$TMPISODIR/Boot/Limine.CONF"
echo "Copying DVDKernel.ZXE over ISO Boot/Kernel.ZXE ..."
# b40: the chain's MakeMyISO wrote the DVD kernel INTO the ISO (the
# extracted Boot/Kernel.ZXE already is it) and no separate
# /Tmp/DVDKernel.ZXE exists on the vdisk — a bare mv died under set -e.
if [ -e "$TMPMOUNT/Tmp/DVDKernel.ZXE" ]; then
    sudo mv "$TMPMOUNT/Tmp/DVDKernel.ZXE" "$TMPISODIR/Boot/Kernel.ZXE"
else
    echo "NOTE: no /Tmp/DVDKernel.ZXE on vdisk — ISO's extracted Boot/Kernel.ZXE is the DVD kernel"
fi
sudo rm -f "$TMPISODIR/Tmp/DVDKernel.ZXE"
umount_tempdisk

# b51: preserve the installed vdisk as the base for --from-vdisk rebuilds —
# the kernel-update path (the staging + the leg-2 chain against the CURRENT
# src tree) skips the leg-1 auto-install entirely.
cp "$TMPDISK" ./zealos-base.raw
echo "Preserved vdisk base: ./zealos-base.raw ($(du -h "$TMPDISK" | cut -f1))"

truncate -s 32K bios_boot.img

xorriso -as mkisofs -R -r -J -b Boot/Limine-BIOS-CD.BIN \
        -no-emul-boot -boot-load-size 4 -boot-info-table \
        --efi-boot Boot/Limine-UEFI-CD.BIN \
        -efi-boot-part --efi-boot-image --protective-msdos-label \
        -append_partition 4 21686148-6449-6E6F-744E-656564454649 bios_boot.img \
        -appended_part_as_gpt \
        "$TMPISODIR" -o ZealOS-limine.iso

rm bios_boot.img

./limine/limine bios-install ZealOS-limine.iso --no-gpt-to-mbr-isohybrid-conversion

if [ "$TESTING" = true ]; then
	if [ ! -d "ovmf" ]; then
	    echo "Downloading OVMF..."
	    mkdir ovmf
	    cd ovmf
	    curl -o OVMF-X64.zip https://efi.akeo.ie/OVMF/OVMF-X64.zip
	    7z x OVMF-X64.zip
	    cd ..
	fi
	echo "Testing limine-zealbooter-xorriso isohybrid boot in UEFI mode ..."
	"$QEMU_BIN_PATH/qemu-system-x86_64" -machine "$ZEALOS_MACHINE" $ZEALOS_CPU $KVM -m "$ZEALOS_RAM" -rtc base=localtime -bios ovmf/OVMF.fd -smp "$ZEALOS_SMP" -cdrom ZealOS-limine.iso $QEMU_HEADLESS
	echo "Testing limine-zealbooter-xorriso isohybrid boot in BIOS mode ..."
	"$QEMU_BIN_PATH/qemu-system-x86_64" -machine "$ZEALOS_MACHINE" $ZEALOS_CPU $KVM -m "$ZEALOS_RAM" -rtc base=localtime -smp "$ZEALOS_SMP" -cdrom ZealOS-limine.iso $QEMU_HEADLESS
	echo "Testing native ZealC MyDistro legacy ISO in BIOS mode ..."
	"$QEMU_BIN_PATH/qemu-system-x86_64" -machine "$ZEALOS_MACHINE" $ZEALOS_CPU $KVM -m "$ZEALOS_RAM" -rtc base=localtime -smp "$ZEALOS_SMP" -cdrom ZealOS-MyDistro.iso $QEMU_HEADLESS
fi

# comment these 2 lines if you want lingering old Distro ISOs
rm -f ./ZealOS-PublicDomain-BIOS-*.iso
rm -f ./ZealOS-BSD2-UEFI-*.iso

mv ./ZealOS-MyDistro.iso ./ZealOS-PublicDomain-BIOS-$(date +%Y-%m-%d-%H_%M_%S).iso
mv ./ZealOS-limine.iso ./ZealOS-BSD2-UEFI-$(date +%Y-%m-%d-%H_%M_%S).iso

echo "Finished."
echo
echo "ISOs built:"
ls | grep ZealOS-P
ls | grep ZealOS-B
echo
