#!/usr/bin/env python3
"""zealos_mbr v4: chunked MBR loader for the ZealOS-AI fork.

BOOT CONTRACT (derived from KStart16.ZC + the image layout, 2026-08-28):
  - Load the kernel FILE at LOAD_BASE = 0x7BE0 (= MEM_BOOT_BASE - sizeof(CZXE),
    sizeof(CZXE)=0x20).  File 0x20 (SYS_KERNEL) then lands at 0x7C00.
  - Enter CS=0x7C0:0, EAX=3 (BOOT_SRC_HARDDRIVE), EBX=0, DL=drive.
    The kernel's GET_IP trick re-bases CS to the file-0 segment; MEM_BOOT_BASE
    = CS*16 = 0x7BE0; GS = CS-2 points at the CZXE header; the fixed area
    (file 0xF8420) lands at SYS_FIXED_AREA = 0x100000.  The kernel applies its
    own patch table at CORE0_32BIT_INIT (adds MEM_BOOT_BASE to each entry).
  - The image is 95% zero padding.  The live content is 5 chunks:
      (0x000000, 0x002EB70), (0x4301B8, 0x430A0B), (0x436C40, 0x437472),
      (0x443790, 0x444C4F), (0x5899F8, 0x591BD0)
    Only those are copied: the padding spans the device region 0xA0000..0x100000
    (EBDA at 0x9FC00, option ROMs, BIOS ROM) -- writing it wedged SeaBIOS's
    AHCI state (the v2 iteration-30 hang).
  - The loader's own stack lives BELOW 0x7BE0 (SS=0x600, SP=0x1BE0).

ROOT CAUSES FOUND 2026-08-28 (v3 faults, gdb-proven):
  1. si (the table pointer) is a LINEAR address (0x6000+END) but the DAP-setup
     reads used DS=0x600 (segmented) -> read VGA ROM garbage into the DAP ->
     INT 13 got garbage lba/count and failed silently (CF never checked).
  2. mov esi,eax in the copy section ZEROES si (SI is ESI's low half).  The
     loop then did add si,12 on the zeroed pointer: iteration 2 read the IVT
     as its table (garbage dest/count), iteration 3's garbage dest hit the ROM
     and #GP'd at the rep.
  3. The scratch cell [0x40C] was written/read through FLAT DS -> linear
     0x40C (BDA), not the intended 0x640C.  Harmless in this table (buf_off
     was 0) but wrong.

v4 fixes: the table pointer lives in a MEMORY CELL at 0x6410 and is re-derived
every iteration; ALL table and DAP access happens through FLAT DS (selector 8)
with LINEAR addresses; the copy needs no scratch cell (esi computed last);
the INT 13 carry flag is checked ('F' marker + halt on failure).

Addressing map (all linear):
  0x6000        loader+table (CS=0x600, self-copied from 0x7C00)
  0x63E0-0x63E5 GDTR (limit, base=0x63EA)
  0x63EA-0x6401 GDT (null + flat32 @0x63F2; 0x63FC+ is unused GDT tail)
  0x63FC-0x640B DAP (16 bytes; seg 0x8000, lba_hi 0, count/buf_off/lba per entry)
  0x640E        drive byte (DL saved)
  0x6410        table-pointer cell (16-bit)
  0x7BE0        CZXE header + kernel start (MEM_BOOT_BASE-0x20)
  0x80000+      INT 13 staging buffer
"""
b = bytearray(512)


def w(off, data):
    b[off:off + len(data)] = data


o = 0


def emit(data):
    global o
    w(o, data)
    o += len(data)


CONT = 0x1A

# --- entry (runs at linear 0x7C00; BIOS enters with CS=0, EIP=0x7C00!) ---
emit(bytes([0xFA]))                                   # cli
emit(bytes([0xFC]))                                   # cld
emit(bytes([0xB8, 0xC0, 0x07]))                       # mov ax,0x7C0 (MBR segment - CS is 0 at entry!)
emit(bytes([0x8E, 0xD8]))                             # mov ds,ax
emit(bytes([0xB8, 0x00, 0x06]))                       # mov ax,0x600
emit(bytes([0x8E, 0xC0]))                             # mov es,ax
emit(bytes([0xB9, 0x00, 0x02]))                       # mov cx,0x200
emit(bytes([0x33, 0xF6]))                             # xor si,si
emit(bytes([0x33, 0xFF]))                             # xor di,di
emit(bytes([0xF3, 0xA4]))                             # rep movsb
emit(bytes([0xEA, CONT, 0x00, 0x00, 0x06]))           # jmp far 0x600:CONT
assert o == CONT

# --- CONT: relocated (CS=0x600, linear 0x6000) ---
emit(bytes([0xB8, 0x00, 0x06]))                       # mov ax,0x600
emit(bytes([0x8E, 0xD8]))                             # mov ds,ax
emit(bytes([0x8E, 0xC0]))                             # mov es,ax
emit(bytes([0xFA]))                                   # cli
emit(bytes([0x8E, 0xD0]))                             # mov ss,ax
emit(bytes([0xBC, 0xE0, 0x1B]))                       # mov sp,0x1BE0 (stack top BELOW CZXE @0x7BE0)
emit(bytes([0xFB]))                                   # sti
emit(bytes([0xB2, 0x80]))                             # mov dl,0x80
emit(bytes([0x88, 0x16, 0xDE, 0x03]))                 # mov [0x3DE],dl (save drive @ linear 0x640E)

# GDTR: limit 0x17, base 0x63EA -- LINEAR address of GDT[0] (0x6000+0x3EA)
emit(bytes([0xC7, 0x06, 0xE0, 0x03, 0x17, 0x00]))     # mov word [0x3E0],0x0017
emit(bytes([0x66, 0xC7, 0x06, 0xE2, 0x03, 0xEA, 0x63, 0x00, 0x00]))  # mov dword [0x3E2],0x63EA
emit(bytes([0x0F, 0x01, 0x16, 0xE0, 0x03]))           # lgdt [0x3E0] (loads once; persists across PE toggles)
emit(bytes([0x66, 0xC7, 0x06, 0xE6, 0x03, 0x00, 0x00, 0x00, 0x00]))  # scratch 0
# GDT[0] null @0x3EA
for i in (0x3EA, 0x3EC, 0x3EE, 0x3F0):
    emit(bytes([0xC7, 0x06, i & 0xFF, i >> 8, 0x00, 0x00]))
# GDT[1] flat32 @0x3F2: 0x00CF92000000FFFF  (access byte at +5 = 0x3F7!)
emit(bytes([0xC7, 0x06, 0xF2, 0x03, 0xFF, 0xFF]))   # +0 limit low = 0xFFFF
emit(bytes([0xC7, 0x06, 0xF4, 0x03, 0x00, 0x00]))   # +2 base 15:0 = 0x0000
emit(bytes([0xC7, 0x06, 0xF6, 0x03, 0x00, 0x92]))   # +4 base 23:16 = 0x00, +5 access = 0x92
emit(bytes([0xC7, 0x06, 0xF8, 0x03, 0xCF, 0x00]))   # +6 flags+limHi = 0xCF, +7 base31:24 = 0
# DAP @0x3FC (linear 0x63FC): size 16, count/buf/LBA set per entry
emit(bytes([0xC7, 0x06, 0xFC, 0x03, 0x10, 0x00]))   # +0 size = 0x10
emit(bytes([0xC7, 0x06, 0xFE, 0x03, 0x00, 0x00]))   # +2 count (per entry, flat phase)
emit(bytes([0xC7, 0x06, 0x00, 0x04, 0x00, 0x00]))   # +4 buf off (per entry, flat phase)
emit(bytes([0xC7, 0x06, 0x02, 0x04, 0x00, 0x80]))   # +6 buf segment 0x8000
emit(bytes([0x66, 0xC7, 0x06, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00]))  # +8 lba lo (per entry, flat phase)
emit(bytes([0x66, 0xC7, 0x06, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00]))  # +12 lba hi = 0
# marker 'S' (once) - serial
emit(bytes([0xB0, 0x53, 0xBA, 0xF8, 0x03, 0xEE]))

# ============ TABLE-DRIVEN LOAD LOOP ============
# The table pointer lives in memory at [0x6410]; si is re-derived each
# iteration so the copy's mov esi,eax (which zeroes SI) cannot corrupt it.
# All table/DAP access happens through FLAT DS (PE on, ds=es=8) with LINEAR
# addresses.  INT 13 runs in real mode with the DAP at DS:0x3FC.

TBLCELL = 0x6410
tbli_emit = o
emit(bytes([0xC7, 0x06, 0x10, 0x04, 0x00, 0x00]))     # mov word [0x6410],TABLE (patched)
bx_emit = o
emit(bytes([0xBB, 0x00, 0x00]))                       # mov bx,N (patched)

LOOP = o
# --- si := [TBLCELL] (real mode, DS=0x600 -> linear 0x6410) ---
emit(bytes([0x8B, 0x36, 0x10, 0x04]))                 # mov si,[0x6410]
# --- flat phase: read table, fill DAP (all LINEAR; no xor needed: the
#     32-bit lba read supplies its own high word, count/buf_off writes are 16-bit) ---
emit(bytes([0x0F, 0x20, 0xC0]))                       # mov eax,cr0
emit(bytes([0x0C, 0x01]))                             # or al,1
emit(bytes([0x0F, 0x22, 0xC0]))                       # mov cr0,eax
emit(bytes([0xB8, 0x08, 0x00]))                       # mov ax,0x8
emit(bytes([0x8E, 0xD8]))                             # mov ds,ax
emit(bytes([0x8E, 0xC0]))                             # mov es,ax
emit(bytes([0x8B, 0x44, 0x0A]))                       # mov ax,[si+10]  (buf_off)
emit(bytes([0xA3, 0x00, 0x64]))                       # mov [0x6400],ax (DAP buf_off, LINEAR)
emit(bytes([0x66, 0x8B, 0x04]))                       # mov eax,[si+0]  (lba)
emit(bytes([0x66, 0xA3, 0x04, 0x64]))                 # mov [0x6404],eax (DAP lba)
emit(bytes([0x8B, 0x44, 0x08]))                       # mov ax,[si+8]   (count)
emit(bytes([0xA3, 0xFE, 0x63]))                       # mov [0x63FE],ax (DAP count)
# --- back to real mode for INT 13 (BX survives: SeaBIOS preserves all
#     registers but AX/CF; proven by fault dump EBX=9 after 2 INT 13s) ---
emit(bytes([0x0F, 0x20, 0xC0]))                       # mov eax,cr0
emit(bytes([0x24, 0xFE]))                             # and al,0xFE
emit(bytes([0x0F, 0x22, 0xC0]))                       # mov cr0,eax
emit(bytes([0xB8, 0x00, 0x06]))                       # mov ax,0x600
emit(bytes([0x8E, 0xD8]))                             # mov ds,ax
emit(bytes([0x8E, 0xC0]))                             # mov es,ax
emit(bytes([0xBE, 0xFC, 0x03]))                       # mov si,0x3FC (DAP, segmented)
emit(bytes([0xB4, 0x42]))                             # mov ah,0x42
emit(bytes([0xB2, 0x80]))                             # mov dl,0x80 (drive)
emit(bytes([0xCD, 0x13]))                             # int 0x13
# CF check: 'F' + halt on read failure.  FAIL is >127 bytes away, so this is
# a NEAR jcc (0F 82 rel16) -- a rel8 jc would wrap and jump BACKWARD.
fail_off = o
emit(bytes([0x0F, 0x82, 0x00, 0x00]))                 # jc near FAIL (patched)
emit(bytes([0x8B, 0x36, 0x10, 0x04]))                 # mov si,[0x6410] (table ptr again)
emit(bytes([0xB0, 0x31, 0xBA, 0xF8, 0x03, 0xEE]))     # serial '1'
# --- flat phase: the copy ---
emit(bytes([0x0F, 0x20, 0xC0]))                       # mov eax,cr0
emit(bytes([0x0C, 0x01]))                             # or al,1
emit(bytes([0x0F, 0x22, 0xC0]))                       # mov cr0,eax
emit(bytes([0xB8, 0x08, 0x00]))                       # mov ax,0x8
emit(bytes([0x8E, 0xD8]))                             # mov ds,ax
emit(bytes([0x8E, 0xC0]))                             # mov es,ax
emit(bytes([0x66, 0x8B, 0x7C, 0x04]))                 # mov edi,[si+4]  (dest)
emit(bytes([0x66, 0x31, 0xC0]))                       # xor eax,eax
emit(bytes([0x8B, 0x44, 0x08]))                       # mov ax,[si+8]  (count)
emit(bytes([0x66, 0xC1, 0xE0, 0x09]))                 # shl eax,9
emit(bytes([0x66, 0x89, 0xC1]))                       # mov ecx,eax
emit(bytes([0x66, 0x31, 0xC0]))                       # xor eax,eax
emit(bytes([0x8B, 0x44, 0x0A]))                       # mov ax,[si+10] (buf_off)
emit(bytes([0x66, 0x05, 0x00, 0x00, 0x08, 0x00]))     # add eax,0x80000
emit(bytes([0x66, 0x89, 0xC6]))                       # mov esi,eax (si dies here - cell is the truth)
emit(bytes([0x67, 0xF3, 0xA4]))                       # rep a32 movsb
# --- back to real mode ---
emit(bytes([0x0F, 0x20, 0xC0]))                       # mov eax,cr0
emit(bytes([0x24, 0xFE]))                             # and al,0xFE
emit(bytes([0x0F, 0x22, 0xC0]))                       # mov cr0,eax
emit(bytes([0xB8, 0x00, 0x06]))                       # mov ax,0x600
emit(bytes([0x8E, 0xD8]))                             # mov ds,ax
emit(bytes([0x8E, 0xC0]))                             # mov es,ax
# advance the table pointer cell
emit(bytes([0x83, 0x06, 0x10, 0x04, 0x0C]))           # add word [0x6410],12
emit(bytes([0x4B]))                                   # dec bx
# LOOP is >127 bytes back, so a rel8 jnz would wrap forward into the table
# (the v4.0 crash: table bytes executed as code -> #UD at the GDTR data).
# Use jz + jmp-near:  jz SKIP (rel8, +3) ; jmp LOOP (E9 rel16) ; SKIP:
emit(bytes([0x74, 0x03]))                             # jz SKIP
jmp_off = o
emit(bytes([0xE9, 0x00, 0x00]))                       # jmp LOOP (patched)
b[jmp_off + 1:jmp_off + 3] = ((LOOP - (jmp_off + 3)) & 0xFFFF).to_bytes(2, "little")

# --- done ---
emit(bytes([0xB0, 0x32, 0xBA, 0xF8, 0x03, 0xEE]))     # serial '2'
emit(bytes([0xFA]))                                   # cli
emit(bytes([0x66, 0x31, 0xC0]))                       # xor eax,eax
emit(bytes([0x66, 0xB8, 0x03, 0x00, 0x00, 0x00]))     # mov eax,3
emit(bytes([0x66, 0x31, 0xDB]))                       # xor ebx,ebx
emit(bytes([0x8A, 0x16, 0xDE, 0x03]))                 # mov dl,[0x3DE]
emit(bytes([0xB0, 0x33, 0xBA, 0xF8, 0x03, 0xEE]))     # serial '3'
emit(bytes([0xEA, 0x00, 0x00, 0xC0, 0x07]))           # jmp far 0x7C0:0

# --- FAIL: 'F' + halt (AFTER done -- the loop-exit jz must NOT fall into it) ---
FAIL = o
emit(bytes([0xB0, 0x46, 0xBA, 0xF8, 0x03, 0xEE]))     # serial 'F'
emit(bytes([0xF4]))                                   # hlt
emit(bytes([0xEB, 0xFD]))                             # jmp $-1
b[fail_off + 2:fail_off + 4] = ((FAIL - (fail_off + 4)) & 0xFFFF).to_bytes(2, "little")
END = o
assert END <= 0x200, "loader overflows: %#x" % END

# ============ BUILD THE ENTRY TABLE ============
LOAD_BASE = 0x7BE0
KERNEL_LBA = 0x2B3AD
CHUNKS = [
    (0x000000, 0x002EB70),
    (0x4301B8, 0x430A0B),
    (0x436C40, 0x437472),
    (0x443790, 0x444C4F),
    (0x5899F8, 0x591BD0),
]
import os
if os.path.exists("/tmp/kernel.zxe"):
    img = open("/tmp/kernel.zxe", "rb").read()
    if len(img) != 0x591BD0:
        raise SystemExit("image size mismatch: %#x" % len(img))
    for i in range(len(img)):
        in_chunk = any(s <= i < e for s, e in CHUNKS)
        if not in_chunk and img[i] != 0:
            raise SystemExit("non-zero byte at 0x%x outside chunks" % i)
    print("image verified: chunks match zero-run model")

entries = []
for s, e in CHUNKS:
    fs = s >> 9
    n = (e - s + 511) >> 9
    buf_off = s & 0x1FF
    dest = LOAD_BASE + s
    while n:
        c = min(n, 64)
        entries.append((KERNEL_LBA + fs, dest, c, buf_off))
        fs += c
        n -= c
        dest += c * 512
N_ENTRIES = len(entries)
print("entries:", N_ENTRIES)
for en in entries:
    print("  lba=%#x dest=%#x count=%d buf_off=%#x" % en)
TABLE = 0x6000 + END
assert END + len(entries) * 12 <= 0x1FE, "no room for table: END=%#x" % END

# C7 06 <disp16> <imm16>: displacement at +2..+3, IMMEDIATE at +4..+5!
b[tbli_emit + 4:tbli_emit + 6] = TABLE.to_bytes(2, "little")
b[bx_emit + 1:bx_emit + 3] = N_ENTRIES.to_bytes(2, "little")

tbl = END
for lba, dest, cnt, boff in entries:
    w(tbl, lba.to_bytes(4, "little") + dest.to_bytes(4, "little") + cnt.to_bytes(2, "little") + boff.to_bytes(2, "little"))
    tbl += 12

b[0x1FE] = 0x55
b[0x1FF] = 0xAA
open("/tmp/zealos_mbr4.bin", "wb").write(bytes(b))
print("zealos_mbr4.bin v4: %d bytes code, %d entries, LOOP at 0x%x, TABLE at 0x%x" % (END, N_ENTRIES, LOOP, TABLE))
