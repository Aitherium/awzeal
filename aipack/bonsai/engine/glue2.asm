; ============================================================================
; glue2.asm -- the Bonsai 2.5 C engine (bonsai.c) as a loadable ZealOS module
;
; Same contract as aipack/engine/zeal_glue.asm (ENGINE.BIN): a flat, base-0,
; RIP-relative image whose first 64 bytes are the ZLE1 header with the six
; entry offsets AILLM.ZC binds as HolyC function pointers. Each entry follows
; the HolyC convention (args on the stack, arg1 at [RBP+16], callee pops),
; saves every GPR but RAX plus MXCSR, runs on a private 1 MiB stack, turns
; on CR4.OSXSAVE / XCR0 AVX for the core it runs on, then calls the SysV C
; entry point.
;
; ZlInit's model argument is either a contiguous GGUF image or a BZPARTS1
; part table (see bonsai.h); AIBONSAI.ZC builds the table from the split
; model files so no single allocation or FAT32 file has to hold 7 GB.
; ============================================================================
bits 64
default rel

extern bz_init
extern bz_begin
extern bz_next
extern bz_end
extern bz_reset
extern bz_stat
extern bz_threads
extern bz_worker

global zl_header

ZL_MAGIC   equ 0x31454C5A           ; "ZLE1"
ZL_VERSION equ 2                    ; 2 = Bonsai C engine, BZPARTS1-aware ZlInit

section .text.zhdr progbits alloc exec nowrite align=64
zl_header:
    dd ZL_MAGIC
    dd ZL_VERSION
    dd zl_init  - zl_header         ; +8   I64 ZlInit(U8 *model_or_parts, I64 size, U8 *arena, I64 arena_size, I64 ctx)
    dd zl_begin - zl_header         ; +12  I64 ZlBegin(U8 *prompt, I64 think)  -> n_prompt
    dd zl_next  - zl_header         ; +16  I64 ZlNext(U8 *buf, I64 cap)       -> bytes, -1 = done
    dd zl_end   - zl_header         ; +20  I64 ZlEnd()                         -> ctx position
    dd zl_reset - zl_header         ; +24  I64 ZlReset()
    dd zl_stat  - zl_header         ; +28  I64 ZlStat(I64 which)
    dd zl_worker - zl_header        ; +32  I64 ZlWorker(I64 tid)   per-core job loop
    dd zl_threads - zl_header       ; +36  I64 ZlThreads(I64 n)    cores per matmul
    times 56 - ($ - zl_header) db 0  ; +40..+55 reserved: same slot meaning as ENGINE.BIN
    db "BONSAI25"                   ; +56  engine id (last 8 header bytes, clear of every slot)

%macro HC_ENTER 0
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    rcx
    push    rdx
    push    rsi
    push    rdi
    push    r8
    push    r9
    push    r10
    push    r11
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8
    stmxcsr [rsp]
    mov     rax, rsp
    lea     rsp, [rel zl_stack_top]
    push    rax                         ; caller RSP (restored by HC_LEAVE)
    sub     rsp, 8                      ; 16-byte aligned for SysV calls
    call    zl_enable_avx
    ldmxcsr [rel zl_mxcsr_default]
%endmacro

%macro HC_LEAVE 1
    vzeroupper
    add     rsp, 8
    pop     rsp
    ldmxcsr [rsp]
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     r11
    pop     r10
    pop     r9
    pop     r8
    pop     rdi
    pop     rsi
    pop     rdx
    pop     rcx
    pop     rbx
    pop     rbp
    ret     %1
%endmacro

; arg N (1-based) of the HolyC call, valid after HC_ENTER (RBP still frames it)
%define HC_ARG(n) [rbp + 8 + 8*(n)]

zl_enable_avx:
    mov     rax, cr4
    or      rax, (1 << 18) | (1 << 10) | (1 << 9)   ; OSXSAVE | OSXMMEXCPT | OSFXSR
    mov     cr4, rax
    xor     ecx, ecx
    xgetbv
    or      eax, 7                      ; x87 | SSE | AVX
    xor     ecx, ecx
    xsetbv
    ret

zl_threads:
    HC_ENTER
    mov     rdi, HC_ARG(1)
    call    bz_threads
    HC_LEAVE 8

; ZlWorker runs on the CALLING task's stack (one per core), re-aligned for AVX.
zl_worker:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    rcx
    push    rdx
    push    rsi
    push    rdi
    push    r8
    push    r9
    push    r10
    push    r11
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8
    stmxcsr [rsp]
    mov     rax, rsp
    and     rsp, -32
    push    rax
    sub     rsp, 24
    call    zl_enable_avx
    ldmxcsr [rel zl_mxcsr_default]
    mov     rdi, HC_ARG(1)
    call    bz_worker
    vzeroupper
    add     rsp, 24
    pop     rsp
    ldmxcsr [rsp]
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     r11
    pop     r10
    pop     r9
    pop     r8
    pop     rdi
    pop     rsi
    pop     rdx
    pop     rcx
    pop     rbx
    pop     rbp
    ret     8

zl_init:
    HC_ENTER
    mov     rdi, HC_ARG(1)
    mov     rsi, HC_ARG(2)
    mov     rdx, HC_ARG(3)
    mov     rcx, HC_ARG(4)
    mov     r8,  HC_ARG(5)
    call    bz_init
    HC_LEAVE 40

zl_begin:
    HC_ENTER
    mov     rdi, HC_ARG(1)
    mov     rsi, HC_ARG(2)
    call    bz_begin
    HC_LEAVE 16

zl_next:
    HC_ENTER
    mov     rdi, HC_ARG(1)
    mov     rsi, HC_ARG(2)
    call    bz_next
    HC_LEAVE 16

zl_end:
    HC_ENTER
    call    bz_end
    HC_LEAVE 0

zl_reset:
    HC_ENTER
    call    bz_reset
    HC_LEAVE 0

zl_stat:
    HC_ENTER
    mov     rdi, HC_ARG(1)
    call    bz_stat
    HC_LEAVE 8

section .rodata
align 4
zl_mxcsr_default: dd 0x1F80

section .bss
alignb 64
zl_stack:          resb 1024 * 1024
zl_stack_top:
