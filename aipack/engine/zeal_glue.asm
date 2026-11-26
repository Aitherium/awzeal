; ============================================================================
; zeal_glue.asm -- llm.asm Qwen3 engine as a loadable ZealOS (HolyC) module
;
; Links with llm.asm's shared/{math,matmul,tokenizer,transformer}.asm built
; -DBARE_METAL. Produces a flat, base-0 image; pack_engine.py turns it into
; ENGINE.BIN (header + relocation table) that AILLM.ZC loads into the code
; heap, relocates, and calls through HolyC function pointers.
;
; HolyC calling convention (ZealOS KernelA.HH SF_ARG1 = 0x10): arguments are
; on the stack, arg1 at [RBP+16] after the standard prologue, and the CALLEE
; pops them (RET n*8). Every entry saves all GPRs but RAX, the MXCSR, and
; switches to a private 512 KiB stack aligned for AVX.
;
; ZealOS runs ring 0 but never enables AVX: each entry sets CR4.OSXSAVE and
; XCR0 |= x87|SSE|AVX on whatever core it is running on (cheap, idempotent).
; ============================================================================
%include "constants.inc"

extern g_ctx
extern model_init
extern transformer_forward
extern tok_build_chat_prompt
extern tok_encode_bpe
extern argmax_f32
extern matmul_q8_0_range
extern gqa_attn_range
%if FAST_Q8
extern quantize_x_q8_0
%endif

global sys_mmap_anon
global sys_munmap
global sys_write_all
global str_len
global tp_matmul_q8_0
global tp_gqa_attn
global zl_header

ZL_MAGIC   equ 0x31454C5A           ; "ZLE1"
ZL_VERSION equ 2
ZL_MAX_CPUS equ 64

section .text.zhdr progbits alloc exec nowrite align=64
; ---- fixed header: AILLM.ZC reads these fields at the image base ----------
zl_header:
    dd ZL_MAGIC
    dd ZL_VERSION
    dd zl_init  - zl_header         ; +8   I64 ZlInit(U8 *model, I64 size, U8 *arena, I64 arena_size, I64 ctx)
    dd zl_begin - zl_header         ; +12  I64 ZlBegin(U8 *prompt, I64 think)       -> n_prompt
    dd zl_next  - zl_header         ; +16  I64 ZlNext(U8 *buf, I64 cap)            -> bytes, -1 = done
    dd zl_end   - zl_header         ; +20  I64 ZlEnd()                              -> ctx position
    dd zl_reset - zl_header         ; +24  I64 ZlReset()
    dd zl_stat  - zl_header         ; +28  I64 ZlStat(I64 which)
    dd zl_worker - zl_header        ; +32  I64 ZlWorker(I64 tid)   run jobs on this core; returns after ~20 ms idle
    dd zl_threads - zl_header       ; +36  I64 ZlThreads(I64 n)    n = BSP + workers (1 = single core)
    dd zl_system - zl_header        ; +40  I64 ZlSystem(U8 *text)  prefill a system turn (tools block) -> ctx pos
    times 64 - ($ - zl_header) db 0

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

; I64 ZlInit(U8 *model, I64 size, U8 *arena, I64 arena_size, I64 ctx)
zl_init:
    HC_ENTER
    mov     rax, HC_ARG(3)
    add     rax, 63
    and     rax, -64
    mov     [rel zl_heap_cur], rax
    mov     rax, HC_ARG(3)
    add     rax, HC_ARG(4)
    mov     [rel zl_heap_end], rax
    mov     qword [rel zl_chat_pos], 0
    mov     qword [rel zl_heap_base_seen], 0
    mov     qword [rel zl_max_tok], 256
    mov     rdi, HC_ARG(1)
    mov     rsi, HC_ARG(2)
    mov     rdx, HC_ARG(5)
    call    model_init
    HC_LEAVE 40

; I64 ZlBegin(U8 *prompt, I64 think) -- tokenize with the Qwen3 chat template,
; prefill from the retained KV position, pick the first token. -> n_prompt
zl_begin:
    HC_ENTER
    lea     rbx, [rel g_ctx]
    mov     rdi, HC_ARG(1)
    mov     rsi, [rbx + CTX_BUF_PROMPT]
    mov     rdx, MAX_PROMPT_TOKENS
    xor     ecx, ecx                    ; raw = 0 (chat template)
    mov     r8, HC_ARG(2)               ; think
    call    tok_build_chat_prompt
    mov     r12, rax                    ; n_prompt
    test    r12, r12
    jz      .out
    ; KV room: reset the conversation when this turn + 64 would not fit
    mov     rax, [rel zl_chat_pos]
    lea     rcx, [rax + r12 + 64]
    cmp     rcx, [rbx + CTX_MAX_SEQ]
    jb      .room
    mov     qword [rel zl_chat_pos], 0
.room:
    mov     r13, [rel zl_chat_pos]      ; pos
    mov     r14, [rbx + CTX_BUF_PROMPT]
    xor     r15d, r15d                  ; i
.pf:
    cmp     r15, r12
    jae     .pf_done
    mov     edi, [r14 + r15*4]
    mov     rsi, r13
    lea     rax, [r15 + 1]
    xor     edx, edx
    cmp     rax, r12
    sete    dl                          ; logits only for the last prompt token
    call    transformer_forward
    inc     r15
    inc     r13
    jmp     .pf
.pf_done:
    mov     [rel zl_pos], r13
    mov     rdi, [rbx + CTX_BUF_LOGITS]
    mov     rsi, [rbx + CTX_VOCAB_SIZE]
    call    argmax_f32
    mov     [rel zl_next_tok], rax
    mov     qword [rel zl_n_gen], 0
    mov     qword [rel zl_last_unfed], 0
.out:
    mov     rax, r12
    HC_LEAVE 16

; I64 ZlNext(U8 *buf, I64 cap) -- emit the pending token's UTF-8 bytes into
; buf, then run one decode step. -> byte count, or -1 when the turn is over.
zl_next:
    HC_ENTER
    lea     rbx, [rel g_ctx]
    mov     r12, [rel zl_next_tok]
    mov     rax, [rel zl_n_gen]
    cmp     rax, [rel zl_max_tok]
    jae     .done
    cmp     r12, [rbx + CTX_EOS_1]
    je      .done
    cmp     r12, [rbx + CTX_EOS_2]
    je      .done
    mov     rax, [rel zl_pos]
    cmp     rax, [rbx + CTX_MAX_SEQ]
    jae     .done
    ; copy token bytes
    xor     r13d, r13d                  ; len
    cmp     r12, [rbx + CTX_VOCAB_SIZE]
    jae     .copied
    mov     rcx, [rbx + CTX_TOK_OFFS_PTR]
    mov     edx, [rcx + r12*4]
    mov     r8d, [rcx + r12*4 + 4]
    sub     r8, rdx
    mov     rsi, [rbx + CTX_TOK_BYTES_PTR]
    add     rsi, rdx
    mov     rdi, HC_ARG(1)
    mov     rcx, HC_ARG(2)
    cmp     r8, rcx
    jbe     .len_ok
    mov     r8, rcx
.len_ok:
    mov     r13, r8
    mov     rcx, r8
    rep     movsb
.copied:
    inc     qword [rel zl_n_gen]
    ; feed the emitted token, pick the next one
    mov     rdi, r12
    mov     rsi, [rel zl_pos]
    mov     edx, 1
    call    transformer_forward
    inc     qword [rel zl_pos]
    mov     rdi, [rbx + CTX_BUF_LOGITS]
    mov     rsi, [rbx + CTX_VOCAB_SIZE]
    call    argmax_f32
    mov     [rel zl_next_tok], rax
    mov     rax, r13
    HC_LEAVE 16
.done:
    mov     rax, -1
    HC_LEAVE 16

; I64 ZlEnd() -- close the assistant turn (<|im_end|>\n) so the next ZlBegin
; continues the same conversation. -> retained context position
zl_end:
    HC_ENTER
    lea     rbx, [rel g_ctx]
    mov     r13, [rel zl_pos]
    lea     rax, [r13 + 2]
    cmp     rax, [rbx + CTX_MAX_SEQ]
    jae     .save
    mov     edi, TOK_IM_END
    mov     rsi, r13
    xor     edx, edx
    call    transformer_forward
    inc     r13
    mov     edi, TOK_NL
    mov     rsi, r13
    xor     edx, edx
    call    transformer_forward
    inc     r13
.save:
    mov     [rel zl_chat_pos], r13
    mov     rax, r13
    HC_LEAVE 0

; I64 ZlReset() -- forget the conversation
zl_reset:
    HC_ENTER
    mov     qword [rel zl_chat_pos], 0
    xor     eax, eax
    HC_LEAVE 0

; I64 ZlStat(I64 which): 0 ctx pos, 1 max_seq, 2 arena bytes used, 3 vocab,
;                        4 tokens generated this turn, 100+n set max_tokens=n
zl_stat:
    HC_ENTER
    lea     rbx, [rel g_ctx]
    mov     rcx, HC_ARG(1)
    cmp     rcx, 100
    jb      .q
    sub     rcx, 100
    mov     [rel zl_max_tok], rcx
    mov     rax, rcx
    jmp     .r
.q: mov     rax, [rel zl_pos]
    test    rcx, rcx
    jz      .r
    mov     rax, [rbx + CTX_MAX_SEQ]
    cmp     rcx, 1
    je      .r
    mov     rax, [rel zl_heap_cur]
    sub     rax, [rel zl_heap_base_seen]
    cmp     rcx, 2
    je      .r
    mov     rax, [rbx + CTX_VOCAB_SIZE]
    cmp     rcx, 3
    je      .r
    mov     rax, [rel zl_n_gen]
.r:
    HC_LEAVE 8

; I64 ZlSystem(U8 *text) -- start a fresh conversation with a system turn:
;   <|im_start|>system\n{text}<|im_end|>\n   prefilled into the KV cache (no output).
; The Hermes tool contract lives in this turn; later ZlBegin turns continue after it.
zl_system:
    HC_ENTER
    lea     rbx, [rel g_ctx]
    mov     r14, [rbx + CTX_BUF_PROMPT]
    mov     dword [r14], TOK_IM_START
    lea     rdi, [rel zl_s_system]
    call    str_len
    mov     rsi, rax
    lea     rdi, [rel zl_s_system]
    lea     rdx, [r14 + 4]
    mov     rcx, 16
    call    tok_encode_bpe
    lea     r12, [rax + 1]              ; tokens so far
    mov     rdi, HC_ARG(1)
    call    str_len
    mov     rsi, rax
    mov     rdi, HC_ARG(1)
    lea     rdx, [r14 + r12*4]
    mov     rcx, MAX_PROMPT_TOKENS - 8
    sub     rcx, r12
    call    tok_encode_bpe
    add     r12, rax
    mov     dword [r14 + r12*4], TOK_IM_END
    mov     dword [r14 + r12*4 + 4], TOK_NL
    add     r12, 2
    xor     r13d, r13d                  ; a system turn always starts the conversation
    xor     r15d, r15d
.sp:
    cmp     r15, r12
    jae     .sp_done
    cmp     r13, [rbx + CTX_MAX_SEQ]
    jae     .sp_done
    mov     edi, [r14 + r15*4]
    mov     rsi, r13
    xor     edx, edx                    ; no logits needed
    call    transformer_forward
    inc     r15
    inc     r13
    jmp     .sp
.sp_done:
    mov     [rel zl_chat_pos], r13
    mov     [rel zl_pos], r13
    mov     rax, r13
    HC_LEAVE 8

; I64 ZlThreads(I64 n) -- cores used per matmul/attention (BSP + n-1 workers)
zl_threads:
    HC_ENTER
    mov     rax, HC_ARG(1)
    cmp     rax, 1
    jge     .lo
    mov     eax, 1
.lo:
    cmp     rax, ZL_MAX_CPUS
    jle     .hi
    mov     eax, ZL_MAX_CPUS
.hi:
    mov     [rel tp_n_threads], rax
    HC_LEAVE 8

; I64 ZlWorker(I64 tid) -- called in a loop by a ZealOS task pinned to one
; core. Runs every job slice for tid until ~20 ms pass with no job, then
; returns jobs-done so the task can Yield. Uses the task's own stack (each
; core needs its own), re-aligned for AVX.
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
    mov     r12, HC_ARG(1)              ; tid
    lea     r13, [rel tp_done_arr]
    mov     rcx, r12
    shl     rcx, 6
    add     r13, rcx                    ; &done[tid]
    xor     r15d, r15d                  ; jobs done this call
.idle_reset:
    mov     rbx, 4000000                ; ~20 ms of pause
.wait:
    mov     eax, [rel tp_gen]
    cmp     eax, [r13]
    jne     .job
    pause
    dec     rbx
    jnz     .wait
    jmp     .out
.job:
    mov     r14d, eax
    mov     rsi, [rel tp_n_threads]
    cmp     r12, rsi
    jae     .mark
    mov     rdi, r12
    mov     rcx, [rel tp_M]
    call    tp_slice
    call    tp_run_slice
    inc     r15
.mark:
    mov     [r13], r14d                 ; TSO release: this core's slice is done
    jmp     .idle_reset
.out:
    mov     rax, r15
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

section .text

; ---- OS shim for the shared engine (SysV ABI) -------------------------------
; void *sys_mmap_anon(uint64 size) -- 64B-aligned zeroed bump allocation from
; the arena ZlInit was given; negative on OOM (model_init checks the sign).
sys_mmap_anon:
    mov     rax, [rel zl_heap_base_seen]
    test    rax, rax
    jnz     .have_base
    mov     rax, [rel zl_heap_cur]
    mov     [rel zl_heap_base_seen], rax
.have_base:
    mov     rax, [rel zl_heap_cur]
    add     rax, 63
    and     rax, -64
    mov     rcx, rdi
    add     rcx, 63
    and     rcx, -64
    lea     rdx, [rax + rcx]
    cmp     rdx, [rel zl_heap_end]
    ja      .oom
    mov     [rel zl_heap_cur], rdx
    mov     r8, rax
    mov     r9, rcx
    vxorps  ymm0, ymm0, ymm0
.z: test    r9, r9
    jz      .zd
    vmovdqa [r8], ymm0
    vmovdqa [r8 + 32], ymm0
    add     r8, 64
    sub     r9, 64
    jmp     .z
.zd:
    ret
.oom:
    mov     rax, -12
    ret

sys_munmap:
    xor     eax, eax
    ret

sys_write_all:                          ; output goes through ZlNext instead
    xor     eax, eax
    ret

str_len:
    xor     eax, eax
.lp:
    cmp     byte [rdi + rax], 0
    je      .ret
    inc     rax
    jmp     .lp
.ret:
    ret

; ---- multi-core dispatch -----------------------------------------------------
; Same lock-free generation/spin-barrier design as llm.asm smp.asm, but the
; workers are ZealOS tasks pinned to CPUs 1..n-1 that sit in ZlWorker(tid).
; tid 0 is the calling (BSP) task. tp_K = 0 marks an attention job.

; (rax = r0, rdx = r1) = slice of M for tid rdi out of n rsi; M in rcx
tp_slice:
    push    rbx
    mov     rbx, rcx                    ; M
    mov     rax, rcx
    imul    rax, rdi
    lea     rax, [rax + rsi - 1]
    xor     edx, edx
    div     rsi
    mov     r8, rax                     ; r0
    lea     rax, [rdi + 1]
    cmp     rax, rsi
    je      .last
    imul    rax, rbx
    lea     rax, [rax + rsi - 1]
    xor     edx, edx
    div     rsi
    mov     rdx, rax
    jmp     .out
.last:
    mov     rdx, rbx
.out:
    mov     rax, r8
    pop     rbx
    ret

; run job slice [r0=rax, r1=rdx) using the tp_* parameters
tp_run_slice:
    cmp     rdx, rax
    jle     .ret
    mov     r8, rax
    mov     r9, rdx
    mov     rdi, [rel tp_W]
    mov     rsi, [rel tp_x]
    mov     rdx, [rel tp_out]
    mov     rcx, [rel tp_K]
%if FAST_Q8
    test    rcx, rcx
    jz      .attn
%endif
    jmp     matmul_q8_0_range           ; (W, x, out, K, r0, r1)
%if FAST_Q8
.attn:
    mov     rcx, r8                     ; h0
    mov     r8, r9                      ; h1
    jmp     gqa_attn_range
%endif
.ret:
    ret

; broadcast the job in tp_* to all workers, run slice 0 here, wait for all
tp_dispatch:
    push    rbx
    push    r12
    push    r13
    mov     rbx, [rel tp_n_threads]
    mov     r12d, 1
    lock xadd dword [rel tp_gen], r12d
    inc     r12d                        ; this job's generation
    xor     edi, edi
    mov     rsi, rbx
    mov     rcx, [rel tp_M]
    call    tp_slice
    call    tp_run_slice
    lea     rdi, [rel tp_done_arr]
    mov     rcx, 1
.wait_tid:
    cmp     rcx, rbx
    jae     .done
    mov     rdx, rcx
    shl     rdx, 6
.spin:
    cmp     dword [rdi + rdx], r12d
    je      .next
    pause
    jmp     .spin
.next:
    inc     rcx
    jmp     .wait_tid
.done:
    lfence
    pop     r13
    pop     r12
    pop     rbx
    ret

; void tp_matmul_q8_0(W, x, out, M, K)
tp_matmul_q8_0:
%if FAST_Q8
    push    rdi
    push    rsi
    push    rdx
    push    rcx
    push    r8
    mov     rdi, rsi
    mov     rsi, r8
    call    quantize_x_q8_0             ; once, shared by every core
    pop     r8
    pop     rcx
    pop     rdx
    pop     rsi
    pop     rdi
%endif
    cmp     qword [rel tp_n_threads], 1
    jle     .single
    mov     [rel tp_W], rdi
    mov     [rel tp_x], rsi
    mov     [rel tp_out], rdx
    mov     [rel tp_M], rcx
    mov     [rel tp_K], r8
    sub     rsp, 8
    call    tp_dispatch
    add     rsp, 8
    ret
.single:
    mov     r9, rcx                     ; r1 = M
    mov     rcx, r8                     ; K
    xor     r8d, r8d                    ; r0 = 0
    jmp     matmul_q8_0_range

%if FAST_Q8
; void tp_gqa_attn(layer_kv_base, pos, scale_gqa_packed, n_heads)
tp_gqa_attn:
    cmp     qword [rel tp_n_threads], 1
    jle     .single
    mov     [rel tp_W], rdi
    mov     [rel tp_x], rsi
    mov     [rel tp_out], rdx
    mov     [rel tp_M], rcx
    mov     qword [rel tp_K], 0
    sub     rsp, 8
    call    tp_dispatch
    add     rsp, 8
    ret
.single:
    mov     r8, rcx
    xor     ecx, ecx
    jmp     gqa_attn_range
%endif

section .rodata
align 4
zl_mxcsr_default: dd 0x1F80
zl_s_system:      db "system", 10, 0

section .bss
alignb 64
zl_heap_cur:       resq 1
zl_heap_end:       resq 1
zl_heap_base_seen: resq 1
zl_chat_pos:       resq 1
zl_pos:            resq 1
zl_next_tok:       resq 1
zl_n_gen:          resq 1
zl_max_tok:        resq 1
zl_last_unfed:     resq 1
alignb 64
tp_gen:            resd 1
alignb 8
tp_n_threads:      resq 1
tp_W:              resq 1
tp_x:              resq 1
tp_out:            resq 1
tp_M:              resq 1
tp_K:              resq 1
alignb 64
tp_done_arr:       resb ZL_MAX_CPUS * 64
alignb 64
zl_stack:          resb 512 * 1024
zl_stack_top:
