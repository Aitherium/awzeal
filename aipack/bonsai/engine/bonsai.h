/* bonsai.h -- freestanding Ternary Bonsai 2 engine (qwen35; PQ2_0/Q2_0/PTQ1_0/Q1_0 + Hadamard;
 * 27B and 4B shapes).
 *
 * One process-wide engine instance. Every entry point is SysV x86-64; the
 * ZealOS glue (glue2.asm) wraps them in the HolyC ABI. No libc: the engine
 * only touches the model bytes and the arena the caller hands to bz_init.
 *
 * Model memory: either a contiguous GGUF image (starts with "GGUF") or a
 * part table (starts with "BZPARTS1"), which lets a loader keep a GGUF split
 * at tensor boundaries in separately allocated chunks:
 *   u8 magic[8] = "BZPARTS1"; u64 n; struct { u64 ptr; u64 len; } part[n];
 * Part i holds file bytes [sum(len[0..i-1]), +len[i]); part 0 holds the
 * whole GGUF header; no tensor may straddle two parts (split_gguf.py).
 */
#ifndef BONSAI_H
#define BONSAI_H
#include <stdint.h>

#define BZ_TMAX 32               /* tokens per batched forward pass (prefill chunk) */

int64_t bz_init(const uint8_t *model, uint64_t size, uint8_t *arena, uint64_t arena_size, int64_t ctx);
int64_t bz_begin(const uint8_t *prompt, int64_t think);   /* chat turn + prefill -> n_prompt (<0 error) */
int64_t bz_next(uint8_t *buf, int64_t cap);               /* pending token bytes, then one decode step; -1 = done */
int64_t bz_end(void);                                     /* close the turn (<|im_end|>\n) -> ctx pos */
int64_t bz_reset(void);                                   /* forget the conversation */
int64_t bz_stat(int64_t which);                           /* 0 pos 1 ctx 2 arena used 3 vocab 4 n_gen 5 last token 6 arena needed; 100+n max tokens */
int64_t bz_threads(int64_t n);                            /* cores per matmul (caller + n-1 workers) */
int64_t bz_worker(int64_t tid);                           /* worker loop for CPU tid; returns after ~20 ms idle */

/* lower-level entry points (parity harness) */
int64_t bz_tokenize(const uint8_t *text, int64_t len, int32_t *out, int64_t cap);  /* raw text, no specials */
int64_t bz_chat_ids(const uint8_t *prompt, int64_t think, int32_t *out, int64_t cap);
float  *bz_eval(const int32_t *ids, int64_t n, int64_t pos, int64_t want_logits);   /* logits of the last id */
int64_t bz_argmax(const float *logits);
int64_t bz_piece(int64_t id, uint8_t *buf, int64_t cap);
int64_t bz_token_id(const char *text);                    /* exact vocab lookup, -1 if absent */

#endif
