/* bonsai.c -- Ternary-Bonsai-2.5-27B forward pass, freestanding C (AVX2/FMA/F16C).
 *
 * Ground truth is the prism llama.cpp fork (src/models/qwen35.cpp, the ggml CPU ops
 * gated_delta_net / ssm_conv / l2_norm / rope_multi, src/llama-graph.cpp
 * build_hadamard_input + build_embd_rows, src/unicode.cpp qwen35 splitter,
 * src/llama-vocab.cpp BPE). Every place that differs on purpose is marked DIVERGE.
 *
 * Build: -ffreestanding -fPIE -mavx2 -mfma -mf16c -O2 -mno-red-zone -fno-math-errno
 *        -fno-tree-loop-distribute-patterns -fno-jump-tables -fno-stack-protector
 * Nothing here holds an absolute pointer in initialised data, so the flat image
 * needs no relocations (pack_engine.py checks).
 */
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#include "bonsai.h"
#include "unicode_flags.h"

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

#define E_FORMAT   -2
#define E_ARCH     -3
#define E_TENSOR   -4
#define E_OOM      -12
#define E_SHAPE    -5
#define E_PARTS    -6

/* ------------------------------------------------------------------ libc-free basics */

static void bz_memcpy(void *d, const void *s, u64 n) {
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(n) : : "memory");
}
static void bz_memset(void *d, int c, u64 n) {
    __asm__ volatile("rep stosb" : "+D"(d), "+c"(n) : "a"(c) : "memory");
}
static int bz_memeq(const void *a, const void *b, u64 n) {
    const u8 *x = a, *y = b;
    for (u64 i = 0; i < n; i++)
        if (x[i] != y[i]) return 0;
    return 1;
}
static u64 bz_strlen(const char *s) {
    u64 n = 0;
    while (s[n]) n++;
    return n;
}
static int str_is(const u8 *p, u64 n, const char *z) {
    u64 m = bz_strlen(z);
    return m == n && bz_memeq(p, z, n);
}

static float f16_to_f32(u16 h) { return _cvtsh_ss(h); }
static u16 f32_to_f16(float f) { return _cvtss_sh(f, 0); }

/* ------------------------------------------------------------------ math (double inside) */

static double d_exp(double x) {
    if (x > 709.0) return 1.0e308 * 10.0;
    if (x < -745.0) return 0.0;
    const double ln2hi = 6.93147180369123816490e-01, ln2lo = 1.90821492927058770002e-10;
    double kd = __builtin_floor(x * 1.4426950408889634 + 0.5);
    i64 k = (i64)kd;
    double r = (x - kd * ln2hi) - kd * ln2lo;
    /* Taylor to r^13 on |r| <= 0.347: error < 1e-17 */
    double p = 1.0 / 6227020800.0;
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    /* scale by 2^k in two steps so k down to -1074 stays representable */
    i64 k1 = k / 2, k2 = k - k1;
    union { double d; u64 u; } a, b;
    a.u = (u64)(k1 + 1023) << 52;
    b.u = (u64)(k2 + 1023) << 52;
    return p * a.d * b.d;
}

static double d_log(double x) {
    if (!(x > 0.0)) return -1.0e308 * 10.0;
    union { double d; u64 u; } v = { x };
    i64 e = (i64)((v.u >> 52) & 0x7ff) - 1023;
    if (e == -1023) { /* subnormal */
        v.d = x * 4503599627370496.0; /* 2^52 */
        e = (i64)((v.u >> 52) & 0x7ff) - 1023 - 52;
    }
    v.u = (v.u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL; /* m in [1,2) */
    double m = v.d;
    if (m > 1.4142135623730951) { m *= 0.5; e++; }
    double f = (m - 1.0) / (m + 1.0), f2 = f * f;
    double s = 1.0 / 23;
    s = s * f2 + 1.0 / 21; s = s * f2 + 1.0 / 19; s = s * f2 + 1.0 / 17;
    s = s * f2 + 1.0 / 15; s = s * f2 + 1.0 / 13; s = s * f2 + 1.0 / 11;
    s = s * f2 + 1.0 / 9;  s = s * f2 + 1.0 / 7;  s = s * f2 + 1.0 / 5;
    s = s * f2 + 1.0 / 3;  s = s * f2 + 1.0;
    return 2.0 * f * s + (double)e * 0.69314718055994530942;
}

/* sin and cos of x for |x| < 1e6 (rope angles stay far below that) */
static void d_sincos(double x, double *sn, double *cs) {
    const double p1 = 1.57079632673412561417e+00, p2 = 6.07710050650619224932e-11,
                 p3 = 2.02226624879595063154e-21;
    double kd = __builtin_floor(x * 0.63661977236758134308 + 0.5);
    double r = ((x - kd * p1) - kd * p2) - kd * p3;
    double r2 = r * r;
    double s = -1.0 / 1307674368000.0;                 /* sin Taylor to r^17 */
    s = s * r2 + 1.0 / 6227020800.0; s = s * r2 - 1.0 / 39916800.0;
    s = s * r2 + 1.0 / 362880.0;     s = s * r2 - 1.0 / 5040.0;
    s = s * r2 + 1.0 / 120.0;        s = s * r2 - 1.0 / 6.0;
    s = r + r * r2 * s;
    double c = 1.0 / 20922789888000.0;                 /* cos Taylor to r^16 */
    c = c * r2 - 1.0 / 87178291200.0; c = c * r2 + 1.0 / 479001600.0;
    c = c * r2 - 1.0 / 3628800.0;     c = c * r2 + 1.0 / 40320.0;
    c = c * r2 - 1.0 / 720.0;         c = c * r2 + 1.0 / 24.0;
    c = c * r2 - 0.5;                 c = 1.0 + r2 * c;
    i64 q = ((i64)kd) & 3;
    switch (q) {
    case 0: *sn = s;  *cs = c;  break;
    case 1: *sn = c;  *cs = -s; break;
    case 2: *sn = -s; *cs = -c; break;
    default: *sn = -c; *cs = s; break;
    }
}

static float f_exp(float x) { return (float)d_exp((double)x); }
static float f_sqrt(float x) { return __builtin_sqrtf(x); }
static float f_silu(float x) { return x / (1.0f + f_exp(-x)); }
static float f_sigmoid(float x) { return 1.0f / (1.0f + f_exp(-x)); }
static float f_softplus(float x) { return x > 20.0f ? x : (float)d_log(1.0 + (double)f_exp(x)); }
/* llama.cpp quantizers use roundf: nearest, ties away from zero */
static int f_round(float x) {
    float t = __builtin_truncf(x);
    float d = x - t;
    if (d >= 0.5f) t += 1.0f;
    else if (d <= -0.5f) t -= 1.0f;
    return (int)t;
}

/* ------------------------------------------------------------------ arena */

static u8 *g_ar_base, *g_ar_cur, *g_ar_end;

static void *ar_alloc(u64 n) {
    u8 *p = (u8 *)(((u64)g_ar_cur + 63) & ~(u64)63);
    n = (n + 63) & ~(u64)63;
    if (p + n > g_ar_end || p + n < p) return 0;
    g_ar_cur = p + n;
    bz_memset(p, 0, n);
    return p;
}

/* ------------------------------------------------------------------ model description */

#define TYPE_F32   0
#define TYPE_PQ2_0 142
#define PQ_BLK     34      /* f16 d + 32 bytes of 2-bit codes, 128 weights */

typedef struct {
    const u8 *data;
    i64 ne0, ne1;
    u32 type;
    int folded;            /* listed in prism.hadamard.weight_names */
} wt_t;

typedef struct {
    int rec;               /* Gated DeltaNet layer (else full attention) */
    int slot;              /* index into the KV cache (attn) or the recurrent state (rec) */
    const float *attn_norm, *post_norm;
    wt_t wq, wk, wv, wo;
    const float *q_norm, *k_norm;
    wt_t wqkv, wgate, wout;
    const float *alpha, *beta, *conv, *dt, *ssm_a, *ssm_norm;
    wt_t fgate, fup, fdown;
} layer_t;

#define MAX_LAYERS 64

typedef struct {
    /* hparams */
    i64 n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim, n_rot;
    i64 conv_k, d_state, n_group, dt_rank, d_inner, fa_interval;
    float rope_base, eps;
    i32 sections[4];
    int gdn_grouped;
    i64 had_block;
    /* derived */
    i64 n_attn, n_rec, key_dim, conv_ch, v_dim;
    /* weights */
    wt_t tok_embd, output;
    const float *out_norm;
    int embd_inverse;
    layer_t L[MAX_LAYERS];
    float *sign5120, *sign6144, *sign17408;
    /* vocab */
    i64 n_vocab, n_merges;
    const u8 **tok_txt;
    u32 *tok_len;
    u8 *tok_type;
    const u8 **mrg_txt;
    u32 *mrg_len;
    u32 *vh, *mh;          /* open-addressing hash tables: id+1 / rank+1 */
    u32 vh_mask, mh_mask;
    i32 eos, eot, im_start, im_end, think, think_end, nl;
    u8 b2u_len[256];
    u8 b2u[256][2];        /* gpt2 bytes_to_unicode as UTF-8 */
    i16 u2b[512];          /* codepoint -> byte (-1 = none) */
} model_t;

static model_t M;

/* runtime state */
static i64 g_ctx;
static u16 *g_kcache, *g_vcache;   /* [n_attn][ctx][n_head_kv*head_dim] f16 */
static float *g_ssm;               /* [n_rec][dt_rank][d_state][d_state], s[j*S+i] = S[i][j] */
static float *g_conv;              /* [n_rec][conv_k-1][conv_ch] oldest first */
static float *g_logits;
/* batch work buffers, [BZ_TMAX][width] */
static float *b_x, *b_xn, *b_h, *b_big, *b_z, *b_g, *b_u, *b_o, *b_k, *b_v, *b_al, *b_be, *b_att;
static void *b_q8;
static float *w_scores, *w_conv;
static i32 *g_ids;
static u32 *g_cpt;
static u8 *g_word;
static i32 *g_sym;

#define MAX_PROMPT_TOKENS 4096
#define MAX_PROMPT_BYTES  16384

/* session */
static i64 s_pos, s_chat_pos, s_next, s_ngen, s_maxtok = 256, s_need;

/* ------------------------------------------------------------------ GGUF */

typedef struct {
    const u8 *p, *end;
    int bad;
} rd_t;

static u64 rd_u(rd_t *r, int n) {
    if (r->p + n > r->end) { r->bad = 1; return 0; }
    u64 v = 0;
    for (int i = 0; i < n; i++) v |= (u64)r->p[i] << (8 * i);
    r->p += n;
    return v;
}
static const u8 *rd_str(rd_t *r, u64 *len) {
    *len = rd_u(r, 8);
    const u8 *s = r->p;
    if (r->p + *len > r->end) { r->bad = 1; return s; }
    r->p += *len;
    return s;
}
static int type_size(u32 t) {
    switch (t) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
}
static void rd_skip(rd_t *r, u32 t) {
    if (t == 8) { u64 n; rd_str(r, &n); return; }
    if (t == 9) {
        u32 et = (u32)rd_u(r, 4);
        u64 n = rd_u(r, 8);
        if (et == 8) { for (u64 i = 0; i < n && !r->bad; i++) { u64 l; rd_str(r, &l); } }
        else { int s = type_size(et); if (!s) r->bad = 1; else if (r->p + n * s > r->end) r->bad = 1; else r->p += n * s; }
        return;
    }
    int s = type_size(t);
    if (!s) { r->bad = 1; return; }
    r->p += s;
}
static i64 rd_int(rd_t *r, u32 t) {
    switch (t) {
    case 0: return (u8)rd_u(r, 1);
    case 1: return (i8)rd_u(r, 1);
    case 2: return (u16)rd_u(r, 2);
    case 3: return (int16_t)rd_u(r, 2);
    case 4: return (u32)rd_u(r, 4);
    case 5: return (i32)rd_u(r, 4);
    case 7: return (u8)rd_u(r, 1);
    case 10: case 11: return (i64)rd_u(r, 8);
    default: rd_skip(r, t); return 0;
    }
}
static float rd_f32(rd_t *r, u32 t) {
    if (t == 6) { union { u32 u; float f; } v; v.u = (u32)rd_u(r, 4); return v.f; }
    if (t == 12) { union { u64 u; double d; } v; v.u = rd_u(r, 8); return (float)v.d; }
    return (float)rd_int(r, t);
}

/* tensor table */
#define MAX_TENSORS 1024
typedef struct {
    const u8 *name;
    u64 nlen;
    u32 type, nd;
    u64 ne[4];
    u64 off;
    int folded;
} tinfo_t;
static tinfo_t *g_ti;
static i64 g_nt;

/* model parts */
#define MAX_PARTS 16
static const u8 *g_part_ptr[MAX_PARTS];
static u64 g_part_off[MAX_PARTS + 1];
static i64 g_nparts;

static const u8 *file_ptr(u64 off, u64 n) {
    for (i64 i = 0; i < g_nparts; i++)
        if (off >= g_part_off[i] && off + n <= g_part_off[i + 1])
            return g_part_ptr[i] + (off - g_part_off[i]);
    return 0;
}

static u64 g_data_start;

static tinfo_t *find_t(const char *name) {
    u64 n = bz_strlen(name);
    for (i64 i = 0; i < g_nt; i++)
        if (g_ti[i].nlen == n && bz_memeq(g_ti[i].name, name, n)) return &g_ti[i];
    return 0;
}

static u64 t_bytes(const tinfo_t *t) {
    u64 n = 1;
    for (u32 i = 0; i < t->nd; i++) n *= t->ne[i];
    if (t->type == TYPE_F32) return n * 4;
    if (t->type == TYPE_PQ2_0) return n / 128 * PQ_BLK;
    return 0;
}

static char g_nbuf[96];
static const char *tname(int il, const char *suffix) {
    /* "blk.<il>.<suffix>" */
    int k = 0;
    const char *b = "blk.";
    while (*b) g_nbuf[k++] = *b++;
    char d[4];
    int nd = 0;
    int v = il;
    do { d[nd++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (nd) g_nbuf[k++] = d[--nd];
    g_nbuf[k++] = '.';
    while (*suffix && k < 94) g_nbuf[k++] = *suffix++;
    g_nbuf[k] = 0;
    return g_nbuf;
}

static int g_err;

static const float *get_f32(const char *name, i64 n) {
    tinfo_t *t = find_t(name);
    if (!t || t->type != TYPE_F32) { g_err = E_TENSOR; return 0; }
    u64 cnt = 1;
    for (u32 i = 0; i < t->nd; i++) cnt *= t->ne[i];
    if ((i64)cnt != n) { g_err = E_SHAPE; return 0; }
    const u8 *p = file_ptr(g_data_start + t->off, cnt * 4);
    if (!p) g_err = E_PARTS;
    return (const float *)p;
}

static wt_t get_w(const char *name, i64 ne0, i64 ne1) {
    wt_t w = { 0, 0, 0, 0, 0 };
    tinfo_t *t = find_t(name);
    if (!t) { g_err = E_TENSOR; return w; }
    if (t->nd != 2 || (i64)t->ne[0] != ne0 || (i64)t->ne[1] != ne1) { g_err = E_SHAPE; return w; }
    if (t->type != TYPE_PQ2_0 && t->type != TYPE_F32) { g_err = E_FORMAT; return w; }
    w.data = file_ptr(g_data_start + t->off, t_bytes(t));
    if (!w.data) g_err = E_PARTS;
    w.ne0 = ne0; w.ne1 = ne1; w.type = t->type; w.folded = t->folded;
    return w;
}

/* ------------------------------------------------------------------ hashing (vocab, merges) */

static u64 h_init(void) { return 1469598103934665603ULL; }
static u64 h_add(u64 h, const u8 *p, u64 n) {
    for (u64 i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static i64 vocab_find(const u8 *s, u64 n) {
    u64 h = h_add(h_init(), s, n);
    for (u32 i = (u32)h & M.vh_mask;; i = (i + 1) & M.vh_mask) {
        u32 e = M.vh[i];
        if (!e) return -1;
        e--;
        if (M.tok_len[e] == n && bz_memeq(M.tok_txt[e], s, n)) return e;
    }
}

/* rank of merge "a b", -1 if none */
static i64 merge_rank(const u8 *a, u64 na, const u8 *b, u64 nb) {
    u64 h = h_add(h_init(), a, na);
    u8 sp = ' ';
    h = h_add(h, &sp, 1);
    h = h_add(h, b, nb);
    for (u32 i = (u32)h & M.mh_mask;; i = (i + 1) & M.mh_mask) {
        u32 e = M.mh[i];
        if (!e) return -1;
        e--;
        if (M.mrg_len[e] == na + 1 + nb && bz_memeq(M.mrg_txt[e], a, na) && M.mrg_txt[e][na] == ' ' &&
            bz_memeq(M.mrg_txt[e] + na + 1, b, nb))
            return e;
    }
}

/* ------------------------------------------------------------------ kernels */

/* one 128-wide activation block, Q8_0 per 32 (llama.cpp quantize_row_q8_0 arithmetic),
 * stored in the order the 2-bit weight unpack produces: q[s*32 + b] = x[4b + s] */
typedef struct {
    i8 q[128];
    float sv[8];           /* per 32-lane-pair scale: d of sub-block lane/2 */
    float cv[8];           /* d * sum(q) / 2 per lane: removes the +1 weight code bias */
} __attribute__((aligned(64))) qblk_t;

static void quant_row(const float *x, i64 K, qblk_t *out) {
    for (i64 b = 0; b < K / 128; b++) {
        qblk_t *o = &out[b];
        const float *xb = x + b * 128;
        for (int k = 0; k < 4; k++) {
            const float *xs = xb + k * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = xs[j] < 0 ? -xs[j] : xs[j];
                if (a > amax) amax = a;
            }
            float d = amax / 127.0f;
            float id = d != 0.0f ? 1.0f / d : 0.0f;
            int sum = 0;
            for (int j = 0; j < 32; j++) {
                int q = f_round(xs[j] * id);
                int e = k * 32 + j;
                o->q[(e & 3) * 32 + (e >> 2)] = (i8)q;
                sum += q;
            }
            float dh = f16_to_f32(f32_to_f16(d));   /* block_q8_0 keeps d as ggml_half */
            o->sv[2 * k] = o->sv[2 * k + 1] = dh;
            o->cv[2 * k] = o->cv[2 * k + 1] = dh * (float)sum * 0.5f;
        }
    }
}

static float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

/* out[t*ostride + r] = W[r] . x[t]  for r < M rows, t < T tokens; W is PQ2_0 [K, M] */
static void mm_pq2(const wt_t *w, const qblk_t *xq, i64 xstride, i64 T, float *out, i64 ostride) {
    const i64 nb = w->ne0 / 128, Mr = w->ne1;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    for (i64 r = 0; r < Mr; r++) {
        const u8 *wr = w->data + r * nb * PQ_BLK;
        for (i64 t0 = 0; t0 < T; t0 += 4) {
            i64 nt = T - t0 < 4 ? T - t0 : 4;
            __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
            const qblk_t *x0 = xq + t0 * xstride;
            for (i64 b = 0; b < nb; b++) {
                const u8 *wb = wr + b * PQ_BLK;
                u16 dh = (u16)(wb[0] | (wb[1] << 8));
                __m256 dw = _mm256_set1_ps(f16_to_f32(dh));
                __m256i v = _mm256_loadu_si256((const __m256i *)(wb + 2));
                __m256i w0 = _mm256_and_si256(v, m3);
                __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(v, 2), m3);
                __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(v, 4), m3);
                __m256i w3 = _mm256_and_si256(_mm256_srli_epi16(v, 6), m3);
#define PQ_DOT(I, ACC)                                                                              \
                {                                                                                   \
                    const qblk_t *xb = x0 + (I) * xstride + b;                                      \
                    __m256i p = _mm256_maddubs_epi16(w0, _mm256_load_si256((const __m256i *)xb->q)); \
                    p = _mm256_add_epi16(p, _mm256_maddubs_epi16(w1, _mm256_load_si256((const __m256i *)(xb->q + 32)))); \
                    p = _mm256_add_epi16(p, _mm256_maddubs_epi16(w2, _mm256_load_si256((const __m256i *)(xb->q + 64)))); \
                    p = _mm256_add_epi16(p, _mm256_maddubs_epi16(w3, _mm256_load_si256((const __m256i *)(xb->q + 96)))); \
                    __m256 f = _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones));                       \
                    f = _mm256_fmsub_ps(f, _mm256_load_ps(xb->sv), _mm256_load_ps(xb->cv));          \
                    ACC = _mm256_fmadd_ps(dw, f, ACC);                                              \
                }
                PQ_DOT(0, acc0);
                if (nt > 1) PQ_DOT(1, acc1);
                if (nt > 2) PQ_DOT(2, acc2);
                if (nt > 3) PQ_DOT(3, acc3);
            }
            out[(t0 + 0) * ostride + r] = hsum8(acc0);
            if (nt > 1) out[(t0 + 1) * ostride + r] = hsum8(acc1);
            if (nt > 2) out[(t0 + 2) * ostride + r] = hsum8(acc2);
            if (nt > 3) out[(t0 + 3) * ostride + r] = hsum8(acc3);
        }
    }
}

/* f32 weight [K, M] times f32 activations (llama.cpp vec_dot_f32; DIVERGE: our summation order) */
static void mm_f32(const float *W, i64 K, i64 Mr, const float *x, i64 xstride, i64 T, float *out, i64 ostride) {
    for (i64 t = 0; t < T; t++) {
        const float *xt = x + t * xstride;
        for (i64 r = 0; r < Mr; r++) {
            const float *wr = W + r * K;
            __m256 a = _mm256_setzero_ps();
            i64 i = 0;
            for (; i + 8 <= K; i += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(wr + i), _mm256_loadu_ps(xt + i), a);
            float s = hsum8(a);
            for (; i < K; i++) s += wr[i] * xt[i];
            out[t * ostride + r] = s;
        }
    }
}

/* normalised Sylvester-Walsh-Hadamard on 1024 floats (ggml_compute_forward_fwht: pre-scale, then butterflies) */
static void fwht1024(float *x) {
    const __m256 sc = _mm256_set1_ps(1.0f / 32.0f);
    for (int i = 0; i < 1024; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), sc));
    for (int len = 1; len < 8; len <<= 1)
        for (int i = 0; i < 1024; i += 2 * len)
            for (int j = 0; j < len; j++) {
                float u = x[i + j], v = x[i + j + len];
                x[i + j] = u + v;
                x[i + j + len] = u - v;
            }
    for (int len = 8; len < 1024; len <<= 1)
        for (int i = 0; i < 1024; i += 2 * len)
            for (int j = 0; j < len; j += 8) {
                __m256 u = _mm256_loadu_ps(x + i + j), v = _mm256_loadu_ps(x + i + j + len);
                _mm256_storeu_ps(x + i + j, _mm256_add_ps(u, v));
                _mm256_storeu_ps(x + i + j + len, _mm256_sub_ps(u, v));
            }
}

static const float *sign_for(i64 width) {
    if (width == 5120) return M.sign5120;
    if (width == 6144) return M.sign6144;
    if (width == 17408) return M.sign17408;
    return 0;
}

/* activation for a matmul: (sign + blockwise WHT when folded), then Q8 blocks */
static void prep_in(const float *x, i64 xstride, i64 T, i64 K, int folded, float *tmp) {
    qblk_t *q = (qblk_t *)b_q8;
    i64 nb = K / 128;
    const float *s = sign_for(K);
    for (i64 t = 0; t < T; t++) {
        const float *xt = x + t * xstride;
        if (folded) {
            for (i64 i = 0; i < K; i++) tmp[i] = xt[i] * s[i];
            for (i64 b = 0; b < K; b += 1024) fwht1024(tmp + b);
            quant_row(tmp, K, q + t * nb);
        } else {
            quant_row(xt, K, q + t * nb);
        }
    }
}

static void matmul(const wt_t *w, i64 T, float *out, i64 ostride) {
    mm_pq2(w, (const qblk_t *)b_q8, w->ne0 / 128, T, out, ostride);
}

/* ggml rms_norm (sum of float squares in double) followed by the weight multiply */
static void rmsnorm(float *y, const float *x, const float *w, i64 n, float eps) {
    double sum = 0.0;
    for (i64 i = 0; i < n; i++) sum += (double)(x[i] * x[i]);
    float mean = (float)(sum / (double)n);
    float scale = 1.0f / f_sqrt(mean + eps);
    for (i64 i = 0; i < n; i++) y[i] = (x[i] * scale) * w[i];
}

static void l2norm(float *x, i64 n, float eps) {
    double sum = 0.0;
    for (i64 i = 0; i < n; i++) sum += (double)(x[i] * x[i]);
    float nr = f_sqrt((float)sum);
    float scale = 1.0f / (nr > eps ? nr : eps);
    for (i64 i = 0; i < n; i++) x[i] *= scale;
}

/* IMROPE with text positions (all sections share pos): neox pairs (i, i+n_rot/2) on the first n_rot dims */
static void rope(float *x, i64 pos) {
    const i64 nr = M.n_rot, half = nr / 2;
    float theta_scale = (float)d_exp(d_log((double)M.rope_base) * (-2.0 / (double)nr));
    float theta = (float)pos;
    for (i64 i = 0; i < half; i++) {
        double sn, cs;
        d_sincos((double)theta, &sn, &cs);
        float c = (float)cs, s = (float)sn;
        float x0 = x[i], x1 = x[i + half];
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

/* ------------------------------------------------------------------ forward */

static void embed(i32 id, float *x) {
    const wt_t *w = &M.tok_embd;
    const i64 nb = w->ne0 / 128;
    const u8 *row = w->data + (i64)id * nb * PQ_BLK;
    for (i64 b = 0; b < nb; b++) {
        const u8 *wb = row + b * PQ_BLK;
        float d = f16_to_f32((u16)(wb[0] | (wb[1] << 8)));
        for (int j = 0; j < 128; j++) {
            int q = (wb[2 + j / 4] >> ((j % 4) * 2)) & 3;
            x[b * 128 + j] = (float)(q - 1) * d;
        }
    }
    if (M.embd_inverse) { /* h = s * (H z) */
        for (i64 b = 0; b < w->ne0; b += 1024) fwht1024(x + b);
        for (i64 i = 0; i < w->ne0; i++) x[i] *= M.sign5120[i];
    }
}

static void attn_layer(const layer_t *L, i64 T, i64 pos) {
    const i64 E = M.n_embd, H = M.n_head, HK = M.n_head_kv, D = M.head_dim;
    const i64 QW = H * D * 2, KW = HK * D, OW = H * D;
    prep_in(b_xn, E, T, E, L->wq.folded, b_h);
    matmul(&L->wq, T, b_big, QW);
    if (L->wk.folded != L->wq.folded) prep_in(b_xn, E, T, E, L->wk.folded, b_h);
    matmul(&L->wk, T, b_k, KW);
    matmul(&L->wv, T, b_v, KW);
    u16 *kc = g_kcache + (i64)L->slot * g_ctx * KW;
    u16 *vc = g_vcache + (i64)L->slot * g_ctx * KW;
    for (i64 t = 0; t < T; t++) {
        float *qf = b_big + t * QW;
        for (i64 h = 0; h < H; h++) {
            float *q = qf + h * 2 * D;
            rmsnorm(q, q, L->q_norm, D, M.eps);
            rope(q, pos + t);
        }
        float *k = b_k + t * KW, *v = b_v + t * KW;
        for (i64 h = 0; h < HK; h++) {
            rmsnorm(k + h * D, k + h * D, L->k_norm, D, M.eps);
            rope(k + h * D, pos + t);
        }
        u16 *kd = kc + (pos + t) * KW, *vd = vc + (pos + t) * KW;
        for (i64 i = 0; i < KW; i += 8) {
            _mm_storeu_si128((__m128i *)(kd + i), _mm256_cvtps_ph(_mm256_loadu_ps(k + i), 0));
            _mm_storeu_si128((__m128i *)(vd + i), _mm256_cvtps_ph(_mm256_loadu_ps(v + i), 0));
        }
    }
    const float kq_scale = 1.0f / f_sqrt((float)D);
    const i64 grp = H / HK;
    for (i64 t = 0; t < T; t++) {
        const i64 n = pos + t + 1;
        float *qf = b_big + t * QW;
        float *o = b_o + t * OW;
        for (i64 h = 0; h < H; h++) {
            const float *q = qf + h * 2 * D;
            const float *gate = q + D;
            const i64 kh = h / grp;
            float mx = -3.0e38f;
            for (i64 p = 0; p < n; p++) {
                const u16 *kr = kc + p * KW + kh * D;
                __m256 a = _mm256_setzero_ps();
                for (i64 i = 0; i < D; i += 8)
                    a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kr + i))), _mm256_loadu_ps(q + i), a);
                float s = hsum8(a) * kq_scale;
                w_scores[p] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (i64 p = 0; p < n; p++) { w_scores[p] = f_exp(w_scores[p] - mx); sum += w_scores[p]; }
            float inv = 1.0f / sum;
            float *oh = o + h * D;
            for (i64 i = 0; i < D; i++) oh[i] = 0.0f;
            for (i64 p = 0; p < n; p++) {
                const u16 *vr = vc + p * KW + kh * D;
                __m256 wp = _mm256_set1_ps(w_scores[p] * inv);
                for (i64 i = 0; i < D; i += 8)
                    _mm256_storeu_ps(oh + i, _mm256_fmadd_ps(wp, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vr + i))), _mm256_loadu_ps(oh + i)));
            }
            for (i64 i = 0; i < D; i++) oh[i] *= f_sigmoid(gate[i]);
        }
    }
    prep_in(b_o, OW, T, OW, L->wo.folded, b_h);
    matmul(&L->wo, T, b_att, E);
}

static void rec_layer(const layer_t *L, i64 T) {
    const i64 E = M.n_embd, S = M.d_state, HV = M.dt_rank, HKk = M.n_group;
    const i64 CH = M.conv_ch, VD = M.v_dim, KD = M.key_dim, CK = M.conv_k;
    prep_in(b_xn, E, T, E, L->wqkv.folded, b_h);
    matmul(&L->wqkv, T, b_big, CH);
    if (L->wgate.folded != L->wqkv.folded) prep_in(b_xn, E, T, E, L->wgate.folded, b_h);
    matmul(&L->wgate, T, b_z, VD);
    mm_f32(L->alpha, E, HV, b_xn, E, T, b_al, HV);
    mm_f32(L->beta, E, HV, b_xn, E, T, b_be, HV);
    float *cs = g_conv + (i64)L->slot * (CK - 1) * CH;
    float *st = g_ssm + (i64)L->slot * HV * S * S;
    const float oscale = 1.0f / f_sqrt((float)S);
    for (i64 t = 0; t < T; t++) {
        float *in = b_big + t * CH;
        /* causal conv1d over [state(CK-1) | in], then SiLU */
        for (i64 c = 0; c < CH; c++) {
            const float *w = L->conv + c * CK;
            float s = 0.0f;
            for (i64 k = 0; k < CK - 1; k++) s += cs[k * CH + c] * w[k];
            s += in[c] * w[CK - 1];
            w_conv[c] = f_silu(s);
        }
        for (i64 k = 0; k + 1 < CK - 1; k++) bz_memcpy(cs + k * CH, cs + (k + 1) * CH, CH * 4);
        bz_memcpy(cs + (CK - 2) * CH, in, CH * 4);
        float *qv = w_conv, *kv = w_conv + KD, *vv = w_conv + 2 * KD;
        for (i64 h = 0; h < 2 * HKk; h++) l2norm(w_conv + h * S, S, M.eps);
        float *o = b_o + t * VD;
        for (i64 h = 0; h < HV; h++) {
            const float *q = qv + (h % HKk) * S, *k = kv + (h % HKk) * S, *v = vv + h * S;
            float beta = f_sigmoid(b_be[t * HV + h]);
            float g = f_softplus(b_al[t * HV + h] + L->dt[h]) * L->ssm_a[h];
            float eg = f_exp(g);
            float *sh = st + h * S * S;
            float *oh = o + h * S;
            const __m256 vg = _mm256_set1_ps(eg);
            for (i64 j = 0; j < S; j++) {
                float *row = sh + j * S;
                __m256 a = _mm256_setzero_ps();
                for (i64 i = 0; i < S; i += 8) {
                    __m256 r = _mm256_mul_ps(_mm256_loadu_ps(row + i), vg);
                    _mm256_storeu_ps(row + i, r);
                    a = _mm256_fmadd_ps(r, _mm256_loadu_ps(k + i), a);
                }
                float delta = (v[j] - hsum8(a)) * beta;
                __m256 vd = _mm256_set1_ps(delta);
                __m256 b = _mm256_setzero_ps();
                for (i64 i = 0; i < S; i += 8) {
                    __m256 r = _mm256_fmadd_ps(_mm256_loadu_ps(k + i), vd, _mm256_loadu_ps(row + i));
                    _mm256_storeu_ps(row + i, r);
                    b = _mm256_fmadd_ps(r, _mm256_loadu_ps(q + i), b);
                }
                oh[j] = hsum8(b) * oscale;
            }
            /* gated norm: rms(o_h) * w * silu(z_h) */
            rmsnorm(oh, oh, L->ssm_norm, S, M.eps);
            const float *z = b_z + t * VD + h * S;
            for (i64 j = 0; j < S; j++) oh[j] = f_silu(z[j]) * oh[j];
        }
        if (M.gdn_grouped) { /* tiled [hd, nk, rep] -> grouped [hd, rep, nk] for ssm_out */
            const i64 rep = HV / HKk;
            float *p = b_h; /* scratch, VD floats */
            for (i64 nk = 0; nk < HKk; nk++)
                for (i64 r = 0; r < rep; r++)
                    bz_memcpy(p + S * (r + rep * nk), o + S * (nk + HKk * r), S * 4);
            bz_memcpy(o, p, VD * 4);
        }
    }
    prep_in(b_o, VD, T, VD, L->wout.folded, b_h);
    matmul(&L->wout, T, b_att, E);
}

static void ffn(const layer_t *L, i64 T) {
    const i64 E = M.n_embd, F = M.n_ff;
    prep_in(b_xn, E, T, E, L->fgate.folded, b_h);
    matmul(&L->fgate, T, b_g, F);
    if (L->fup.folded != L->fgate.folded) prep_in(b_xn, E, T, E, L->fup.folded, b_h);
    matmul(&L->fup, T, b_u, F);
    for (i64 i = 0; i < T * F; i++) b_g[i] = f_silu(b_g[i]) * b_u[i];
    prep_in(b_g, F, T, F, L->fdown.folded, b_u);
    matmul(&L->fdown, T, b_att, E);
}

static void forward_chunk(const i32 *ids, i64 T, i64 pos, int want_logits) {
    const i64 E = M.n_embd;
    for (i64 t = 0; t < T; t++) embed(ids[t], b_x + t * E);
    for (i64 il = 0; il < M.n_layer; il++) {
        const layer_t *L = &M.L[il];
        for (i64 t = 0; t < T; t++) rmsnorm(b_xn + t * E, b_x + t * E, L->attn_norm, E, M.eps);
        if (L->rec) rec_layer(L, T);
        else attn_layer(L, T, pos);
        for (i64 i = 0; i < T * E; i++) b_x[i] += b_att[i];
        for (i64 t = 0; t < T; t++) rmsnorm(b_xn + t * E, b_x + t * E, L->post_norm, E, M.eps);
        ffn(L, T);
        for (i64 i = 0; i < T * E; i++) b_x[i] += b_att[i];
    }
    if (want_logits) {
        rmsnorm(b_xn, b_x + (T - 1) * E, M.out_norm, E, M.eps);
        prep_in(b_xn, E, 1, E, M.output.folded, b_h);
        matmul(&M.output, 1, g_logits, M.n_vocab);
    }
}

float *bz_eval(const i32 *ids, i64 n, i64 pos, i64 want_logits) {
    if (n <= 0 || pos + n > g_ctx) return 0;
    for (i64 i = 0; i < n; i += BZ_TMAX) {
        i64 T = n - i < BZ_TMAX ? n - i : BZ_TMAX;
        forward_chunk(ids + i, T, pos + i, want_logits && i + T == n);
    }
    return g_logits;
}

i64 bz_argmax(const float *l) {
    i64 best = 0;
    for (i64 i = 1; i < M.n_vocab; i++)
        if (l[i] > l[best]) best = i;
    return best;
}

/* ------------------------------------------------------------------ tokenizer */

static u32 cpt_flags(u32 c) {
    /* UZ bits: 1 N, 2 L, 4 M, 8 whitespace, 16 any */
    i64 lo = 0, hi = UZ_N - 1;
    if (c >= 0x110000) return 0;
    while (lo < hi) {
        i64 mid = (lo + hi + 1) / 2;
        if (uz_start[mid] <= c) lo = mid;
        else hi = mid - 1;
    }
    return uz_flags[lo];
}
#define FN 1
#define FL 2
#define FM 4
#define FW 8
#define FA 16

static i64 utf8_decode(const u8 *s, i64 n, u32 *out, i64 cap) {
    i64 i = 0, k = 0;
    while (i < n && k < cap) {
        u8 c = s[i];
        u32 cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { cp = c; len = 1; }  /* DIVERGE: llama.cpp throws on invalid UTF-8 */
        if (i + len > n) { cp = c; len = 1; }
        for (int j = 1; j < len; j++) cp = (cp << 6) | (s[i + j] & 0x3F);
        out[k++] = cp;
        i += len;
    }
    return k;
}

static int utf8_encode(u32 cp, u8 *o) {
    if (cp < 0x80) { o[0] = (u8)cp; return 1; }
    if (cp < 0x800) { o[0] = (u8)(0xC0 | (cp >> 6)); o[1] = (u8)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { o[0] = (u8)(0xE0 | (cp >> 12)); o[1] = (u8)(0x80 | ((cp >> 6) & 0x3F)); o[2] = (u8)(0x80 | (cp & 0x3F)); return 3; }
    o[0] = (u8)(0xF0 | (cp >> 18)); o[1] = (u8)(0x80 | ((cp >> 12) & 0x3F)); o[2] = (u8)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (u8)(0x80 | (cp & 0x3F));
    return 4;
}

/* BPE one pre-tokenized word (codepoints cp[a..b)); appends token ids */
static i64 bpe_word(const u32 *cp, i64 a, i64 b, i32 *out, i64 k, i64 cap) {
    /* byte-encode the word's UTF-8 into the gpt2 mapped alphabet */
    i64 wl = 0;
    for (i64 i = a; i < b; i++) {
        u8 tmp[4];
        int n = utf8_encode(cp[i], tmp);
        for (int j = 0; j < n; j++) {
            u8 by = tmp[j];
            for (int z = 0; z < M.b2u_len[by]; z++) {
                if (wl < MAX_PROMPT_BYTES * 2) g_word[wl++] = M.b2u[by][z];
            }
        }
    }
    /* symbols: start offsets of each UTF-8 char; g_sym[2*i] = off, [2*i+1] = len */
    i64 ns = 0;
    for (i64 i = 0; i < wl;) {
        u8 c = g_word[i];
        int len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
        if (i + len > wl) len = (int)(wl - i);
        g_sym[2 * ns] = (i32)i;
        g_sym[2 * ns + 1] = len;
        ns++;
        i += len;
    }
    /* merge the lowest-rank adjacent pair (leftmost on ties) until none merges */
    for (;;) {
        i64 best = -1, best_rank = -1;
        for (i64 i = 0; i + 1 < ns; i++) {
            i64 r = merge_rank(g_word + g_sym[2 * i], (u64)g_sym[2 * i + 1], g_word + g_sym[2 * i + 2], (u64)g_sym[2 * i + 3]);
            if (r >= 0 && (best < 0 || r < best_rank)) { best = i; best_rank = r; }
        }
        if (best < 0) break;
        g_sym[2 * best + 1] += g_sym[2 * best + 3];
        for (i64 i = best + 1; i + 1 < ns; i++) { g_sym[2 * i] = g_sym[2 * i + 2]; g_sym[2 * i + 1] = g_sym[2 * i + 3]; }
        ns--;
    }
    for (i64 i = 0; i < ns && k < cap; i++) {
        i64 id = vocab_find(g_word + g_sym[2 * i], (u64)g_sym[2 * i + 1]);
        if (id >= 0) { out[k++] = (i32)id; continue; }
        /* llama.cpp fallback: one token per byte of the symbol */
        for (i32 j = 0; j < g_sym[2 * i + 1] && k < cap; j++) {
            i64 bid = vocab_find(g_word + g_sym[2 * i] + j, 1);
            if (bid >= 0) out[k++] = (i32)bid;
        }
    }
    return k;
}

/* qwen35 pre-tokenizer (unicode_regex_split_custom_qwen35), then BPE */
i64 bz_tokenize(const u8 *text, i64 len, i32 *out, i64 cap) {
    if (len > MAX_PROMPT_BYTES) len = MAX_PROMPT_BYTES;
    const i64 n = utf8_decode(text, len, g_cpt, MAX_PROMPT_BYTES);
    const u32 OOR = 0xFFFFFFFFu;
#define CPT(p) ((p) < n ? g_cpt[p] : OOR)
#define FLG(p) ((p) < n ? cpt_flags(g_cpt[p]) : 0u)
    i64 k = 0, prev = 0, pos = 0;
#define ADD(e) do { if ((e) > prev) k = bpe_word(g_cpt, prev, (e), out, k, cap); prev = (e); } while (0)
    while (pos < n) {
        u32 c = CPT(pos);
        u32 f = FLG(pos);
        if (c == '\'' && pos + 1 < n) {
            u32 c1 = CPT(pos + 1);
            if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { ADD(pos + 2); pos += 2; continue; }
            if (pos + 2 < n) {
                u32 c2 = CPT(pos + 2);
                if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) { ADD(pos + 3); pos += 3; continue; }
            }
        }
        /* [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ */
        if (!(c == '\r' || c == '\n' || (f & FN))) {
            u32 f1 = FLG(pos + 1);
            if ((f & (FL | FM)) || (f1 & (FL | FM))) {
                pos++;
                while (FLG(pos) & (FL | FM)) pos++;
                ADD(pos);
                continue;
            }
        }
        /* \p{N} */
        if (f & FN) { pos++; ADD(pos); continue; }
        /* <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]* */
        u32 f2 = c == ' ' ? FLG(pos + 1) : f;
        if (!(f2 & (FW | FL | FM | FN)) && f) {
            pos += (c == ' ');
            while (!(f2 & (FW | FL | FM | FN)) && f2) f2 = FLG(++pos);
            u32 c2 = CPT(pos);
            while (c2 == '\r' || c2 == '\n') c2 = CPT(++pos);
            ADD(pos);
            continue;
        }
        i64 nws = 0, last_rn = 0;
        while (FLG(pos + nws) & FW) {
            u32 c2 = CPT(pos + nws);
            if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1;
            nws++;
        }
        if (last_rn > 0) { pos = last_rn; ADD(pos); continue; }          /* \s*[\r\n]+ */
        if (nws > 1 && CPT(pos + nws) != OOR) { pos += nws - 1; ADD(pos); continue; }  /* \s+(?!\S) */
        if (nws > 0) { pos += nws; ADD(pos); continue; }                  /* \s+ */
        pos++;
        ADD(pos);
    }
#undef ADD
#undef CPT
#undef FLG
    return k;
}

static i64 push_text(const char *s, i32 *out, i64 k, i64 cap) {
    return k + bz_tokenize((const u8 *)s, (i64)bz_strlen(s), out + k, cap - k);
}

i64 bz_chat_ids(const u8 *prompt, i64 think, i32 *out, i64 cap) {
    /* <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n[\n</think>\n\n] */
    i64 k = 0;
    if (cap < 16) return 0;
    out[k++] = M.im_start;
    {
        /* "user\n" + prompt is one raw-text fragment, exactly as llama.cpp splits around specials */
        i64 pl = (i64)bz_strlen((const char *)prompt);
        if (pl > MAX_PROMPT_BYTES - 8) pl = MAX_PROMPT_BYTES - 8;
        u8 *frag = (u8 *)b_h; /* free scratch at this point (>= 64 KiB) */
        bz_memcpy(frag, "user\n", 5);
        bz_memcpy(frag + 5, prompt, (u64)pl);
        k += bz_tokenize(frag, pl + 5, out + k, cap - k - 12);
    }
    if (k + 12 > cap) return 0;
    out[k++] = M.im_end;
    k = push_text("\n", out, k, cap);
    out[k++] = M.im_start;
    k = push_text("assistant\n", out, k, cap);
    out[k++] = M.think;
    if (think) {
        k = push_text("\n", out, k, cap);
    } else {
        k = push_text("\n\n", out, k, cap);
        out[k++] = M.think_end;
        k = push_text("\n\n", out, k, cap);
    }
    return k;
}

i64 bz_piece(i64 id, u8 *buf, i64 cap) {
    if (id < 0 || id >= M.n_vocab) return 0;
    const u8 *s = M.tok_txt[id];
    u64 n = M.tok_len[id];
    u8 ty = M.tok_type[id];
    if (ty == 3) return 0;                       /* control: llama.cpp token_to_piece(special=false) */
    i64 k = 0;
    if (ty == 4) {                               /* user-defined: raw text */
        for (u64 i = 0; i < n && k < cap; i++) buf[k++] = s[i];
        return k;
    }
    u32 cps[256];
    i64 nc = utf8_decode(s, (i64)n, cps, 256);
    for (i64 i = 0; i < nc && k < cap; i++) {
        u32 c = cps[i];
        if (c < 512 && M.u2b[c] >= 0) buf[k++] = (u8)M.u2b[c];
        else { u8 t[4]; int m = utf8_encode(c, t); for (int j = 0; j < m && k < cap; j++) buf[k++] = t[j]; }
    }
    return k;
}

i64 bz_token_id(const char *text) { return vocab_find((const u8 *)text, bz_strlen(text)); }

/* ------------------------------------------------------------------ init */

static void build_b2u(void) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        u32 cp;
        if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) cp = (u32)b;
        else cp = 256u + (u32)n++;
        M.b2u_len[b] = (u8)utf8_encode(cp, M.b2u[b]);
    }
    for (int i = 0; i < 512; i++) M.u2b[i] = -1;
    n = 0;
    for (int b = 0; b < 256; b++) {
        u32 cp;
        if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) cp = (u32)b;
        else cp = 256u + (u32)n++;
        M.u2b[cp] = (i16)b;
    }
}

static u32 pow2_at_least(u64 n) {
    u32 p = 1;
    while (p < n) p <<= 1;
    return p;
}

i64 bz_init(const u8 *model, u64 size, u8 *arena, u64 arena_size, i64 ctx) {
    g_err = 0;
    g_ar_base = g_ar_cur = arena;
    g_ar_end = arena + arena_size;
    bz_memset(&M, 0, sizeof(M));
    s_pos = s_chat_pos = s_ngen = 0;
    s_maxtok = 256;
    if (ctx < 64) ctx = 64;
    g_ctx = ctx;

    /* parts */
    if (bz_memeq(model, "GGUF", 4)) {
        g_nparts = 1;
        g_part_ptr[0] = model;
        g_part_off[0] = 0;
        g_part_off[1] = size;
    } else if (bz_memeq(model, "BZPARTS1", 8)) {
        const u64 *d = (const u64 *)model;
        g_nparts = (i64)d[1];
        if (g_nparts < 1 || g_nparts > MAX_PARTS) return E_PARTS;
        g_part_off[0] = 0;
        for (i64 i = 0; i < g_nparts; i++) {
            g_part_ptr[i] = (const u8 *)d[2 + 2 * i];
            g_part_off[i + 1] = g_part_off[i] + d[3 + 2 * i];
        }
        if (!bz_memeq(g_part_ptr[0], "GGUF", 4)) return E_FORMAT;
    } else {
        return E_FORMAT;
    }

    rd_t r = { g_part_ptr[0] + 4, g_part_ptr[0] + (g_part_off[1] - g_part_off[0]), 0 };
    u32 ver = (u32)rd_u(&r, 4);
    if (ver < 2) return E_FORMAT;
    i64 nt = (i64)rd_u(&r, 8), nkv = (i64)rd_u(&r, 8);
    if (nt > MAX_TENSORS) return E_FORMAT;
    u64 align = 32;
    int arch_ok = 0;
    const u8 *sign_vals = 0, *sign_w = 0, *wnames = 0, *inames = 0;
    u64 n_sign_vals = 0, n_sign_w = 0, n_wnames = 0, n_inames = 0;
    const u8 *toks = 0, *types = 0, *merges = 0;
    u64 n_toks = 0, n_types = 0, n_merges = 0;
    M.eos = 248046; M.eot = 248044;
    M.fa_interval = 4;
    for (i64 i = 0; i < nkv && !r.bad; i++) {
        u64 kl;
        const u8 *k = rd_str(&r, &kl);
        u32 t = (u32)rd_u(&r, 4);
#define KEY(z) str_is(k, kl, z)
        if (KEY("general.architecture")) {
            u64 l; const u8 *s = rd_str(&r, &l);
            arch_ok = str_is(s, l, "qwen35");
        } else if (KEY("general.alignment")) align = (u64)rd_int(&r, t);
        else if (KEY("qwen35.block_count")) M.n_layer = rd_int(&r, t);
        else if (KEY("qwen35.embedding_length")) M.n_embd = rd_int(&r, t);
        else if (KEY("qwen35.feed_forward_length")) M.n_ff = rd_int(&r, t);
        else if (KEY("qwen35.attention.head_count")) M.n_head = rd_int(&r, t);
        else if (KEY("qwen35.attention.head_count_kv")) M.n_head_kv = rd_int(&r, t);
        else if (KEY("qwen35.attention.key_length")) M.head_dim = rd_int(&r, t);
        else if (KEY("qwen35.rope.dimension_count")) M.n_rot = rd_int(&r, t);
        else if (KEY("qwen35.rope.freq_base")) M.rope_base = rd_f32(&r, t);
        else if (KEY("qwen35.attention.layer_norm_rms_epsilon")) M.eps = rd_f32(&r, t);
        else if (KEY("qwen35.ssm.conv_kernel")) M.conv_k = rd_int(&r, t);
        else if (KEY("qwen35.ssm.state_size")) M.d_state = rd_int(&r, t);
        else if (KEY("qwen35.ssm.group_count")) M.n_group = rd_int(&r, t);
        else if (KEY("qwen35.ssm.time_step_rank")) M.dt_rank = rd_int(&r, t);
        else if (KEY("qwen35.ssm.inner_size")) M.d_inner = rd_int(&r, t);
        else if (KEY("qwen35.full_attention_interval")) M.fa_interval = rd_int(&r, t);
        else if (KEY("qwen35.rope.dimension_sections") && t == 9) {
            u32 et = (u32)rd_u(&r, 4); u64 n = rd_u(&r, 8);
            for (u64 j = 0; j < n; j++) { i64 v = rd_int(&r, et); if (j < 4) M.sections[j] = (i32)v; }
        } else if (KEY("prism.hadamard.block_size")) M.had_block = rd_int(&r, t);
        else if (KEY("prism.hadamard.gdn_v_grouped")) M.gdn_grouped = (int)rd_int(&r, t);
        else if (t == 9 && (KEY("prism.hadamard.sign_values") || KEY("prism.hadamard.sign_widths") ||
                            KEY("prism.hadamard.weight_names") || KEY("prism.hadamard.inverse_weight_names") ||
                            KEY("tokenizer.ggml.tokens") || KEY("tokenizer.ggml.token_type") || KEY("tokenizer.ggml.merges"))) {
            u32 et = (u32)rd_u(&r, 4); u64 n = rd_u(&r, 8);
            const u8 *start = r.p;
            if (KEY("prism.hadamard.sign_values")) { sign_vals = start; n_sign_vals = n; if (et != 5) return E_FORMAT; }
            else if (KEY("prism.hadamard.sign_widths")) { sign_w = start; n_sign_w = n; if (et != 5) return E_FORMAT; }
            else if (KEY("prism.hadamard.weight_names")) { wnames = start; n_wnames = n; }
            else if (KEY("prism.hadamard.inverse_weight_names")) { inames = start; n_inames = n; }
            else if (KEY("tokenizer.ggml.tokens")) { toks = start; n_toks = n; }
            else if (KEY("tokenizer.ggml.token_type")) { types = start; n_types = n; if (et != 5) return E_FORMAT; }
            else { merges = start; n_merges = n; }
            /* skip the payload */
            if (et == 8) { for (u64 j = 0; j < n && !r.bad; j++) { u64 l; rd_str(&r, &l); } }
            else { int s = type_size(et); r.p += n * (u64)s; if (r.p > r.end) r.bad = 1; }
        } else if (KEY("tokenizer.ggml.eos_token_id")) M.eos = (i32)rd_int(&r, t);
        else if (KEY("tokenizer.ggml.padding_token_id")) M.eot = (i32)rd_int(&r, t);
        else rd_skip(&r, t);
#undef KEY
    }
    if (r.bad) return E_FORMAT;
    if (!arch_ok) return E_ARCH;

    g_ti = (tinfo_t *)ar_alloc(sizeof(tinfo_t) * (u64)nt);
    if (!g_ti) return E_OOM;
    g_nt = nt;
    for (i64 i = 0; i < nt && !r.bad; i++) {
        tinfo_t *ti = &g_ti[i];
        ti->name = rd_str(&r, &ti->nlen);
        ti->nd = (u32)rd_u(&r, 4);
        if (ti->nd > 4) return E_FORMAT;
        for (u32 j = 0; j < 4; j++) ti->ne[j] = 1;
        for (u32 j = 0; j < ti->nd; j++) ti->ne[j] = rd_u(&r, 8);
        ti->type = (u32)rd_u(&r, 4);
        ti->off = rd_u(&r, 8);
    }
    if (r.bad) return E_FORMAT;
    g_data_start = ((u64)(r.p - g_part_ptr[0]) + align - 1) / align * align;

    /* hparams sanity: this engine is the 27B shape's code path, not a generic qwen35 */
    if (M.n_layer <= 0 || M.n_layer > MAX_LAYERS || M.n_embd != 5120 || M.head_dim % 8 || M.d_state % 8 ||
        M.n_head % M.n_head_kv || M.dt_rank % M.n_group || M.conv_k < 2 || M.n_rot > M.head_dim)
        return E_SHAPE;
    M.key_dim = M.d_state * M.n_group;
    M.v_dim = M.d_inner;
    M.conv_ch = 2 * M.key_dim + M.v_dim;
    if (M.d_inner != M.d_state * M.dt_rank) return E_SHAPE;
    if (M.eps <= 0) M.eps = 1e-6f;

    /* Hadamard metadata: folded weight names, inverse table, sign vectors per width */
    if (wnames) {
        rd_t w = { wnames, r.end, 0 };
        for (u64 i = 0; i < n_wnames; i++) {
            u64 l; const u8 *s = rd_str(&w, &l);
            for (i64 j = 0; j < nt; j++)
                if (g_ti[j].nlen == l && bz_memeq(g_ti[j].name, s, l)) g_ti[j].folded = 1;
        }
        if (M.had_block != 1024) return E_FORMAT;
    }
    if (inames) {
        rd_t w = { inames, r.end, 0 };
        for (u64 i = 0; i < n_inames; i++) {
            u64 l; const u8 *s = rd_str(&w, &l);
            if (str_is(s, l, "token_embd.weight")) M.embd_inverse = 1;
            else return E_FORMAT;
        }
    }
    if (wnames || inames) {
        if (!sign_vals || !sign_w) return E_FORMAT;
        u64 off = 0;
        for (u64 i = 0; i < n_sign_w; i++) {
            i32 width = (i32)(sign_w[4 * i] | (sign_w[4 * i + 1] << 8) | (sign_w[4 * i + 2] << 16) | ((u32)sign_w[4 * i + 3] << 24));
            if (width <= 0 || off + (u64)width > n_sign_vals) return E_FORMAT;
            float *dst = (float *)ar_alloc((u64)width * 4);
            if (!dst) return E_OOM;
            for (i32 j = 0; j < width; j++) {
                const u8 *p = sign_vals + 4 * (off + (u64)j);
                i32 v = (i32)(p[0] | (p[1] << 8) | (p[2] << 16) | ((u32)p[3] << 24));
                dst[j] = (float)v;
            }
            if (width == 5120) M.sign5120 = dst;
            else if (width == 6144) M.sign6144 = dst;
            else if (width == 17408) M.sign17408 = dst;
            off += (u64)width;
        }
        if (!M.sign5120 || !M.sign6144 || !M.sign17408) return E_FORMAT;
    }

    /* weights */
    const i64 E = M.n_embd, F = M.n_ff, H = M.n_head, HK = M.n_head_kv, D = M.head_dim;
    {
        tinfo_t *te = find_t("token_embd.weight");
        if (!te || te->nd != 2 || (i64)te->ne[0] != E) return E_TENSOR;
        M.n_vocab = (i64)te->ne[1];
    }
    M.tok_embd = get_w("token_embd.weight", E, M.n_vocab);
    if (M.tok_embd.type != TYPE_PQ2_0) return E_FORMAT;
    if (find_t("output.weight")) M.output = get_w("output.weight", E, M.n_vocab);
    else M.output = M.tok_embd;                       /* tied (no inverse interaction: not our checkpoint) */
    M.out_norm = get_f32("output_norm.weight", E);
    i64 na = 0, nr = 0;
    for (i64 il = 0; il < M.n_layer; il++) {
        layer_t *L = &M.L[il];
        L->rec = ((il + 1) % M.fa_interval) != 0;
        L->attn_norm = get_f32(tname((int)il, "attn_norm.weight"), E);
        L->post_norm = get_f32(tname((int)il, "post_attention_norm.weight"), E);
        if (L->rec) {
            L->slot = (int)nr++;
            L->wqkv = get_w(tname((int)il, "attn_qkv.weight"), E, M.conv_ch);
            L->wgate = get_w(tname((int)il, "attn_gate.weight"), E, M.v_dim);
            L->wout = get_w(tname((int)il, "ssm_out.weight"), M.v_dim, E);
            L->alpha = get_f32(tname((int)il, "ssm_alpha.weight"), E * M.dt_rank);
            L->beta = get_f32(tname((int)il, "ssm_beta.weight"), E * M.dt_rank);
            L->conv = get_f32(tname((int)il, "ssm_conv1d.weight"), M.conv_k * M.conv_ch);
            L->dt = get_f32(tname((int)il, "ssm_dt.bias"), M.dt_rank);
            L->ssm_a = get_f32(tname((int)il, "ssm_a"), M.dt_rank);
            L->ssm_norm = get_f32(tname((int)il, "ssm_norm.weight"), M.d_state);
        } else {
            L->slot = (int)na++;
            L->wq = get_w(tname((int)il, "attn_q.weight"), E, H * D * 2);
            L->wk = get_w(tname((int)il, "attn_k.weight"), E, HK * D);
            L->wv = get_w(tname((int)il, "attn_v.weight"), E, HK * D);
            L->wo = get_w(tname((int)il, "attn_output.weight"), H * D, E);
            L->q_norm = get_f32(tname((int)il, "attn_q_norm.weight"), D);
            L->k_norm = get_f32(tname((int)il, "attn_k_norm.weight"), D);
        }
        L->fgate = get_w(tname((int)il, "ffn_gate.weight"), E, F);
        L->fup = get_w(tname((int)il, "ffn_up.weight"), E, F);
        L->fdown = get_w(tname((int)il, "ffn_down.weight"), F, E);
        if (g_err) return g_err;
    }
    if (g_err) return g_err;
    M.n_attn = na;
    M.n_rec = nr;
    /* every matmul weight must be PQ2_0 here (the F32 path covers only alpha/beta) */
    for (i64 il = 0; il < M.n_layer; il++) {
        layer_t *L = &M.L[il];
        /* (no table of &L->w pointers: the vectoriser turns one into an absolute relocation) */
        u32 bad = L->rec ? (L->wqkv.type ^ TYPE_PQ2_0) | (L->wgate.type ^ TYPE_PQ2_0) | (L->wout.type ^ TYPE_PQ2_0)
                         : (L->wq.type ^ TYPE_PQ2_0) | (L->wk.type ^ TYPE_PQ2_0) | (L->wv.type ^ TYPE_PQ2_0) | (L->wo.type ^ TYPE_PQ2_0);
        if (bad) return E_FORMAT;
        if (L->fgate.type != TYPE_PQ2_0 || L->fup.type != TYPE_PQ2_0 || L->fdown.type != TYPE_PQ2_0) return E_FORMAT;
    }
    if (M.output.type != TYPE_PQ2_0) return E_FORMAT;

    /* vocab + merges */
    if (!toks || (i64)n_toks != M.n_vocab || !merges) return E_FORMAT;
    M.n_merges = (i64)n_merges;
    M.tok_txt = (const u8 **)ar_alloc(8 * (u64)M.n_vocab);
    M.tok_len = (u32 *)ar_alloc(4 * (u64)M.n_vocab);
    M.tok_type = (u8 *)ar_alloc((u64)M.n_vocab);
    M.mrg_txt = (const u8 **)ar_alloc(8 * n_merges);
    M.mrg_len = (u32 *)ar_alloc(4 * n_merges);
    u32 vs = pow2_at_least((u64)M.n_vocab * 2), ms = pow2_at_least(n_merges * 2);
    M.vh = (u32 *)ar_alloc(4ull * vs);
    M.mh = (u32 *)ar_alloc(4ull * ms);
    if (!M.tok_txt || !M.tok_len || !M.tok_type || !M.mrg_txt || !M.mrg_len || !M.vh || !M.mh) return E_OOM;
    M.vh_mask = vs - 1;
    M.mh_mask = ms - 1;
    {
        rd_t w = { toks, r.end, 0 };
        for (i64 i = 0; i < M.n_vocab; i++) {
            u64 l; const u8 *s = rd_str(&w, &l);
            M.tok_txt[i] = s;
            M.tok_len[i] = (u32)l;
            u64 h = h_add(h_init(), s, l);
            u32 j = (u32)h & M.vh_mask;
            while (M.vh[j]) j = (j + 1) & M.vh_mask;
            M.vh[j] = (u32)i + 1;
        }
        for (i64 i = 0; i < M.n_vocab; i++) {
            if (types && (u64)i < n_types) M.tok_type[i] = types[4 * i];
            else M.tok_type[i] = 1;
        }
        rd_t m = { merges, r.end, 0 };
        for (u64 i = 0; i < n_merges; i++) {
            u64 l; const u8 *s = rd_str(&m, &l);
            M.mrg_txt[i] = s;
            M.mrg_len[i] = (u32)l;
            u64 h = h_add(h_init(), s, l);
            u32 j = (u32)h & M.mh_mask;
            while (M.mh[j]) j = (j + 1) & M.mh_mask;
            M.mh[j] = (u32)i + 1;
        }
        if (w.bad || m.bad) return E_FORMAT;
    }
    build_b2u();
    M.im_start = (i32)vocab_find((const u8 *)"<|im_start|>", 12);
    M.im_end = (i32)vocab_find((const u8 *)"<|im_end|>", 10);
    M.think = (i32)vocab_find((const u8 *)"<think>", 7);
    M.think_end = (i32)vocab_find((const u8 *)"</think>", 8);
    if (M.im_start < 0 || M.im_end < 0 || M.think < 0 || M.think_end < 0) return E_FORMAT;

    /* runtime buffers */
    const i64 KW = HK * D, QW = H * D * 2;
    g_kcache = (u16 *)ar_alloc(2ull * (u64)(na * ctx * KW));
    g_vcache = (u16 *)ar_alloc(2ull * (u64)(na * ctx * KW));
    g_ssm = (float *)ar_alloc(4ull * (u64)(nr * M.dt_rank * M.d_state * M.d_state));
    g_conv = (float *)ar_alloc(4ull * (u64)(nr * (M.conv_k - 1) * M.conv_ch));
    g_logits = (float *)ar_alloc(4ull * (u64)M.n_vocab);
    const i64 TM = BZ_TMAX;
    i64 big = QW > M.conv_ch ? QW : M.conv_ch;
    i64 wide = F > M.v_dim ? F : M.v_dim;
    if (wide < H * D) wide = H * D;
    b_x = (float *)ar_alloc(4ull * (u64)(TM * E));
    b_xn = (float *)ar_alloc(4ull * (u64)(TM * E));
    b_h = (float *)ar_alloc(4ull * (u64)(wide > 16384 ? wide : 16384));
    b_big = (float *)ar_alloc(4ull * (u64)(TM * big));
    b_z = (float *)ar_alloc(4ull * (u64)(TM * M.v_dim));
    b_g = (float *)ar_alloc(4ull * (u64)(TM * F));
    b_u = (float *)ar_alloc(4ull * (u64)(TM * F > wide ? TM * F : wide));
    b_o = (float *)ar_alloc(4ull * (u64)(TM * wide));
    b_k = (float *)ar_alloc(4ull * (u64)(TM * KW));
    b_v = (float *)ar_alloc(4ull * (u64)(TM * KW));
    b_al = (float *)ar_alloc(4ull * (u64)(TM * M.dt_rank));
    b_be = (float *)ar_alloc(4ull * (u64)(TM * M.dt_rank));
    b_att = (float *)ar_alloc(4ull * (u64)(TM * E));
    b_q8 = ar_alloc(sizeof(qblk_t) * (u64)(TM * (wide / 128 + 1)));
    w_scores = (float *)ar_alloc(4ull * (u64)ctx);
    w_conv = (float *)ar_alloc(4ull * (u64)M.conv_ch);
    g_ids = (i32 *)ar_alloc(4ull * MAX_PROMPT_TOKENS);
    g_cpt = (u32 *)ar_alloc(4ull * MAX_PROMPT_BYTES);
    g_word = (u8 *)ar_alloc(2ull * MAX_PROMPT_BYTES + 64);
    g_sym = (i32 *)ar_alloc(8ull * 2 * MAX_PROMPT_BYTES);
    if (!g_kcache || !g_vcache || !g_ssm || !g_conv || !g_logits || !b_x || !b_xn || !b_h || !b_big || !b_z ||
        !b_g || !b_u || !b_o || !b_k || !b_v || !b_al || !b_be || !b_att || !b_q8 || !w_scores || !w_conv ||
        !g_ids || !g_cpt || !g_word || !g_sym)
        return E_OOM;
    s_need = (i64)(g_ar_cur - g_ar_base);
    return 0;
}

/* ------------------------------------------------------------------ session (ZlBegin/ZlNext/ZlEnd) */

static void state_clear(void) {
    bz_memset(g_ssm, 0, 4ull * (u64)(M.n_rec * M.dt_rank * M.d_state * M.d_state));
    bz_memset(g_conv, 0, 4ull * (u64)(M.n_rec * (M.conv_k - 1) * M.conv_ch));
}

i64 bz_begin(const u8 *prompt, i64 think) {
    i64 n = bz_chat_ids(prompt, think, g_ids, MAX_PROMPT_TOKENS);
    if (n <= 0) return n;
    /* context room: start over when this turn + 64 would not fit (recurrent state too) */
    if (s_chat_pos + n + 64 >= g_ctx) { s_chat_pos = 0; }
    if (s_chat_pos == 0) state_clear();
    float *l = bz_eval(g_ids, n, s_chat_pos, 1);
    if (!l) return -1;
    s_pos = s_chat_pos + n;
    s_next = bz_argmax(l);
    s_ngen = 0;
    return n;
}

i64 bz_next(u8 *buf, i64 cap) {
    if (s_ngen >= s_maxtok || s_next == M.eos || s_next == M.eot || s_next == M.im_end || s_pos >= g_ctx) return -1;
    i64 nb = bz_piece(s_next, buf, cap);
    s_ngen++;
    i32 id = (i32)s_next;
    float *l = bz_eval(&id, 1, s_pos, 1);
    if (!l) return -1;
    s_pos++;
    s_next = bz_argmax(l);
    return nb;
}

i64 bz_end(void) {
    i64 p = s_pos;
    if (p + 2 < g_ctx) {
        i32 ids[2];
        ids[0] = M.im_end;
        i64 k = push_text("\n", ids, 1, 2);
        bz_eval(ids, k, p, 0);
        p += k;
    }
    s_chat_pos = p;
    return p;
}

i64 bz_reset(void) {
    s_chat_pos = 0;
    s_pos = 0;
    return 0;
}

i64 bz_stat(i64 which) {
    if (which >= 100) { s_maxtok = which - 100; return s_maxtok; }
    switch (which) {
    case 0: return s_pos;
    case 1: return g_ctx;
    case 2: return (i64)(g_ar_cur - g_ar_base);
    case 3: return M.n_vocab;
    case 4: return s_ngen;
    case 5: return s_next;
    default: return s_need;
    }
}
