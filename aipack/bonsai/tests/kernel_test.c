/* kernel_test.c -- per-block parity of the engine's matmul formats against the fork oracles.
 *
 * Includes the engine source so the static kernels are reachable, and checks, per format:
 *   dequant  : scalar decoder (embedding path) == oracle dequant, bit for bit
 *   codes    : SIMD decoder (matmul path) == oracle trits/codes, every element
 *   dot      : a scalar transcription of the fork's vec_dot (int per 32, f32 d1*sumi, f32 d0*sumi)
 *              == oracle dot bit for bit; the SIMD kernel within 1e-5 relative (the TS tests'
 *              tolerance: ptq1_0-layout.test.ts) -- its f32 summation order is ours
 *   batching : T tokens and row splits give results bit-identical to T=1 / one slice
 * plus the FWHT at 1024 (oracle) and 512 (dense double), and quant_row's two layouts.
 * Exit 0 = all pass, 1 = a failure (printed). Build: gcc -O2 -mavx2 -mfma -mf16c.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../engine/bonsai.c"
#include "oracle.h"

static int g_fail, g_checks;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_fail++; if (g_fail < 20) { printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

/* where element e sits in a grouped-layout 128-byte block (dec_ptq1/dec_q1 output, quant_row nat) */
static int gpos(int e) { int k = e >> 5, r = e & 31; return (r >> 3) * 32 + 8 * k + (r & 7); }

/* grouped-layout activation block from oracle q8 (d per 32, qs) -- what quant_row(nat) emits */
static void q8_nat(const float *d, const signed char *qs, i64 nb, qblk_t *o) {
    for (i64 b = 0; b < nb; b++)
        for (int k = 0; k < 4; k++) {
            int s0 = 0, s1 = 0;
            for (int j = 0; j < 32; j++) {
                int q = qs[(b * 4 + k) * 32 + j];
                o[b].q[gpos(k * 32 + j)] = (i8)q;
                if ((j & 7) < 4) s0 += q; else s1 += q;
            }
            float dk = d[b * 4 + k];
            o[b].sv[2 * k] = o[b].sv[2 * k + 1] = dk;
            o[b].cv[2 * k] = dk * (float)s0;
            o[b].cv[2 * k + 1] = dk * (float)s1;
        }
}
static void q8_perm(const float *d, const signed char *qs, i64 nb, qblk_t *o) {
    for (i64 b = 0; b < nb; b++)
        for (int k = 0; k < 4; k++) {
            int s = 0;
            for (int j = 0; j < 32; j++) {
                int q = qs[(b * 4 + k) * 32 + j], e = k * 32 + j;
                o[b].q[(e & 3) * 32 + (e >> 2)] = (i8)q;
                s += q;
            }
            float dk = d[b * 4 + k];
            o[b].sv[2 * k] = o[b].sv[2 * k + 1] = dk;
            o[b].cv[2 * k] = o[b].cv[2 * k + 1] = dk * (float)s * 0.5f;
        }
}

/* the fork's generic vec_dot, f32 per step, with the element codes from blk_code */
static float ref_dot(const u8 *row, u32 type, i64 K, const float *d, const signed char *qs) {
    const i64 bs = blk_bytes(type);
    volatile float sumf = 0.0f;
    for (i64 i = 0; i < K / 128; i++) {
        const u8 *wb = row + i * bs;
        volatile float sumi = 0.0f;
        for (int k = 0; k < 4; k++) {
            int sb = 0;
            for (int b = 0; b < 32; b++) sb += blk_code(wb, type, k * 32 + b) * qs[(i * 4 + k) * 32 + b];
            volatile float p = d[i * 4 + k] * (float)sb;
            sumi = sumi + p;
        }
        volatile float p = blk_d(wb, type) * sumi;
        sumf = sumf + p;
    }
    return sumf;
}

static float rel(float a, float b) { float m = fabsf(b) > 1e-6f ? fabsf(b) : 1e-6f; return fabsf(a - b) / m; }

static qblk_t xq[64 * 8] __attribute__((aligned(64)));
static float outv[64 * 8];

static void dot_suite(const char *nm, u32 type, int nat, const u8 *rows, int nd, i64 K, const float *q8d,
                      const signed char *q8qs, const float *oracle) {
    const i64 nb = K / 128, bs = blk_bytes(type);
    float worst = 0;
    for (int i = 0; i < nd; i++) {
        const u8 *row = rows + i * nb * bs;
        const float *d = q8d + i * K / 32;
        const signed char *qs = q8qs + i * K;
        float r = ref_dot(row, type, K, d, qs);
        if (oracle) CHECK(r == oracle[i], "%s ref dot %d: %.9g != oracle %.9g", nm, i, r, oracle[i]);
        if (nat) q8_nat(d, qs, nb, xq); else q8_perm(d, qs, nb, xq);
        wt_t w = { row, K, 1, type, 0 };
        mm_rows(&w, xq, nb, 1, outv, 1, 0, 1);
        float e = rel(outv[0], oracle ? oracle[i] : r);
        if (e > worst) worst = e;
        CHECK(e < 1e-5f, "%s simd dot %d: %.9g vs %.9g (rel %.3g)", nm, i, outv[0], oracle ? oracle[i] : r, e);
        /* batch of 7 identical tokens must equal T=1 exactly */
        for (int t = 1; t < 7; t++) memcpy(xq + t * nb, xq, sizeof(qblk_t) * nb);
        float o7[7];
        mm_rows(&w, xq, nb, 7, o7, 1, 0, 1);
        for (int t = 0; t < 7; t++) CHECK(o7[t] == outv[0], "%s batch t=%d %.9g != %.9g", nm, t, o7[t], outv[0]);
    }
    /* all rows as one matrix against activation 0: whole == split slices, bit for bit */
    if (nat) q8_nat(q8d, q8qs, nb, xq); else q8_perm(q8d, q8qs, nb, xq);
    wt_t W = { rows, K, nd, type, 0 };
    float full[16], part[16];
    mm_rows(&W, xq, nb, 1, full, 1, 0, nd);
    for (i64 tid = 0; tid < 3; tid++) { i64 r0, r1; tp_slice(tid, 3, nd, &r0, &r1); mm_rows(&W, xq, nb, 1, part, 1, r0, r1); }
    for (int i = 0; i < nd; i++) CHECK(full[i] == part[i], "%s split row %d", nm, i);
    printf("%-7s dot: %d rows K=%ld, %s, simd worst rel %.2e, batch+split exact\n", nm, nd, (long)K,
           oracle ? "scalar ref == oracle bit-exact" : "no dot oracle: simd vs scalar ref", worst);
}

static void codes_suite(const char *nm, u32 type, const u8 *bytes, int nbk, const float *deq) {
    const i64 bs = blk_bytes(type);
    int bad_deq = 0, bad_simd = 0;
    for (int b = 0; b < nbk; b++) {
        const u8 *wb = bytes + b * bs;
        float x[128];
        dequant_row(wb, type, 128, x);
        for (int j = 0; j < 128; j++) if (x[j] != deq[b * 128 + j]) bad_deq++;
        if (blk_nat(type)) {
            __m256i c[4];
            u8 cb[128];
            if (type == TYPE_PTQ1_0) dec_ptq1(wb, c); else dec_q1(wb, c);
            memcpy(cb, c, 128);
            float d = blk_d(wb, type);
            for (int j = 0; j < 128; j++) if ((float)((int)cb[gpos(j)] - 1) * d != deq[b * 128 + j]) bad_simd++;
        }
    }
    CHECK(bad_deq == 0, "%s dequant mismatches %d", nm, bad_deq);
    CHECK(bad_simd == 0, "%s simd decode mismatches %d", nm, bad_simd);
    printf("%-7s blocks: %d x 128, scalar dequant mismatches %d, simd decode mismatches %d\n", nm, nbk, bad_deq, bad_simd);
}

static void fwht_suite(void) {
    float t[1024];
    double worst = 0;
    for (int i = 0; i < 1024; i++) t[i] = fw_x[i] * (float)fw_s[i];
    fwht_n(t, 1024);
    for (int i = 0; i < 1024; i++) { double e = fabs(t[i] - fw_fwd[i]); if (e > worst) worst = e; }
    CHECK(worst < 2e-5, "fwht1024 max abs err %.3g", worst);
    printf("fwht    1024 vs oracle H(s*x): max abs err %.2e\n", worst);
    /* 512: against the dense (-1)^popcount(r&c)/sqrt(512) matrix in double */
    float u[512];
    double w2 = 0;
    for (int i = 0; i < 512; i++) u[i] = fw_x[i] * (float)fw_s[i];
    fwht_n(u, 512);
    for (int r = 0; r < 512; r++) {
        double s = 0;
        for (int c = 0; c < 512; c++) s += (__builtin_popcount(r & c) & 1 ? -1.0 : 1.0) * (double)(fw_x[c] * (float)fw_s[c]);
        s /= sqrt(512.0);
        double e = fabs(u[r] - s);
        if (e > w2) w2 = e;
    }
    CHECK(w2 < 2e-5, "fwht512 max abs err %.3g", w2);
    printf("fwht     512 vs dense double: max abs err %.2e\n", w2);
}

static void quant_suite(void) {
    float x[512];
    u32 s = 12345;
    for (int i = 0; i < 512; i++) { s = s * 1103515245u + 12345u; x[i] = ((float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f) * (i % 7 + 1); }
    static qblk_t a[4], b[4];
    quant_row(x, 512, a, 0);
    quant_row(x, 512, b, 1);
    int bad = 0;
    for (int k = 0; k < 4; k++)
        for (int e = 0; e < 128; e++) if (a[k].q[(e & 3) * 32 + (e >> 2)] != b[k].q[gpos(e)]) bad++;
    for (int k = 0; k < 4; k++)
        for (int j = 0; j < 8; j++) {
            if (a[k].sv[j] != b[k].sv[j]) bad++;
            float ta = a[k].cv[j & ~1] * 2, tb = b[k].cv[j & ~1] + b[k].cv[j | 1];
            if (fabsf(ta - tb) > 1e-6f * fabsf(tb) + 1e-12f) bad++;
        }
    CHECK(bad == 0, "quant_row layouts disagree (%d)", bad);
    printf("q8_0    interleaved vs grouped layout: %d disagreements\n", bad);
}

#include <stdlib.h>
#include <time.h>
/* --bench: single-core matvec throughput per format on an in-RAM 2560 x 9216 matrix (T=1) */
static void bench(void) {
    const i64 K = 2560, R = 9216, nb = K / 128;
    static qblk_t bx[64] __attribute__((aligned(64)));
    float *x = malloc(4 * K), *o = malloc(4 * R);
    for (i64 i = 0; i < K; i++) x[i] = (float)((i * 7919) % 201) / 100.0f - 1.0f;
    u32 types[3] = { TYPE_PTQ1_0, TYPE_PQ2_0, TYPE_Q1_0 };
    for (int ti = 0; ti < 3; ti++) {
        u32 ty = types[ti];
        i64 bs = blk_bytes(ty);
        u8 *wd = malloc(R * nb * bs);
        for (i64 i = 0; i < R * nb * bs; i++) wd[i] = (u8)((i * 2654435761u) >> 13);
        for (i64 i = 0; i < R * nb; i++) { u8 *b = wd + i * bs + (ty == TYPE_PTQ1_0 ? 26 : 0); b[0] = 0x00; b[1] = 0x3c; }
        quant_row(x, K, bx, blk_nat(ty));
        wt_t w = { wd, K, R, ty, 0 };
        struct timespec a, c;
        mm_rows(&w, bx, nb, 1, o, 1, 0, R);
        clock_gettime(CLOCK_MONOTONIC, &a);
        int it = 30;
        for (int k = 0; k < it; k++) mm_rows(&w, bx, nb, 1, o, 1, 0, R);
        clock_gettime(CLOCK_MONOTONIC, &c);
        double s = (c.tv_sec - a.tv_sec) + (c.tv_nsec - a.tv_nsec) * 1e-9;
        printf("bench type %3u: %.2f Gweights/s  %.2f GB/s (1 core, T=1)\n", ty, (double)K * R * it / s * 1e-9,
               (double)R * nb * bs * it / s * 1e-9);
        free(wd);
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--bench")) { bench(); return 0; }
    codes_suite("PTQ1_0", TYPE_PTQ1_0, ptq1_bytes, PTQ1_NB, ptq1_deq);
    {   /* trits too (the oracle's own element order) */
        int bad = 0;
        for (int b = 0; b < PTQ1_NB; b++)
            for (int j = 0; j < 128; j++) if (blk_code(ptq1_bytes + b * 28, TYPE_PTQ1_0, j) != ptq1_trits[b * 128 + j]) bad++;
        CHECK(bad == 0, "ptq1 trits %d", bad);
    }
    dot_suite("PTQ1_0", TYPE_PTQ1_0, 1, ptq1_rows, PTQ1_ND, PTQ1_K, ptq1_q8d, ptq1_q8qs, ptq1_dot);
    codes_suite("Q1_0", TYPE_Q1_0, q1_bytes, Q1_NB, q1_deq);
    dot_suite("Q1_0", TYPE_Q1_0, 1, q1_rows, Q1_ND, Q1_K, q1_q8d, q1_q8qs, q1_dot);
    codes_suite("PQ2_0", TYPE_PQ2_0, pq2_bytes, PQ2_NB, pq2_deq);
    codes_suite("Q2_0", TYPE_Q2_0, pq2_bytes, PQ2_NB, pq2_deq);
    {   /* 2-bit dots: oracle blocks as rows of 4, q8 borrowed from the PTQ1 oracle (no PQ2 dot oracle) */
        dot_suite("PQ2_0", TYPE_PQ2_0, 0, pq2_bytes, PTQ1_ND, 512, ptq1_q8d, ptq1_q8qs, 0);
        dot_suite("Q2_0", TYPE_Q2_0, 0, pq2_bytes, PTQ1_ND, 512, ptq1_q8d, ptq1_q8qs, 0);
        int bad = 0;
        for (int b = 0; b < PQ2_NB; b++)
            for (int j = 0; j < 128; j++) if (blk_code(pq2_bytes + b * 34, TYPE_PQ2_0, j) != pq2_codes[b * 128 + j] - 1) bad++;
        CHECK(bad == 0, "pq2 codes %d", bad);
    }
    fwht_suite();
    quant_suite();
    printf("%s: %d checks, %d failed\n", g_fail ? "KERNEL-PARITY FAIL" : "KERNEL-PARITY PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
