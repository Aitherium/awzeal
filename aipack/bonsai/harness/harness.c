/* harness.c -- Linux userspace proof for the freestanding Bonsai engine.
 *
 * mmap()s the GGUF read-only (never copies it), gives bz_init an anonymous arena,
 * and calls the same entry points the ZealOS glue calls.
 *
 *   harness <gguf> tok   "<text>"                  raw-text token ids
 *   harness <gguf> chat  "<question>" [n] [ids]    chat-template ids, prefill, n greedy tokens;
 *                                                  ids = comma list to compare the prompt with
 *   harness <gguf> session "<question>"            ZlBegin/ZlNext/ZlEnd path (what ZealOS runs)

 *
 * env BZ_CTX (default 512), BZ_ARENA_MB (default 448).
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../engine/bonsai.h"

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void top5(const float *l, int64_t nv) {
    int64_t best[5] = { -1, -1, -1, -1, -1 };
    for (int64_t i = 0; i < nv; i++) {
        for (int k = 0; k < 5; k++) {
            if (best[k] < 0 || l[i] > l[best[k]]) {
                for (int j = 4; j > k; j--) best[j] = best[j - 1];
                best[k] = i;
                break;
            }
        }
    }
    /* log-softmax of the top entries */
    double mx = l[best[0]], s = 0;
    for (int64_t i = 0; i < nv; i++) s += __builtin_exp((double)l[i] - mx);
    double lse = mx + __builtin_log(s);
    printf("   top5:");
    for (int k = 0; k < 5; k++) printf(" %ld(%.4f)", (long)best[k], (double)l[best[k]] - lse);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <gguf> tok|chat|session ...\n", argv[0]);
        return 2;
    }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st;
    fstat(fd, &st);
    uint8_t *model = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, fd, 0);
    if (model == MAP_FAILED) { perror("mmap"); return 2; }
    madvise(model, st.st_size, MADV_SEQUENTIAL);
    int64_t ctx = getenv("BZ_CTX") ? atol(getenv("BZ_CTX")) : 512;
    uint64_t asz = (uint64_t)(getenv("BZ_ARENA_MB") ? atol(getenv("BZ_ARENA_MB")) : 448) << 20;
    uint8_t *arena = mmap(0, asz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) { perror("mmap arena"); return 2; }
    /* BZ_CUTS="o1,o2,...": hand the engine a BZPARTS1 table of separate, non-adjacent
       mappings cut at those file offsets (what AIBONSAI.ZC builds from BZnnn.BIN) */
    static uint64_t parts[2 + 2 * 16];
    const uint8_t *init_ptr = model;
    if (getenv("BZ_CUTS")) {
        uint64_t cut[17];
        int nc = 0;
        cut[nc++] = 0;
        for (char *p = getenv("BZ_CUTS"); *p && nc < 16;) { cut[nc++] = strtoull(p, &p, 10); if (*p == ',') p++; }
        cut[nc] = st.st_size;
        memcpy(parts, "BZPARTS1", 8);
        parts[1] = nc;
        long pg = sysconf(_SC_PAGESIZE);
        for (int i = 0; i < nc; i++) {
            uint64_t a = cut[i] / pg * pg, len = cut[i + 1] - cut[i];
            /* a guard page between parts proves nothing relies on adjacency */
            uint8_t *m = mmap(0, len + (cut[i] - a) + pg, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, fd, a);
            if (m == MAP_FAILED) { perror("mmap part"); return 2; }
            parts[2 + 2 * i] = (uint64_t)(m + (cut[i] - a));
            parts[3 + 2 * i] = len;
            printf("part %d: file [%llu, %llu) at %p\n", i, (unsigned long long)cut[i], (unsigned long long)cut[i + 1], (void *)parts[2 + 2 * i]);
        }
        init_ptr = (const uint8_t *)parts;
    }
    double t0 = now();
    int64_t rc = bz_init(init_ptr, st.st_size, arena, asz, ctx);
    double t1 = now();
    printf("bz_init rc=%ld %.2fs arena_used=%.1f MiB ctx=%ld vocab=%ld\n", (long)rc, t1 - t0,
           bz_stat(2) / 1048576.0, (long)bz_stat(1), (long)bz_stat(3));
    if (rc < 0) return 1;
    const char *mode = argv[2];
    static int32_t ids[8192];
    if (!strcmp(mode, "tok") && argc > 3) {
        int64_t n = bz_tokenize((const uint8_t *)argv[3], strlen(argv[3]), ids, 8192);
        for (int64_t i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
        printf("\n");
        return 0;
    }
    if (!strcmp(mode, "chat") && argc > 3) {
        int ngen = argc > 4 ? atoi(argv[4]) : 8;
        int64_t n = bz_chat_ids((const uint8_t *)argv[3], 0, ids, 8192);
        printf("prompt_ids");
        for (int64_t i = 0; i < n; i++) printf("%s%d", i ? "," : " ", ids[i]);
        printf("\n");
        int ok = 1;
        if (argc > 5) {
            int32_t ref[8192];
            int64_t nr = 0;
            for (char *p = argv[5]; *p && nr < 8192;) { ref[nr++] = (int32_t)strtol(p, &p, 10); if (*p == ',') p++; }
            ok = nr == n && !memcmp(ref, ids, n * 4);
            printf("PROMPT-PARITY %s (ours %ld, ref %ld)\n", ok ? "PASS" : "FAIL", (long)n, (long)nr);
        }
        double a = now();
        float *l = bz_eval(ids, n, 0, 1);
        double b = now();
        printf("prefill %ld tok %.2fs\n", (long)n, b - a);
        fflush(stdout);
        int64_t pos = n;
        int32_t out[256];
        double dec = 0;
        for (int i = 0; i < ngen; i++) {
            int32_t t = (int32_t)bz_argmax(l);
            out[i] = t;
            uint8_t pc[256];
            int64_t pl = bz_piece(t, pc, 255);
            pc[pl] = 0;
            printf("gen[%d] = %d '%s'\n", i, t, pc);
            top5(l, bz_stat(3));
            fflush(stdout);
            if (i + 1 == ngen) break;
            a = now();
            l = bz_eval(&t, 1, pos++, 1);
            b = now();
            dec += b - a;
            printf("   decode %.2fs\n", b - a);
        }
        printf("greedy");
        for (int i = 0; i < ngen; i++) printf("%s%d", i ? "," : " ", out[i]);
        printf("\n");
        if (ngen > 1) printf("decode %.3f tok/s (%d tokens, %.2fs)\n", (ngen - 1) / dec, ngen - 1, dec);
        return ok ? 0 : 1;
    }
    if (!strcmp(mode, "session") && argc > 3) {
        double a = now();
        int64_t n = bz_begin((const uint8_t *)argv[3], 0);
        double b = now();
        printf("ZLOUT-BEGIN prefill=%ld %.2fs\n", (long)n, b - a);
        uint8_t buf[256];
        int64_t k;
        int ng = 0;
        while ((k = bz_next(buf, 255)) >= 0) { buf[k] = 0; printf("%s", buf); fflush(stdout); ng++; }
        double c = now();
        bz_end();
        printf("\nZLOUT-END decode=%d tok %.2fs (%.3f tok/s) ctx=%ld\n", ng, c - b, ng / (c - b), (long)bz_stat(0));
        return 0;
    }
    fprintf(stderr, "bad mode\n");
    return 2;
}
