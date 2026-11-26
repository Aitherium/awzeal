/* freestd.c -- the four libc symbols a freestanding GCC build may still call
 * (struct copies and clears). Linked only into the flat ZealOS image. */
#include <stddef.h>

void *memcpy(void *d, const void *s, size_t n) {
    void *r = d;
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(n) : : "memory");
    return r;
}

void *memmove(void *d, const void *s, size_t n) {
    unsigned char *dp = d;
    const unsigned char *sp = s;
    if (dp == sp || n == 0) return d;
    if (dp < sp || dp >= sp + n) return memcpy(d, s, n);
    while (n--) dp[n] = sp[n];
    return d;
}

void *memset(void *d, int c, size_t n) {
    void *r = d;
    __asm__ volatile("rep stosb" : "+D"(d), "+c"(n) : "a"(c) : "memory");
    return r;
}

int memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *x = a, *y = b;
    for (size_t i = 0; i < n; i++)
        if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
