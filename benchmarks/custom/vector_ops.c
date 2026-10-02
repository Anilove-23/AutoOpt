/* vector_ops.c — memory-intensive, SIMD-vectorization candidate */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1 << 22)   /* 4 million elements */

static float a[N], b[N], c[N];

void vec_add(float *restrict r, const float *restrict x,
             const float *restrict y, int n) {
    for (int i = 0; i < n; i++) r[i] = x[i] + y[i];
}

void vec_dot(double *out, const float *x, const float *y, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)x[i] * y[i];
    *out = acc;
}

int main(void) {
    for (int i = 0; i < N; i++) { a[i] = (float)i * 0.001f; b[i] = (float)(N - i) * 0.001f; }
    clock_t s = clock();
    vec_add(c, a, b, N);
    double dot;
    vec_dot(&dot, a, b, N);
    clock_t e = clock();
    printf("c[0]=%.3f dot=%.3e  Time: %.4f s\n", c[0], dot,
           (double)(e - s) / CLOCKS_PER_SEC);
    return 0;
}
