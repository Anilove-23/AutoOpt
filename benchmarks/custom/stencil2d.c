/* stencil2d.c — 2-D 5-point stencil, loop-heavy, memory-intensive (PolyBench-style) */
#include <stdio.h>
#include <time.h>

#define NX 512
#define NY 512
#define TSTEPS 20

static double A[NX][NY], B[NX][NY];

void init(void) {
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++) {
            A[i][j] = (double)(i * NY + j) / (NX * NY);
            B[i][j] = 0.0;
        }
}

void stencil(void) {
    for (int t = 0; t < TSTEPS; t++) {
        for (int i = 1; i < NX - 1; i++)
            for (int j = 1; j < NY - 1; j++)
                B[i][j] = 0.2 * (A[i][j] + A[i-1][j] + A[i+1][j]
                                           + A[i][j-1] + A[i][j+1]);
        for (int i = 1; i < NX - 1; i++)
            for (int j = 1; j < NY - 1; j++)
                A[i][j] = B[i][j];
    }
}

int main(void) {
    init();
    clock_t s = clock();
    stencil();
    clock_t e = clock();
    printf("A[NX/2][NY/2]=%.6f  Time: %.4f s\n",
           A[NX/2][NY/2], (double)(e - s) / CLOCKS_PER_SEC);
    return 0;
}
