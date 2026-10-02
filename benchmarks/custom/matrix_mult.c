/* matrix_mult.c — dense matrix multiplication (compute-heavy, loop-heavy) */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 256

static double A[N][N], B[N][N], C[N][N];

void matrix_multiply(void) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];
}

void init(void) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(i * N + j) / (N * N);
            B[i][j] = (double)(j * N + i) / (N * N);
            C[i][j] = 0.0;
        }
}

int main(void) {
    init();
    clock_t start = clock();
    matrix_multiply();
    clock_t end   = clock();
    printf("Time: %.4f s  C[0][0]=%.6f\n",
           (double)(end - start) / CLOCKS_PER_SEC, C[0][0]);
    return 0;
}
