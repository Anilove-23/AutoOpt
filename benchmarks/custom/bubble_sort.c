/* bubble_sort.c — branch-heavy, memory-sequential */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 8000

static int arr[N];

void bubble_sort(int *a, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (a[j] > a[j + 1]) {
                int tmp = a[j];
                a[j]     = a[j + 1];
                a[j + 1] = tmp;
            }
}

int main(void) {
    srand(42);
    for (int i = 0; i < N; i++) arr[i] = rand() % N;
    clock_t s = clock();
    bubble_sort(arr, N);
    clock_t e = clock();
    printf("Time: %.4f s  arr[0]=%d\n",
           (double)(e - s) / CLOCKS_PER_SEC, arr[0]);
    return 0;
}
