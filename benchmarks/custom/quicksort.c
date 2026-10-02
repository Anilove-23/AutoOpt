/* quicksort.c — recursive divide-and-conquer, call-heavy + branch-heavy */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 500000

static int arr[N];

int partition(int *a, int lo, int hi) {
    int pivot = a[hi], i = lo - 1;
    for (int j = lo; j < hi; j++) {
        if (a[j] <= pivot) {
            i++;
            int tmp = a[i]; a[i] = a[j]; a[j] = tmp;
        }
    }
    int tmp = a[i + 1]; a[i + 1] = a[hi]; a[hi] = tmp;
    return i + 1;
}

void quicksort(int *a, int lo, int hi) {
    if (lo < hi) {
        int p = partition(a, lo, hi);
        quicksort(a, lo, p - 1);
        quicksort(a, p + 1, hi);
    }
}

int main(void) {
    srand(42);
    for (int i = 0; i < N; i++) arr[i] = rand();
    clock_t s = clock();
    quicksort(arr, 0, N - 1);
    clock_t e = clock();
    printf("arr[0]=%d arr[N-1]=%d  Time: %.4f s\n", arr[0], arr[N - 1],
           (double)(e - s) / CLOCKS_PER_SEC);
    return 0;
}
