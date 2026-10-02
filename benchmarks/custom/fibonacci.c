/* fibonacci.c — recursive, call-heavy (tests inlining / stack optimization) */
#include <stdio.h>
#include <time.h>

long long fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(void) {
    int target = 42;
    clock_t s = clock();
    long long result = fib(target);
    clock_t e = clock();
    printf("fib(%d) = %lld  Time: %.4f s\n", target, result,
           (double)(e - s) / CLOCKS_PER_SEC);
    return 0;
}
