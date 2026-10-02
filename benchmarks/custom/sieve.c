/* sieve.c — Sieve of Eratosthenes — bit-manipulation + loop-heavy */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LIMIT 10000000

int main(void) {
    char *composite = (char *)calloc(LIMIT + 1, 1);
    clock_t s = clock();
    for (int i = 2; (long long)i * i <= LIMIT; i++)
        if (!composite[i])
            for (int j = i * i; j <= LIMIT; j += i)
                composite[j] = 1;
    int count = 0;
    for (int i = 2; i <= LIMIT; i++) if (!composite[i]) count++;
    clock_t e = clock();
    printf("Primes up to %d: %d  Time: %.4f s\n", LIMIT, count,
           (double)(e - s) / CLOCKS_PER_SEC);
    free(composite);
    return 0;
}
