/* string_ops.c — control-flow-heavy, irregular memory access */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define STR_LEN  512
#define NUM_STRS 20000

static char strings[NUM_STRS][STR_LEN];

void gen_strings(void) {
    static const char charset[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    srand(7);
    for (int i = 0; i < NUM_STRS; i++) {
        int len = 10 + rand() % (STR_LEN - 10);
        for (int j = 0; j < len; j++)
            strings[i][j] = charset[rand() % (sizeof charset - 1)];
        strings[i][len] = '\0';
    }
}

int count_vowels(const char *s) {
    int cnt = 0;
    for (; *s; s++)
        switch (*s) {
            case 'a': case 'e': case 'i': case 'o': case 'u': cnt++; break;
        }
    return cnt;
}

int main(void) {
    gen_strings();
    clock_t st = clock();
    long total = 0;
    for (int i = 0; i < NUM_STRS; i++) total += count_vowels(strings[i]);
    clock_t en = clock();
    printf("Total vowels: %ld  Time: %.4f s\n", total,
           (double)(en - st) / CLOCKS_PER_SEC);
    return 0;
}
