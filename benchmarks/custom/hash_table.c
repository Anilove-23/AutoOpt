/* hash_table.c — irregular memory, high branch count, pointer-heavy */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TABLE_SIZE 65537   /* prime */
#define NUM_OPS    300000

typedef struct Entry { unsigned key; int val; struct Entry *next; } Entry;

static Entry *table[TABLE_SIZE];

static unsigned hash(unsigned k) {
    k ^= k >> 16;
    k *= 0x45d9f3b;
    k ^= k >> 16;
    return k % TABLE_SIZE;
}

void ht_insert(unsigned k, int v) {
    unsigned h = hash(k);
    Entry *e = (Entry *)malloc(sizeof(Entry));
    e->key = k; e->val = v; e->next = table[h]; table[h] = e;
}

int ht_lookup(unsigned k, int *out) {
    for (Entry *e = table[hash(k)]; e; e = e->next)
        if (e->key == k) { *out = e->val; return 1; }
    return 0;
}

int main(void) {
    srand(99);
    clock_t s = clock();
    for (int i = 0; i < NUM_OPS; i++) ht_insert((unsigned)rand(), i);
    int hits = 0, v;
    for (int i = 0; i < NUM_OPS; i++) if (ht_lookup((unsigned)rand(), &v)) hits++;
    clock_t e = clock();
    printf("Hits=%d  Time: %.4f s\n", hits,
           (double)(e - s) / CLOCKS_PER_SEC);
    return 0;
}
