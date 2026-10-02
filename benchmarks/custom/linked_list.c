/* linked_list.c — pointer-chasing, irregular memory, bad for vectorization */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 200000

typedef struct Node { int val; struct Node *next; } Node;

Node *build_list(int n) {
    Node *head = NULL;
    for (int i = n - 1; i >= 0; i--) {
        Node *nd = (Node *)malloc(sizeof(Node));
        nd->val  = i;
        nd->next = head;
        head     = nd;
    }
    return head;
}

long sum_list(Node *h) {
    long s = 0;
    while (h) { s += h->val; h = h->next; }
    return s;
}

void free_list(Node *h) {
    while (h) { Node *t = h->next; free(h); h = t; }
}

int main(void) {
    clock_t s = clock();
    Node *list = build_list(N);
    long  sum  = sum_list(list);
    clock_t e  = clock();
    free_list(list);
    printf("Sum=%ld  Time: %.4f s\n", sum,
           (double)(e - s) / CLOCKS_PER_SEC);
    return 0;
}
