"""
scripts/generate_programs.py
============================
Generates N random C++ programs covering 15 DSA algorithm categories.
Each program is a complete, self-contained, compilable .cpp file with:
  - A randomized algorithm implementation
  - Randomized parameters (array size, data type, variant, etc.)
  - main() with timing via clock() and a result print (prevents dead-code elimination)

DSA Categories & Variants
--------------------------
 1. Bubble Sort            8. Dynamic Programming (8 sub-variants)
 2. Insertion Sort         9. String Pattern Matching (4 sub-variants)
 3. Selection Sort        10. Mathematical Operations
 4. Merge Sort            11. Matrix Operations
 5. Quick Sort            12. Heap Operations
 6. Linked List           13. Graph Algorithms (BFS/DFS/Dijkstra)
 7. Hash Table            14. Binary Search Tree
                          15. Divide & Conquer / Miscellaneous

Usage:
    python scripts/generate_programs.py --count 10000 --out benchmarks/generated/
"""

from __future__ import annotations
import argparse
import os
import random
import string
from pathlib import Path
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Helper to pick random parameters
# ─────────────────────────────────────────────────────────────────────────────

RNG = random.Random()   # seeded per-program

SIZES       = [100, 256, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
INT_TYPES   = ["int", "long long", "unsigned int", "unsigned long long"]
FLOAT_TYPES = ["float", "double"]
ALL_TYPES   = INT_TYPES + FLOAT_TYPES


def rsize(lo=100, hi=50000):   return RNG.choice([s for s in SIZES if lo <= s <= hi])
def rtype(cat="int"):
    if cat == "int":   return RNG.choice(INT_TYPES)
    if cat == "float": return RNG.choice(FLOAT_TYPES)
    return RNG.choice(ALL_TYPES)

HEADER = """\
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <stack>
#include <string>
#include <functional>
#include <climits>
#include <numeric>
using namespace std;
"""

TIMER_START = "    clock_t _t0 = clock();"
TIMER_END   = """\
    clock_t _t1 = clock();
    cout << "time=" << (double)(_t1-_t0)/CLOCKS_PER_SEC << " result=" << _result << endl;
"""

def wrap_main(setup: str, algo: str, result_expr: str, extra_decl: str = "") -> str:
    return f"""\
int main() {{
    srand(42);
{extra_decl}
{setup}
    {TIMER_START.strip()}
{algo}
    long long _result = (long long)({result_expr});
{TIMER_END}    return 0;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUBBLE SORT
# ─────────────────────────────────────────────────────────────────────────────

def gen_bubble_sort(pid: int) -> str:
    n  = rsize(200, 20000)
    T  = rtype("int")
    return HEADER + f"""
void bubbleSort({T}* a, int n) {{
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n-i-1; j++)
            if (a[j] > a[j+1]) {{ {T} t=a[j]; a[j]=a[j+1]; a[j+1]=t; }}
}}
""" + wrap_main(
    f"    const int N={n}; {T} a[{n}];\n    for(int i=0;i<N;i++) a[i]=({T})(rand()%{n});",
    "    bubbleSort(a,N);",
    "a[0]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. INSERTION SORT
# ─────────────────────────────────────────────────────────────────────────────

def gen_insertion_sort(pid: int) -> str:
    n = rsize(200, 30000)
    T = rtype("int")
    return HEADER + f"""
void insertionSort({T}* a, int n) {{
    for (int i=1; i<n; i++) {{
        {T} key = a[i]; int j=i-1;
        while (j>=0 && a[j]>key) {{ a[j+1]=a[j]; j--; }}
        a[j+1] = key;
    }}
}}
""" + wrap_main(
    f"    const int N={n}; {T} a[{n}];\n    for(int i=0;i<N;i++) a[i]=({T})(rand()%{n});",
    "    insertionSort(a,N);",
    "a[0]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELECTION SORT
# ─────────────────────────────────────────────────────────────────────────────

def gen_selection_sort(pid: int) -> str:
    n = rsize(100, 15000)
    T = rtype("int")
    return HEADER + f"""
void selectionSort({T}* a, int n) {{
    for (int i=0; i<n-1; i++) {{
        int m=i;
        for (int j=i+1; j<n; j++) if (a[j]<a[m]) m=j;
        {T} t=a[i]; a[i]=a[m]; a[m]=t;
    }}
}}
""" + wrap_main(
    f"    const int N={n}; {T} a[{n}];\n    for(int i=0;i<N;i++) a[i]=({T})(rand()%{n});",
    "    selectionSort(a,N);",
    "a[0]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MERGE SORT
# ─────────────────────────────────────────────────────────────────────────────

def gen_merge_sort(pid: int) -> str:
    n = rsize(1000, 100000)
    T = rtype("int")
    return HEADER + f"""
void merge({T}* a, int l, int m, int r) {{
    int n1=m-l+1, n2=r-m;
    {T} L[n1], R[n2];
    for(int i=0;i<n1;i++) L[i]=a[l+i];
    for(int j=0;j<n2;j++) R[j]=a[m+1+j];
    int i=0,j=0,k=l;
    while(i<n1&&j<n2) a[k++]=(L[i]<=R[j])?L[i++]:R[j++];
    while(i<n1) a[k++]=L[i++];
    while(j<n2) a[k++]=R[j++];
}}
void mergeSort({T}* a, int l, int r) {{
    if (l<r) {{ int m=l+(r-l)/2; mergeSort(a,l,m); mergeSort(a,m+1,r); merge(a,l,m,r); }}
}}
""" + wrap_main(
    f"    const int N={n}; vector<{T}> a(N);\n    for(int i=0;i<N;i++) a[i]=({T})(rand()%N);",
    "    mergeSort(a.data(),0,N-1);",
    "a[0]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. QUICK SORT
# ─────────────────────────────────────────────────────────────────────────────

def gen_quick_sort(pid: int) -> str:
    n     = rsize(1000, 100000)
    T     = rtype("int")
    pivot = RNG.choice(["last", "first", "median"])
    pivot_code = {
        "last":   f"    {T} p=a[hi]; int i=lo-1;",
        "first":  f"    swap(a[lo],a[lo+(hi-lo)/2]); {T} p=a[hi]; int i=lo-1;",
        "median": f"    int m=lo+(hi-lo)/2; if(a[m]<a[lo]) swap(a[lo],a[m]); if(a[hi]<a[lo]) swap(a[lo],a[hi]); if(a[m]<a[hi]) swap(a[m],a[hi]); {T} p=a[hi]; int i=lo-1;",
    }[pivot]
    return HEADER + f"""
void quickSort({T}* a, int lo, int hi) {{
    if (lo>=hi) return;
    {pivot_code}
    for(int j=lo;j<hi;j++) if(a[j]<=p) {{ i++; swap(a[i],a[j]); }}
    swap(a[i+1],a[hi]); int pi=i+1;
    quickSort(a,lo,pi-1); quickSort(a,pi+1,hi);
}}
""" + wrap_main(
    f"    const int N={n}; vector<{T}> a(N);\n    for(int i=0;i<N;i++) a[i]=({T})(rand()%N);",
    "    quickSort(a.data(),0,N-1);",
    "a[0]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. HEAP SORT
# ─────────────────────────────────────────────────────────────────────────────

def gen_heap_sort(pid: int) -> str:
    n = rsize(1000, 100000)
    T = rtype("int")
    return HEADER + f"""
void heapify({T}* a, int n, int i) {{
    int lg=i, l=2*i+1, r=2*i+2;
    if(l<n&&a[l]>a[lg]) lg=l;
    if(r<n&&a[r]>a[lg]) lg=r;
    if(lg!=i) {{ swap(a[i],a[lg]); heapify(a,n,lg); }}
}}
void heapSort({T}* a, int n) {{
    for(int i=n/2-1;i>=0;i--) heapify(a,n,i);
    for(int i=n-1;i>0;i--) {{ swap(a[0],a[i]); heapify(a,i,0); }}
}}
""" + wrap_main(
    f"    const int N={n}; vector<{T}> a(N);\n    for(int i=0;i<N;i++) a[i]=({T})(rand()%N);",
    "    heapSort(a.data(),N);",
    "a[0]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 7. RADIX SORT (int only)
# ─────────────────────────────────────────────────────────────────────────────

def gen_radix_sort(pid: int) -> str:
    n = rsize(1000, 100000)
    return HEADER + """
void countSort(int* a, int n, int exp) {
    vector<int> out(n); int cnt[10]={};
    for(int i=0;i<n;i++) cnt[(a[i]/exp)%10]++;
    for(int i=1;i<10;i++) cnt[i]+=cnt[i-1];
    for(int i=n-1;i>=0;i--) { out[--cnt[(a[i]/exp)%10]]=a[i]; }
    for(int i=0;i<n;i++) a[i]=out[i];
}
void radixSort(int* a, int n) {
    int mx=*max_element(a,a+n);
    for(int e=1;mx/e>0;e*=10) countSort(a,n,e);
}
""" + wrap_main(
    f"    const int N={n}; vector<int> a(N);\n    for(int i=0;i<N;i++) a[i]=rand()%{n};",
    "    radixSort(a.data(),N);",
    "a[0]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 8. COUNTING SORT
# ─────────────────────────────────────────────────────────────────────────────

def gen_counting_sort(pid: int) -> str:
    n     = rsize(1000, 100000)
    maxv  = RNG.choice([256, 1000, 10000, 65536])
    return HEADER + f"""
void countingSort(int* a, int n, int maxv) {{
    vector<int> cnt(maxv+1,0), out(n);
    for(int i=0;i<n;i++) cnt[a[i]]++;
    for(int i=1;i<=maxv;i++) cnt[i]+=cnt[i-1];
    for(int i=n-1;i>=0;i--) out[--cnt[a[i]]]=a[i];
    for(int i=0;i<n;i++) a[i]=out[i];
}}
""" + wrap_main(
    f"    const int N={n}; vector<int> a(N);\n    for(int i=0;i<N;i++) a[i]=rand()%{maxv};",
    f"    countingSort(a.data(),N,{maxv});",
    "a[0]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 9. BINARY SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def gen_binary_search(pid: int) -> str:
    n    = rsize(1000, 100000)
    reps = RNG.randint(10000, 500000)
    T    = rtype("int")
    return HEADER + f"""
int binarySearch({T}* a, int n, {T} x) {{
    int lo=0, hi=n-1;
    while(lo<=hi) {{
        int m=lo+(hi-lo)/2;
        if(a[m]==x) return m;
        if(a[m]<x) lo=m+1; else hi=m-1;
    }}
    return -1;
}}
""" + wrap_main(
    f"    const int N={n}, REPS={reps}; vector<{T}> a(N);\n    for(int i=0;i<N;i++) a[i]=({T})(i*2);\n    long long acc=0;",
    f"    for(int r=0;r<REPS;r++) acc+=binarySearch(a.data(),N,({T})(rand()%N*2));",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SINGLY LINKED LIST
# ─────────────────────────────────────────────────────────────────────────────

def gen_linked_list(pid: int) -> str:
    n    = rsize(1000, 50000)
    ops  = RNG.randint(n, n*3)
    variant = RNG.choice(["sum", "reverse", "search"])
    if variant == "sum":
        op_code = "    long long s=0; Node* p=head; while(p){s+=p->val;p=p->next;} long long res=s;"
    elif variant == "reverse":
        op_code = "    Node *p=head,*pr=nullptr,*nx;\n    while(p){nx=p->next;p->next=pr;pr=p;p=nx;}\n    head=pr; long long res=head?head->val:0;"
    else:
        half = n // 2
        op_code = f"    int target={half}; Node* p=head; long long res=0; while(p){{if(p->val==target)res=p->val;p=p->next;}}"
    return HEADER + f"""
struct Node {{ int val; Node* next; Node(int v):val(v),next(nullptr){{}} }};
""" + wrap_main(
    f"    const int N={n}; Node* head=nullptr;\n    for(int i=N-1;i>=0;i--){{Node*nd=new Node(i);nd->next=head;head=nd;}}",
    op_code,
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 11. STACK (array-based)
# ─────────────────────────────────────────────────────────────────────────────

def gen_stack(pid: int) -> str:
    n    = rsize(1000, 100000)
    ops  = n * RNG.randint(2, 10)
    return HEADER + f"""
struct Stack {{
    int* data; int top, cap;
    Stack(int c):top(-1),cap(c){{data=new int[c];}}
    ~Stack(){{delete[]data;}}
    void push(int x){{if(top<cap-1)data[++top]=x;}}
    int  pop(){{return top>=0?data[top--]:-1;}}
    bool empty(){{return top<0;}}
}};
""" + wrap_main(
    f"    const int N={n}; Stack st(N); long long acc=0;",
    f"    for(int i=0;i<{ops};i++){{\n        if(i%3!=0||st.empty()) st.push(rand()%N);\n        else acc+=st.pop();\n    }}",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 12. QUEUE (circular array)
# ─────────────────────────────────────────────────────────────────────────────

def gen_queue(pid: int) -> str:
    n   = rsize(1000, 50000)
    ops = n * RNG.randint(2, 8)
    return HEADER + f"""
struct Queue {{
    int* data; int front, back, sz, cap;
    Queue(int c):front(0),back(0),sz(0),cap(c){{data=new int[c];}}
    ~Queue(){{delete[]data;}}
    void enqueue(int x){{if(sz<cap){{data[back]= x;back=(back+1)%cap;sz++;}}}}
    int  dequeue(){{if(!sz)return -1;int v=data[front];front=(front+1)%cap;sz--;return v;}}
}};
""" + wrap_main(
    f"    const int N={n}; Queue q(N); long long acc=0;",
    f"    for(int i=0;i<{ops};i++){{\n        if(i%3!=0||q.sz==0) q.enqueue(rand()%N);\n        else acc+=q.dequeue();\n    }}",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 13. BINARY SEARCH TREE
# ─────────────────────────────────────────────────────────────────────────────

def gen_bst(pid: int) -> str:
    n   = rsize(500, 20000)
    ops = RNG.randint(n, n*3)
    return HEADER + f"""
struct BSTNode {{ int val; BSTNode *l,*r; BSTNode(int v):val(v),l(nullptr),r(nullptr){{}} }};
BSTNode* insert(BSTNode* root, int v) {{
    if(!root) return new BSTNode(v);
    if(v<root->val) root->l=insert(root->l,v);
    else root->r=insert(root->r,v);
    return root;
}}
bool search(BSTNode* root, int v) {{
    while(root) {{ if(v==root->val) return true; root=v<root->val?root->l:root->r; }}
    return false;
}}
long long inorder_sum(BSTNode* root) {{
    if(!root) return 0;
    return inorder_sum(root->l)+root->val+inorder_sum(root->r);
}}
""" + wrap_main(
    f"    const int N={n}; BSTNode* root=nullptr;\n    for(int i=0;i<N;i++) root=insert(root,rand()%N);\n    long long acc=0;",
    f"    for(int i=0;i<{ops};i++) acc+=search(root,rand()%N)?1:0;\n    acc+=inorder_sum(root);",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 14. HASH TABLE (chaining)
# ─────────────────────────────────────────────────────────────────────────────

def gen_hash_table(pid: int) -> str:
    buckets = RNG.choice([997, 4999, 9973, 49999, 99991])
    n       = rsize(1000, 50000)
    ops     = n * RNG.randint(2, 5)
    return HEADER + f"""
const int BUCKETS={buckets};
struct HNode {{ int key,val; HNode* next; HNode(int k,int v):key(k),val(v),next(nullptr){{}} }};
HNode* table[BUCKETS]={{}};
unsigned hfn(int k){{return ((unsigned)k*2654435761u)%BUCKETS;}}
void hinsert(int k,int v){{unsigned h=hfn(k);HNode*nd=new HNode(k,v);nd->next=table[h];table[h]=nd;}}
int hlookup(int k){{for(HNode*n=table[hfn(k)];n;n=n->next)if(n->key==k)return n->val;return -1;}}
""" + wrap_main(
    f"    for(int i=0;i<{n};i++) hinsert(rand()%{ops},i); long long acc=0;",
    f"    for(int i=0;i<{ops};i++) acc+=hlookup(rand()%{ops});",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 15. PRIORITY QUEUE / MIN-HEAP
# ─────────────────────────────────────────────────────────────────────────────

def gen_priority_queue(pid: int) -> str:
    n   = rsize(1000, 100000)
    ops = n * RNG.randint(2, 6)
    return HEADER + f"""
struct MinHeap {{
    vector<int> h;
    void push(int x){{h.push_back(x);push_heap(h.begin(),h.end(),greater<int>());}}
    int  top(){{return h.empty()?-1:h.front();}}
    void pop(){{if(!h.empty()){{pop_heap(h.begin(),h.end(),greater<int>());h.pop_back();}}}}
    bool empty(){{return h.empty();}}
}};
""" + wrap_main(
    f"    MinHeap pq; long long acc=0;",
    f"    for(int i=0;i<{ops};i++){{\n        if(i%3!=0||pq.empty()) pq.push(rand()%{n});\n        else {{ acc+=pq.top(); pq.pop(); }}\n    }}",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 16. BFS
# ─────────────────────────────────────────────────────────────────────────────

def gen_bfs(pid: int) -> str:
    v   = RNG.choice([100, 200, 500, 1000, 2000, 5000])
    e   = min(v * RNG.randint(2, 8), v*(v-1)//2)
    return HEADER + f"""
const int V={v};
vector<int> adj[V];
long long bfs(int src) {{
    vector<bool> vis(V,false); queue<int> q; long long dist=0;
    vis[src]=true; q.push(src);
    while(!q.empty()) {{
        int u=q.front(); q.pop();
        for(int nb:adj[u]) if(!vis[nb]) {{ vis[nb]=true; q.push(nb); dist++; }}
    }}
    return dist;
}}
""" + wrap_main(
    f"    for(int i=0;i<{e};i++){{\n        int u=rand()%V, v=rand()%V;\n        if(u!=v){{adj[u].push_back(v);adj[v].push_back(u);}}\n    }}",
    "    long long res=bfs(0);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 17. DFS
# ─────────────────────────────────────────────────────────────────────────────

def gen_dfs(pid: int) -> str:
    v   = RNG.choice([100, 200, 500, 1000, 2000, 5000])
    e   = min(v * RNG.randint(2, 6), v*(v-1)//2)
    return HEADER + f"""
const int VV={v};
vector<int> gadj[VV];
long long dfs_count=0;
void dfs(int u, vector<bool>& vis) {{
    vis[u]=true; dfs_count++;
    for(int nb:gadj[u]) if(!vis[nb]) dfs(nb,vis);
}}
""" + wrap_main(
    f"    for(int i=0;i<{e};i++){{\n        int u=rand()%VV, vv=rand()%VV;\n        if(u!=vv){{gadj[u].push_back(vv);gadj[vv].push_back(u);}}\n    }}\n    vector<bool> vis(VV,false);",
    "    dfs(0,vis);",
    "dfs_count"
)


# ─────────────────────────────────────────────────────────────────────────────
# 18. DIJKSTRA
# ─────────────────────────────────────────────────────────────────────────────

def gen_dijkstra(pid: int) -> str:
    v = RNG.choice([100, 200, 500, 1000, 2000])
    e = min(v * RNG.randint(3, 10), v*(v-1))
    return HEADER + f"""
const int DV={v};
vector<pair<int,int>> dadj[DV];
long long dijkstra(int src) {{
    vector<long long> dist(DV,LLONG_MAX);
    priority_queue<pair<long long,int>,vector<pair<long long,int>>,greater<>> pq;
    dist[src]=0; pq.push({{0,src}});
    while(!pq.empty()) {{
        long long d=pq.top().first; int u=pq.top().second; pq.pop();
        if(d>dist[u]) continue;
        for(int _i=0;_i<(int)dadj[u].size();_i++) {{ int w=dadj[u][_i].first,nb=dadj[u][_i].second; if(dist[u]+w<dist[nb]) {{
            dist[nb]=dist[u]+w; pq.push({{dist[nb],nb}}); }}
        }}
    }}
    long long s=0; for(auto x:dist) if(x!=LLONG_MAX) s+=x;
    return s;
}}
""" + wrap_main(
    f"    for(int i=0;i<{e};i++){{\n        int u=rand()%DV, vv=rand()%DV, w=rand()%100+1;\n        if(u!=vv){{dadj[u].push_back({{w,vv}});dadj[vv].push_back({{w,u}});}}\n    }}",
    "    long long res=dijkstra(0);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 19. FLOYD-WARSHALL
# ─────────────────────────────────────────────────────────────────────────────

def gen_floyd_warshall(pid: int) -> str:
    v = RNG.choice([50, 100, 150, 200, 300])
    return HEADER + f"""
const int FV={v};
long long dist[FV][FV];
void floydWarshall() {{
    for(int k=0;k<FV;k++) for(int i=0;i<FV;i++) for(int j=0;j<FV;j++)
        if(dist[i][k]!=LLONG_MAX&&dist[k][j]!=LLONG_MAX)
            dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
}}
""" + wrap_main(
    f"    for(int i=0;i<FV;i++) for(int j=0;j<FV;j++) dist[i][j]=(i==j)?0:(rand()%2?rand()%100+1:LLONG_MAX);\n    long long res=0;",
    "    floydWarshall(); for(int i=0;i<FV;i++) if(dist[0][i]!=LLONG_MAX) res+=dist[0][i];",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 20. 0/1 KNAPSACK (DP)
# ─────────────────────────────────────────────────────────────────────────────

def gen_knapsack(pid: int) -> str:
    n  = RNG.randint(10, 200)
    W  = RNG.randint(100, 5000)
    return HEADER + f"""
int knapsack(int* wt, int* val, int n, int W) {{
    vector<vector<int>> dp(n+1,vector<int>(W+1,0));
    for(int i=1;i<=n;i++) for(int w=0;w<=W;w++) {{
        dp[i][w]=dp[i-1][w];
        if(wt[i-1]<=w) dp[i][w]=max(dp[i][w],dp[i-1][w-wt[i-1]]+val[i-1]);
    }}
    return dp[n][W];
}}
""" + wrap_main(
    f"    const int N={n}, CAP={W}; int wt[{n}], val[{n}];\n    for(int i=0;i<N;i++){{wt[i]=rand()%{max(1,W//n)}+1;val[i]=rand()%100+1;}}",
    "    long long res=knapsack(wt,val,N,CAP);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 21. LCS (DP)
# ─────────────────────────────────────────────────────────────────────────────

def gen_lcs(pid: int) -> str:
    n = RNG.randint(100, 2000)
    m = RNG.randint(100, 2000)
    chars = RNG.randint(4, 26)
    return HEADER + f"""
int lcs(string& a, string& b) {{
    int n=a.size(), m=b.size();
    vector<vector<int>> dp(n+1,vector<int>(m+1,0));
    for(int i=1;i<=n;i++) for(int j=1;j<=m;j++)
        dp[i][j]=(a[i-1]==b[j-1])?dp[i-1][j-1]+1:max(dp[i-1][j],dp[i][j-1]);
    return dp[n][m];
}}
""" + wrap_main(
    f"    const int N={n},M={m},C={chars}; string a(N,'a'),b(M,'a');\n    for(auto& c:a) c='a'+rand()%C;\n    for(auto& c:b) c='a'+rand()%C;",
    "    long long res=lcs(a,b);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 22. LONGEST INCREASING SUBSEQUENCE (DP)
# ─────────────────────────────────────────────────────────────────────────────

def gen_lis(pid: int) -> str:
    n = rsize(500, 20000)
    return HEADER + f"""
int lis(int* a, int n) {{
    vector<int> dp;
    for(int i=0;i<n;i++) {{
        auto it=lower_bound(dp.begin(),dp.end(),a[i]);
        if(it==dp.end()) dp.push_back(a[i]);
        else *it=a[i];
    }}
    return dp.size();
}}
""" + wrap_main(
    f"    const int N={n}; vector<int> a(N);\n    for(int i=0;i<N;i++) a[i]=rand()%N;",
    "    long long res=lis(a.data(),N);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 23. COIN CHANGE (DP)
# ─────────────────────────────────────────────────────────────────────────────

def gen_coin_change(pid: int) -> str:
    nc     = RNG.randint(3, 20)
    amount = RNG.randint(100, 5000)
    reps   = RNG.randint(1, 50)
    return HEADER + f"""
long long coinChange(int* coins, int nc, int amount) {{
    vector<long long> dp(amount+1,LLONG_MAX/2);
    dp[0]=0;
    for(int i=1;i<=amount;i++) for(int j=0;j<nc;j++)
        if(coins[j]<=i&&dp[i-coins[j]]+1<dp[i]) dp[i]=dp[i-coins[j]]+1;
    return dp[amount]==(LLONG_MAX/2)?-1:dp[amount];
}}
""" + wrap_main(
    f"    const int NC={nc}, AMT={amount}; int coins[{nc}];\n    for(int i=0;i<NC;i++) coins[i]=rand()%50+1; long long acc=0;",
    f"    for(int r=0;r<{reps};r++) acc+=coinChange(coins,NC,AMT);",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 24. EDIT DISTANCE (DP)
# ─────────────────────────────────────────────────────────────────────────────

def gen_edit_distance(pid: int) -> str:
    n = RNG.randint(50, 1000)
    m = RNG.randint(50, 1000)
    chars = RNG.randint(2, 10)
    return HEADER + f"""
int editDist(string& a, string& b) {{
    int n=a.size(),m=b.size();
    vector<vector<int>> dp(n+1,vector<int>(m+1));
    for(int i=0;i<=n;i++) dp[i][0]=i;
    for(int j=0;j<=m;j++) dp[0][j]=j;
    for(int i=1;i<=n;i++) for(int j=1;j<=m;j++) {{
        if(a[i-1]==b[j-1]) dp[i][j]=dp[i-1][j-1];
        else dp[i][j]=1+min({{dp[i-1][j],dp[i][j-1],dp[i-1][j-1]}});
    }}
    return dp[n][m];
}}
""" + wrap_main(
    f"    const int N={n},M={m},C={chars}; string a(N,'a'),b(M,'a');\n    for(auto& c:a) c='a'+rand()%C;\n    for(auto& c:b) c='a'+rand()%C;",
    "    long long res=editDist(a,b);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 25. MATRIX MULTIPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def gen_matrix_mult(pid: int) -> str:
    n  = RNG.choice([32, 64, 128, 256, 512])
    T  = rtype("float")
    return HEADER + f"""
const int MN={n};
{T} A[MN][MN], B[MN][MN], C[MN][MN];
void matmul() {{
    for(int i=0;i<MN;i++) for(int k=0;k<MN;k++) for(int j=0;j<MN;j++)
        C[i][j]+=A[i][k]*B[k][j];
}}
""" + wrap_main(
    f"    for(int i=0;i<MN;i++) for(int j=0;j<MN;j++) {{ A[i][j]=({T})(rand()%100); B[i][j]=({T})(rand()%100); C[i][j]=0; }}",
    "    matmul();",
    "(long long)C[0][0]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 26. PRIME SIEVE
# ─────────────────────────────────────────────────────────────────────────────

def gen_sieve(pid: int) -> str:
    limit = RNG.choice([100000, 500000, 1000000, 5000000, 10000000])
    reps  = RNG.randint(1, 5)
    return HEADER + f"""
int sieve(int limit) {{
    vector<bool> is_prime(limit+1,true);
    is_prime[0]=is_prime[1]=false;
    for(int i=2;(long long)i*i<=limit;i++)
        if(is_prime[i]) for(int j=i*i;j<=limit;j+=i) is_prime[j]=false;
    int cnt=0; for(int i=2;i<=limit;i++) if(is_prime[i]) cnt++;
    return cnt;
}}
""" + wrap_main(
    f"    long long acc=0;",
    f"    for(int r=0;r<{reps};r++) acc+=sieve({limit});",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 27. STENCIL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def gen_stencil(pid: int) -> str:
    n  = RNG.choice([64, 128, 256, 512])
    ts = RNG.randint(5, 50)
    T  = rtype("float")
    return HEADER + f"""
const int SN={n};
{T} grid[SN][SN], tmp[SN][SN];
void stencil(int steps) {{
    for(int t=0;t<steps;t++) {{
        for(int i=1;i<SN-1;i++) for(int j=1;j<SN-1;j++)
            tmp[i][j]=0.25f*(grid[i-1][j]+grid[i+1][j]+grid[i][j-1]+grid[i][j+1]);
        for(int i=1;i<SN-1;i++) for(int j=1;j<SN-1;j++) grid[i][j]=tmp[i][j];
    }}
}}
""" + wrap_main(
    f"    for(int i=0;i<SN;i++) for(int j=0;j<SN;j++) grid[i][j]=({T})(rand()%100);",
    f"    stencil({ts});",
    "(long long)grid[SN/2][SN/2]"
)


# ─────────────────────────────────────────────────────────────────────────────
# 28. KMP STRING MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def gen_kmp(pid: int) -> str:
    text_len    = RNG.randint(10000, 200000)
    pat_len     = RNG.randint(3, 100)
    chars       = RNG.randint(2, 26)
    return HEADER + f"""
int kmpSearch(const string& text, const string& pat) {{
    int n=text.size(), m=pat.size();
    vector<int> lps(m,0); int cnt=0;
    for(int i=1,k=0;i<m;) {{
        if(pat[i]==pat[k]) {{ lps[i++]=++k; }}
        else if(k) k=lps[k-1]; else lps[i++]=0;
    }}
    for(int i=0,j=0;i<n;) {{
        if(text[i]==pat[j]) {{ i++; j++; }}
        if(j==m) {{ cnt++; j=lps[j-1]; }}
        else if(i<n&&text[i]!=pat[j]) {{ if(j) j=lps[j-1]; else i++; }}
    }}
    return cnt;
}}
""" + wrap_main(
    f"    const int TL={text_len},PL={pat_len},C={chars};\n    string text(TL,'a'), pat(PL,'a');\n    for(auto& c:text) c='a'+rand()%C;\n    for(auto& c:pat)  c='a'+rand()%C;",
    "    long long res=kmpSearch(text,pat);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 29. VECTOR OPS (SIMD candidate)
# ─────────────────────────────────────────────────────────────────────────────

def gen_vector_ops(pid: int) -> str:
    n  = rsize(100000, 100000)
    T  = rtype("float")
    op = RNG.choice(["add", "dot", "scale", "norm"])
    if op == "add":
        kernel = f"    for(int i=0;i<N;i++) c[i]=a[i]+b[i]; (void)c[0];"
        result = "c[0]"
    elif op == "dot":
        kernel = f"    {T} acc=0; for(int i=0;i<N;i++) acc+=a[i]*b[i];"
        result = "(long long)acc"
    elif op == "scale":
        kernel = f"    {T} s=1.5f; for(int i=0;i<N;i++) a[i]*=s;"
        result = "(long long)a[0]"
    else:
        kernel = f"    {T} norm=0; for(int i=0;i<N;i++) norm+=a[i]*a[i]; norm=sqrt(norm);"
        result = "(long long)norm"
    return HEADER + f"""
""" + wrap_main(
    f"    const int N={n}; vector<{T}> a(N),b(N),c(N);\n    for(int i=0;i<N;i++){{a[i]=({T})(rand()%1000)*0.001f;b[i]=({T})(rand()%1000)*0.001f;}}",
    kernel,
    result
)


# ─────────────────────────────────────────────────────────────────────────────
# 30. FIBONACCI (recursive + memoized + iterative variants)
# ─────────────────────────────────────────────────────────────────────────────

def gen_fibonacci(pid: int) -> str:
    variant = RNG.choice(["recursive", "memoized", "iterative", "matrix"])
    if variant == "recursive":
        n   = RNG.randint(25, 38)
        code = f"""
long long fib(int n) {{ return n<=1?n:fib(n-1)+fib(n-2); }}
"""
        call = f"    long long res=fib({n});"
    elif variant == "memoized":
        n    = RNG.randint(50, 1000)
        code = f"""
long long memo[{n+1}];
long long fib(int n) {{ if(n<=1) return n; if(memo[n]) return memo[n]; return memo[n]=fib(n-1)+fib(n-2); }}
"""
        call = f"    memset(memo,0,sizeof(memo)); long long res=fib({n});"
    elif variant == "iterative":
        n    = rsize(1000, 100000)
        reps = RNG.randint(100, 5000)
        code = ""
        call = f"    long long res=0,a=0,b=1;\n    for(int r=0;r<{reps};r++){{a=0;b=1;for(int i=0;i<{n};i++){{long long c=(a+b)%1000000007;a=b;b=c;}}res=b;}}"
    else:  # matrix
        n   = RNG.randint(50, 1000)
        reps = RNG.randint(10, 500)
        code = """
typedef vector<vector<long long>> Mat;
Mat matmul2(const Mat& A, const Mat& B) {
    Mat C(2,vector<long long>(2,0));
    for(int i=0;i<2;i++) for(int k=0;k<2;k++) for(int j=0;j<2;j++)
        C[i][j]=(C[i][j]+A[i][k]*B[k][j])%1000000007;
    return C;
}
Mat matpow(Mat M, int p) {
    Mat R={{{{1,0}},{{0,1}}}};
    while(p>0){if(p&1)R=matmul2(R,M);M=matmul2(M,M);p>>=1;}
    return R;
}
long long fibmat(int n){
    if(n<=1)return n;
    Mat M={{{{1,1}},{{1,0}}}};
    return matpow(M,n-1)[0][0];
}
"""
        call = f"    long long res=0; for(int r=0;r<{reps};r++) res=fibmat({n});"

    return HEADER + code + wrap_main("", call, "res")


# ─────────────────────────────────────────────────────────────────────────────
# 31. DOUBLY LINKED LIST
# ─────────────────────────────────────────────────────────────────────────────

def gen_doubly_linked_list(pid: int) -> str:
    n = rsize(1000, 30000)
    return HEADER + f"""
struct DNode {{ int val; DNode *prev, *next; DNode(int v):val(v),prev(nullptr),next(nullptr){{}} }};
struct DList {{
    DNode *head=nullptr, *tail=nullptr; int sz=0;
    void push_back(int v) {{
        DNode* nd=new DNode(v);
        if(!tail){{head=tail=nd;}} else {{tail->next=nd;nd->prev=tail;tail=nd;}} sz++;
    }}
    void remove(DNode* nd) {{
        if(nd->prev) nd->prev->next=nd->next; else head=nd->next;
        if(nd->next) nd->next->prev=nd->prev; else tail=nd->prev;
        delete nd; sz--;
    }}
    long long sum() {{ long long s=0; for(DNode*p=head;p;p=p->next) s+=p->val; return s; }}
}};
""" + wrap_main(
    f"    DList dl; for(int i=0;i<{n};i++) dl.push_back(rand()%{n});",
    "    long long res=dl.sum();",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 32. SEGMENT TREE (range sum)
# ─────────────────────────────────────────────────────────────────────────────

def gen_segment_tree(pid: int) -> str:
    n    = rsize(1000, 100000)
    qops = RNG.randint(n, n*5)
    return HEADER + f"""
const int SEGN={n};
long long seg[4*SEGN];
void build(int* a,int nd,int s,int e){{
    if(s==e){{seg[nd]=a[s];return;}}
    int m=s+(e-s)/2;
    build(a,2*nd,s,m); build(a,2*nd+1,m+1,e);
    seg[nd]=seg[2*nd]+seg[2*nd+1];
}}
void update(int nd,int s,int e,int i,int v){{
    if(s==e){{seg[nd]=v;return;}}
    int m=s+(e-s)/2;
    if(i<=m) update(2*nd,s,m,i,v); else update(2*nd+1,m+1,e,i,v);
    seg[nd]=seg[2*nd]+seg[2*nd+1];
}}
long long query(int nd,int s,int e,int l,int r){{
    if(r<s||e<l) return 0;
    if(l<=s&&e<=r) return seg[nd];
    int m=s+(e-s)/2;
    return query(2*nd,s,m,l,r)+query(2*nd+1,m+1,e,l,r);
}}
""" + wrap_main(
    f"    const int N={n}, Q={qops}; vector<int> a(N);\n    for(int i=0;i<N;i++) a[i]=rand()%1000;\n    build(a.data(),1,0,N-1); long long acc=0;",
    f"    for(int q=0;q<Q;q++){{\n        int l=rand()%N,r=rand()%N; if(l>r) swap(l,r);\n        if(q%5==0) update(1,0,N-1,l,rand()%1000);\n        else acc+=query(1,0,N-1,l,r);\n    }}",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 33. TRIE
# ─────────────────────────────────────────────────────────────────────────────

def gen_trie(pid: int) -> str:
    n     = RNG.randint(1000, 30000)
    wlen  = RNG.randint(3, 15)
    chars = RNG.randint(2, 26)
    return HEADER + f"""
struct TrieNode {{ TrieNode* ch[26]={{}}; bool end=false; }};
struct Trie {{
    TrieNode* root=new TrieNode();
    void insert(const string& s){{
        TrieNode* cur=root;
        for(char c:s){{int i=c-'a';if(!cur->ch[i])cur->ch[i]=new TrieNode();cur=cur->ch[i];}}
        cur->end=true;
    }}
    bool search(const string& s){{
        TrieNode* cur=root;
        for(char c:s){{int i=c-'a';if(!cur->ch[i])return false;cur=cur->ch[i];}}
        return cur->end;
    }}
}};
string randword(int len, int C) {{
    string s(len,'a'); for(auto& c:s) c='a'+rand()%C; return s;
}}
""" + wrap_main(
    f"    const int N={n}, WL={wlen}, C={chars}; Trie t; long long acc=0;",
    f"    for(int i=0;i<N;i++) t.insert(randword(WL,C));\n    for(int i=0;i<N;i++) acc+=t.search(randword(WL,C))?1:0;",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 34. UNION-FIND (DSU)
# ─────────────────────────────────────────────────────────────────────────────

def gen_union_find(pid: int) -> str:
    n   = rsize(1000, 100000)
    ops = n * RNG.randint(2, 10)
    return HEADER + f"""
struct DSU {{
    vector<int> p, rank_;
    DSU(int n):p(n),rank_(n,0){{iota(p.begin(),p.end(),0);}}
    int find(int x){{return p[x]==x?x:p[x]=find(p[x]);}}
    void unite(int a,int b){{a=find(a);b=find(b);if(a==b)return;if(rank_[a]<rank_[b])swap(a,b);p[b]=a;if(rank_[a]==rank_[b])rank_[a]++;}}
    bool same(int a,int b){{return find(a)==find(b);}}
}};
""" + wrap_main(
    f"    const int N={n}, OPS={ops}; DSU dsu(N); long long acc=0;",
    f"    for(int i=0;i<OPS;i++){{\n        int a=rand()%N, b=rand()%N;\n        if(i%3==0) dsu.unite(a,b);\n        else acc+=dsu.same(a,b)?1:0;\n    }}",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 35. CONVOLUTION / PREFIX SUM
# ─────────────────────────────────────────────────────────────────────────────

def gen_prefix_sum(pid: int) -> str:
    n    = rsize(10000, 100000)
    ops  = RNG.randint(n, n*10)
    dim  = RNG.choice([1, 2])
    if dim == 1:
        return HEADER + f"""
""" + wrap_main(
    f"    const int N={n}, Q={ops}; vector<long long> a(N), pre(N+1,0);\n    for(int i=0;i<N;i++) a[i]=rand()%1000;\n    for(int i=0;i<N;i++) pre[i+1]=pre[i]+a[i];\n    long long acc=0;",
    f"    for(int q=0;q<Q;q++){{int l=rand()%N,r=rand()%N;if(l>r)swap(l,r);acc+=pre[r+1]-pre[l];}}",
    "acc"
)
    else:
        sz = RNG.choice([100, 200, 300, 500])
        return HEADER + f"""
const int PS={sz};
long long grid2[PS][PS], pre2[PS+1][PS+1];
""" + wrap_main(
    f"    for(int i=0;i<PS;i++) for(int j=0;j<PS;j++) grid2[i][j]=rand()%100;\n    for(int i=1;i<=PS;i++) for(int j=1;j<=PS;j++) pre2[i][j]=grid2[i-1][j-1]+pre2[i-1][j]+pre2[i][j-1]-pre2[i-1][j-1];\n    long long acc=0;",
    f"    for(int q=0;q<{ops};q++){{int r1=rand()%PS,c1=rand()%PS,r2=rand()%PS,c2=rand()%PS;if(r1>r2)swap(r1,r2);if(c1>c2)swap(c1,c2);acc+=pre2[r2+1][c2+1]-pre2[r1][c2+1]-pre2[r2+1][c1]+pre2[r1][c1];}}",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 36. AVL TREE (simplified)
# ─────────────────────────────────────────────────────────────────────────────

def gen_avl_tree(pid: int) -> str:
    n = rsize(500, 10000)
    return HEADER + f"""
struct AVL {{
    int v, h; AVL *l, *r;
    AVL(int v):v(v),h(1),l(nullptr),r(nullptr){{}}
}};
int ht(AVL* n){{return n?n->h:0;}}
int bf(AVL* n){{return n?ht(n->l)-ht(n->r):0;}}
void upd(AVL* n){{if(n)n->h=1+max(ht(n->l),ht(n->r));}}
AVL* rr(AVL* y){{AVL*x=y->l,*T2=x->r;x->r=y;y->l=T2;upd(y);upd(x);return x;}}
AVL* lr(AVL* x){{AVL*y=x->r,*T2=y->l;y->l=x;x->r=T2;upd(x);upd(y);return y;}}
AVL* ins(AVL* n,int k){{
    if(!n)return new AVL(k);
    if(k<n->v)n->l=ins(n->l,k);else if(k>n->v)n->r=ins(n->r,k);else return n;
    upd(n);int b=bf(n);
    if(b>1&&k<n->l->v) return rr(n);
    if(b<-1&&k>n->r->v) return lr(n);
    if(b>1&&k>n->l->v){{n->l=lr(n->l);return rr(n);}}
    if(b<-1&&k<n->r->v){{n->r=rr(n->r);return lr(n);}}
    return n;
}}
long long insum(AVL* n){{return n?insum(n->l)+n->v+insum(n->r):0;}}
""" + wrap_main(
    f"    const int N={n}; AVL* root=nullptr;\n    for(int i=0;i<N;i++) root=ins(root,rand()%N);",
    "    long long res=insum(root);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 37. BELLMAN-FORD
# ─────────────────────────────────────────────────────────────────────────────

def gen_bellman_ford(pid: int) -> str:
    v = RNG.choice([50, 100, 200, 500])
    e = min(v * RNG.randint(2, 6), v*(v-1))
    return HEADER + f"""
const int BV={v}, BE={e};
struct Edge{{int u,v,w;}};
Edge edges[BE];
long long bellmanFord(int src){{
    vector<long long> dist(BV,LLONG_MAX/2);
    dist[src]=0;
    for(int i=0;i<BV-1;i++) for(int j=0;j<BE;j++)
        if(dist[edges[j].u]+edges[j].w<dist[edges[j].v])
            dist[edges[j].v]=dist[edges[j].u]+edges[j].w;
    long long s=0; for(auto x:dist) if(x<LLONG_MAX/2) s+=x; return s;
}}
""" + wrap_main(
    f"    for(int i=0;i<BE;i++){{edges[i].u=rand()%BV;edges[i].v=rand()%BV;edges[i].w=rand()%100+1;}}",
    "    long long res=bellmanFord(0);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 38. TOPOLOGICAL SORT (Kahn's)
# ─────────────────────────────────────────────────────────────────────────────

def gen_topo_sort(pid: int) -> str:
    v = RNG.choice([100, 500, 1000, 2000, 5000])
    e = min(v * RNG.randint(1, 4), v*(v-1)//2)
    return HEADER + f"""
const int TV={v};
vector<int> tadj[TV];
int indeg[TV];
long long topoSort(){{
    queue<int> q; long long order=0;
    for(int i=0;i<TV;i++) if(!indeg[i]) q.push(i);
    while(!q.empty()){{
        int u=q.front(); q.pop(); order+=u;
        for(int nb:tadj[u]) if(--indeg[nb]==0) q.push(nb);
    }}
    return order;
}}
""" + wrap_main(
    f"    for(int i=0;i<{e};i++){{\n        int u=rand()%TV, vv=rand()%TV;\n        if(u<vv){{tadj[u].push_back(vv);indeg[vv]++;}}\n    }}",
    "    long long res=topoSort();",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# 39. ROD CUTTING (DP)
# ─────────────────────────────────────────────────────────────────────────────

def gen_rod_cutting(pid: int) -> str:
    n    = RNG.randint(10, 100)
    reps = RNG.randint(1, 100)
    return HEADER + f"""
int rodCut(int* price, int n){{
    vector<int> dp(n+1,0);
    for(int i=1;i<=n;i++) for(int j=1;j<=i;j++)
        dp[i]=max(dp[i],price[j-1]+dp[i-j]);
    return dp[n];
}}
""" + wrap_main(
    f"    const int N={n}; int price[{n}]; for(int i=0;i<N;i++) price[i]=rand()%100+1; long long acc=0;",
    f"    for(int r=0;r<{reps};r++) acc+=rodCut(price,N);",
    "acc"
)


# ─────────────────────────────────────────────────────────────────────────────
# 40. CONVEX HULL (Graham Scan)
# ─────────────────────────────────────────────────────────────────────────────

def gen_convex_hull(pid: int) -> str:
    n = RNG.choice([100, 500, 1000, 5000, 10000, 50000])
    return HEADER + f"""
struct P {{ long long x,y; }};
long long cross(P O,P A,P B){{return (A.x-O.x)*(B.y-O.y)-(A.y-O.y)*(B.x-O.x);}}
int convexHull(vector<P>& pts){{
    int n=pts.size(); if(n<3)return n;
    sort(pts.begin(),pts.end(),[](P a,P b){{return a.x<b.x||(a.x==b.x&&a.y<b.y);}});
    vector<P> h; int k=0;
    for(int i=0;i<n;i++){{while(k>=2&&cross(h[k-2],h[k-1],pts[i])<=0){{h.pop_back();k--;}}h.push_back(pts[i]);k++;}}
    for(int i=n-2,t=k+1;i>=0;i--){{while(k>=t&&cross(h[k-2],h[k-1],pts[i])<=0){{h.pop_back();k--;}}h.push_back(pts[i]);k++;}}
    return k-1;
}}
""" + wrap_main(
    f"    const int N={n}; vector<P> pts(N);\n    for(int i=0;i<N;i++){{pts[i].x=rand()%10000;pts[i].y=rand()%10000;}}",
    "    long long res=convexHull(pts);",
    "res"
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry of all generators
# ─────────────────────────────────────────────────────────────────────────────

GENERATORS: list[tuple[str, Callable, float]] = [
    # (category_name, generator_fn, relative_weight)
    ("bubble_sort",       gen_bubble_sort,      1.5),
    ("insertion_sort",    gen_insertion_sort,    1.5),
    ("selection_sort",    gen_selection_sort,    1.0),
    ("merge_sort",        gen_merge_sort,        2.0),
    ("quick_sort",        gen_quick_sort,        2.5),
    ("heap_sort",         gen_heap_sort,         2.0),
    ("radix_sort",        gen_radix_sort,        1.0),
    ("counting_sort",     gen_counting_sort,     1.0),
    ("binary_search",     gen_binary_search,     1.5),
    ("linked_list",       gen_linked_list,       2.0),
    ("stack",             gen_stack,             1.5),
    ("queue",             gen_queue,             1.5),
    ("bst",               gen_bst,               2.0),
    ("hash_table",        gen_hash_table,        2.0),
    ("priority_queue",    gen_priority_queue,    1.5),
    ("bfs",               gen_bfs,               2.5),
    ("dfs",               gen_dfs,               2.5),
    ("dijkstra",          gen_dijkstra,          2.5),
    ("floyd_warshall",    gen_floyd_warshall,    1.5),
    ("knapsack",          gen_knapsack,          2.0),
    ("lcs",               gen_lcs,               2.0),
    ("lis",               gen_lis,               1.5),
    ("coin_change",       gen_coin_change,       1.5),
    ("edit_distance",     gen_edit_distance,     1.5),
    ("matrix_mult",       gen_matrix_mult,       2.5),
    ("sieve",             gen_sieve,             1.5),
    ("stencil",           gen_stencil,           2.0),
    ("kmp",               gen_kmp,               2.0),
    ("vector_ops",        gen_vector_ops,        2.5),
    ("fibonacci",         gen_fibonacci,         2.0),
    ("doubly_linked_list",gen_doubly_linked_list,1.5),
    ("segment_tree",      gen_segment_tree,      2.0),
    ("trie",              gen_trie,              1.5),
    ("union_find",        gen_union_find,        1.5),
    ("prefix_sum",        gen_prefix_sum,        2.0),
    ("avl_tree",          gen_avl_tree,          1.5),
    ("bellman_ford",      gen_bellman_ford,      1.5),
    ("topo_sort",         gen_topo_sort,         1.5),
    ("rod_cutting",       gen_rod_cutting,       1.0),
    ("convex_hull",       gen_convex_hull,       1.5),
]

NAMES    = [g[0] for g in GENERATORS]
GEN_FNS  = [g[1] for g in GENERATORS]
WEIGHTS  = [g[2] for g in GENERATORS]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(count: int, out_dir: Path, seed: int = 0) -> None:
    global RNG
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tally per-category
    cat_counts: dict[str, int] = {n: 0 for n in NAMES}
    files_written = 0
    errors = 0

    print(f"Generating {count} programs -> {out_dir} ...")
    for pid in range(1, count + 1):
        # Pick a generator weighted by relative frequency
        RNG = random.Random(pid * 31337 + seed)
        gen_fn = RNG.choices(GEN_FNS, weights=WEIGHTS, k=1)[0]
        cat    = NAMES[GEN_FNS.index(gen_fn)]

        fname  = out_dir / f"prog_{pid:06d}_{cat}.cpp"
        try:
            code = gen_fn(pid)
            fname.write_text(code, encoding="utf-8")
            cat_counts[cat] += 1
            files_written += 1
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  [ERROR] pid={pid} cat={cat}: {exc}")

        if pid % 500 == 0:
            print(f"  {pid}/{count} ({pid*100//count}%)  errors={errors}")

    print(f"\nDone: {files_written} programs written, {errors} errors")
    print(f"Category breakdown:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        bar = "#" * (cnt * 40 // max(cat_counts.values(), default=1))
        print(f"  {cat:<25} {cnt:>5}  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--out",   default="benchmarks/generated")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    generate_all(args.count, Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
