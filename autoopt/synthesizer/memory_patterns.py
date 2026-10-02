"""
autoopt.synthesizer.memory_patterns
===================================
Procedural generators for diverse memory access patterns and dependence structures.
Directly models LLVM analysis pass targets:
  - aa-eval & basic-aa: Aliasing vs non-aliasing pointer relations
  - da (dependence analysis): Loop-carried vs independent accesses
  - memdep: Redundant load/store dependencies (GVN / DSE targets)
  - scev-aa: Strided and affine memory recurrences
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class MemoryKernel:
    name: str
    description: str
    setup_code: str
    kernel_code: str
    result_expr: str
    memory_type: str


class MemoryPatternGenerator:
    """Generates parameterized C++ kernels with diverse memory access characteristics."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def gen_sequential(self, n: int, dtype: str = "float") -> MemoryKernel:
        """Contiguous sequential memory access A[i] (unit stride, vectorization candidate)."""
        return MemoryKernel(
            name="seq_unit_stride",
            description="Contiguous array traversal with unit stride",
            setup_code=f"""
    const int N = {n};
    vector<{dtype}> a(N), b(N), c(N);
    for(int i=0; i<N; ++i) {{ a[i]=({dtype})(i*0.5f); b[i]=({dtype})((N-i)*0.25f); }}
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        c[i] = a[i] * 1.5f + b[i] * 0.75f;
    }}
""",
            result_expr="c[0] + c[N-1]",
            memory_type="sequential",
        )

    def gen_strided(self, n: int, stride: int, dtype: str = "float") -> MemoryKernel:
        """Non-unit strided memory access A[i * stride] (cache line hopping, SCEV target)."""
        total_sz = n * stride
        return MemoryKernel(
            name=f"strided_s{stride}",
            description=f"Non-unit strided memory access with stride {stride}",
            setup_code=f"""
    const int N = {n}, STRIDE = {stride};
    const int SZ = {total_sz};
    vector<{dtype}> data(SZ);
    for(int i=0; i<SZ; ++i) data[i]=({dtype})(i % 100);
    {dtype} acc = 0;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        acc += data[i * STRIDE] * 1.25f;
    }}
""",
            result_expr="acc",
            memory_type="strided",
        )

    def gen_indirect_gather_scatter(self, n: int, dtype: str = "int") -> MemoryKernel:
        """Indirect memory access A[idx[i]] (scatter/gather, memory latency bound)."""
        return MemoryKernel(
            name="indirect_gather_scatter",
            description="Indirect random memory access via index table",
            setup_code=f"""
    const int N = {n};
    vector<{dtype}> data(N), idx(N);
    for(int i=0; i<N; ++i) {{ data[i]=({dtype})(i*3); idx[i]=(rand() % N); }}
    long long acc = 0;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        int target = idx[i];
        acc += data[target];
        data[target] = (data[target] ^ 0x5A5A) + 1;
    }}
""",
            result_expr="acc + data[0]",
            memory_type="indirect",
        )

    def gen_pointer_chasing(self, nodes: int) -> MemoryKernel:
        """Pointer-chasing traversal across heap-allocated linked list (serial latency)."""
        return MemoryKernel(
            name="pointer_chasing_list",
            description="Linked list node traversal with pointer-induced dependencies",
            setup_code=f"""
    struct PNode {{ int val; PNode* next; }};
    const int NODES = {nodes};
    vector<PNode> pool(NODES);
    for(int i=0; i<NODES-1; ++i) {{ pool[i].val = i; pool[i].next = &pool[i+1]; }}
    pool[NODES-1].val = NODES-1; pool[NODES-1].next = nullptr;
    PNode* head = &pool[0];
    long long acc = 0;
""",
            kernel_code=f"""
    PNode* curr = head;
    while(curr != nullptr) {{
        acc += curr->val;
        curr = curr->next;
    }}
""",
            result_expr="acc",
            memory_type="pointer_chasing",
        )

    def gen_loop_carried_dependence(self, n: int, dtype: str = "float") -> MemoryKernel:
        """True loop-carried dependency A[i] = A[i-1] + B[i] (RAW dependence, tests DA)."""
        return MemoryKernel(
            name="loop_carried_raw",
            description="True loop-carried dependency (RAW hazard prevents naive vectorization)",
            setup_code=f"""
    const int N = {n};
    vector<{dtype}> a(N, 1.0f), b(N, 0.5f);
""",
            kernel_code=f"""
    for(int i=1; i<N; ++i) {{
        a[i] = a[i-1] * 0.999f + b[i];
    }}
""",
            result_expr="a[N-1]",
            memory_type="dependence_raw",
        )

    def gen_redundant_loads_stores(self, n: int, dtype: str = "int") -> MemoryKernel:
        """Redundant loads and dead stores (target for GVN, DSE, and memdep passes)."""
        return MemoryKernel(
            name="redundant_memdep",
            description="Redundant memory access pattern with redundant loads and stores",
            setup_code=f"""
    const int N = {n};
    vector<{dtype}> mem(N);
    {dtype} shared_slot = 42;
    long long total = 0;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        // Redundant load of shared_slot (GVN candidate)
        {dtype} val1 = shared_slot;
        {dtype} val2 = shared_slot;
        mem[i] = val1 + val2 + (i % 7);
        // Overwritten store (DSE candidate)
        shared_slot = i;
        shared_slot = i + 1;
        total += mem[i];
    }}
""",
            result_expr="total + shared_slot",
            memory_type="redundant_gvn",
        )

    def gen_2d_stencil(self, dim: int, timesteps: int, dtype: str = "float") -> MemoryKernel:
        """2D 5-point spatial stencil with row-major and column-major neighborhood access."""
        return MemoryKernel(
            name=f"stencil_{dim}x{dim}",
            description=f"2D 5-point stencil across {dim}x{dim} grid with {timesteps} steps",
            setup_code=f"""
    const int D = {dim};
    vector<vector<{dtype}>> g(D, vector<{dtype}>(D, 1.0f));
    vector<vector<{dtype}>> next_g(D, vector<{dtype}>(D, 0.0f));
""",
            kernel_code=f"""
    for(int t=0; t<{timesteps}; ++t) {{
        for(int i=1; i<D-1; ++i) {{
            for(int j=1; j<D-1; ++j) {{
                next_g[i][j] = 0.25f * (g[i-1][j] + g[i+1][j] + g[i][j-1] + g[i][j+1]);
            }}
        }}
        for(int i=1; i<D-1; ++i)
            for(int j=1; j<D-1; ++j)
                g[i][j] = next_g[i][j];
    }}
""",
            result_expr="g[D/2][D/2]",
            memory_type="spatial_stencil",
        )

    def gen_triangular_access(self, n: int, dtype: str = "int") -> MemoryKernel:
        """Triangular array access (inner loop bound j < i, non-rectangular iteration space)."""
        return MemoryKernel(
            name="triangular_matrix",
            description="Lower triangular matrix traversal with dynamic inner loop trip count",
            setup_code=f"""
    const int N = {n};
    vector<{dtype}> mat(N * N, 1);
    long long sum = 0;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        for(int j=0; j<=i; ++j) {{
            sum += mat[i * N + j];
        }}
    }}
""",
            result_expr="sum",
            memory_type="triangular",
        )
