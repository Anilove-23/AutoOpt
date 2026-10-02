"""
autoopt.synthesizer.loop_shapes
===============================
Procedural generators for diverse loop geometries, trip counts, and induction variables.
Directly models LLVM analysis pass targets:
  - loops: Natural loop depth and nesting trees
  - scalar-evolution (SCEV): Induction variable arithmetic and trip counts
  - licm: Loop Invariant Code Motion opportunities
  - loop-unroll & loop-unswitch: Canonicalization and branch unswitching
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LoopKernel:
    name: str
    description: str
    setup_code: str
    kernel_code: str
    result_expr: str
    loop_type: str


class LoopShapeGenerator:
    """Generates parameterized loop structures with diverse trip counts and topologies."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def gen_canonical_unroll(self, n: int, unroll_hint: int = 4) -> LoopKernel:
        """Countable canonical loop with simple induction variable i++ (loop unrolling candidate)."""
        return LoopKernel(
            name="canonical_countable",
            description=f"Countable canonical loop (trip count {n}) for unrolling analysis",
            setup_code=f"""
    const int N = {n};
    vector<int> arr(N);
    for(int i=0; i<N; ++i) arr[i] = i ^ 0x3C;
    long long total = 0;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        total += arr[i] * 3 + (arr[i] >> 2);
    }}
""",
            result_expr="total",
            loop_type="canonical",
        )

    def gen_geometric_induction(self, max_val: int = 10000000, multiplier: int = 2) -> LoopKernel:
        """Geometric induction variable i *= 2 (logarithmic trip count, SCEV non-affine)."""
        return LoopKernel(
            name=f"geometric_step_m{multiplier}",
            description=f"Geometric induction step (i*={multiplier}) with logarithmic trip count",
            setup_code=f"""
    const long long LIMIT = {max_val}LL;
    long long acc = 0;
""",
            kernel_code=f"""
    for(int rep=0; rep<50000; ++rep) {{
        for(long long i=1; i<LIMIT; i*={multiplier}) {{
            acc += (i ^ rep);
        }}
    }}
""",
            result_expr="acc",
            loop_type="geometric",
        )

    def gen_nested_3d(self, dim: int) -> LoopKernel:
        """3-level nested loop (Depth 3, matrix multiplication or 3D tensor contraction)."""
        return LoopKernel(
            name=f"nested_3d_d{dim}",
            description=f"3-level nested loop ({dim}x{dim}x{dim}) with loop-nest interchanges",
            setup_code=f"""
    const int D = {dim};
    vector<int> A(D*D, 1), B(D*D, 2), C(D*D, 0);
""",
            kernel_code=f"""
    for(int i=0; i<D; ++i) {{
        for(int k=0; k<D; ++k) {{
            for(int j=0; j<D; ++j) {{
                C[i*D + j] += A[i*D + k] * B[k*D + j];
            }}
        }}
    }}
""",
            result_expr="C[0] + C[D*D-1]",
            loop_type="nested_3d",
        )

    def gen_licm_candidate(self, n: int) -> LoopKernel:
        """Loop containing heavy loop-invariant expressions (ideal target for LICM)."""
        return LoopKernel(
            name="licm_invariant_hoisting",
            description="Loop invariant arithmetic and load expressions eligible for hoisting",
            setup_code=f"""
    const int N = {n};
    vector<float> a(N);
    float invariant_factor = 3.14159f;
    float invariant_offset = 2.71828f;
    float acc = 0.0f;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        // Invariant expression that LICM should hoist into preheader
        float hoisted_term = (invariant_factor * invariant_factor) + (invariant_offset / 1.414f);
        a[i] = (float)i * hoisted_term;
        acc += a[i];
    }}
""",
            result_expr="acc",
            loop_type="licm_hoisting",
        )

    def gen_multi_exit_loop(self, n: int, search_target: int) -> LoopKernel:
        """Loop with early break condition (multi-exit CFG, complex exit-block insertion)."""
        return LoopKernel(
            name="multi_exit_search",
            description="Multi-exit loop with early break condition and non-deterministic trip count",
            setup_code=f"""
    const int N = {n};
    vector<int> data(N);
    for(int i=0; i<N; ++i) data[i] = i * 2;
    int target = {search_target};
    int found_idx = -1;
""",
            kernel_code=f"""
    for(int rep=0; rep<1000; ++rep) {{
        for(int i=0; i<N; ++i) {{
            if (data[i] == target) {{
                found_idx = i;
                break; // Multi-exit CFG edge
            }}
        }}
    }}
""",
            result_expr="found_idx",
            loop_type="multi_exit",
        )

    def gen_loop_with_branches(self, n: int) -> LoopKernel:
        """Loop body containing conditional branches (target for loop-unswitching)."""
        return LoopKernel(
            name="loop_unswitch_branches",
            description="Loop containing branch predicates for loop-unswitching analysis",
            setup_code=f"""
    const int N = {n};
    vector<int> a(N), b(N, 0);
    for(int i=0; i<N; ++i) a[i] = i;
    bool enable_mode = true;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        if (enable_mode) {{  // Loop-invariant predicate -> loop-unswitch target
            b[i] = a[i] * 2 + 1;
        }} else {{
            b[i] = a[i] * 3 - 1;
        }}
    }}
""",
            result_expr="b[N-1]",
            loop_type="loop_unswitch",
        )
