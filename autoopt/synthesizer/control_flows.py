"""
autoopt.synthesizer.control_flows
=================================
Procedural generators for control-flow topologies and branch characteristics.
Directly models LLVM analysis pass targets:
  - domtree & domfrontier: Forward and post-dominator frontiers
  - jump-threading: Correlated branch conditions across basic blocks
  - simplifycfg: Branch merging and dead block elimination
  - lower-switch: Jump table vs binary branch search trees
  - tailcallelim: Tail recursion transformation opportunities
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class CFGKernel:
    name: str
    description: str
    setup_code: str
    kernel_code: str
    result_expr: str
    cfg_type: str


class ControlFlowGenerator:
    """Generates parameterized control-flow graph topologies and branch structures."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def gen_diamond_branches(self, n: int) -> CFGKernel:
        """Symmetric if-else diamond branches (balanced dominance frontiers)."""
        return CFGKernel(
            name="cfg_diamond_branch",
            description="Symmetric if-else diamond control flow structure",
            setup_code=f"""
    const int N = {n};
    vector<int> a(N);
    for(int i=0; i<N; ++i) a[i] = rand() % 100;
    long long total = 0;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        int val = a[i];
        if ((val & 1) == 0) {{
            total += val * 3 + 7;
        }} else {{
            total -= val * 2 - 5;
        }}
    }}
""",
            result_expr="total",
            cfg_type="diamond",
        )

    def gen_cascade_ladder(self, n: int, stages: int = 5) -> CFGKernel:
        """Deep if-else-if cascade ladder (tests block placement and branch probability)."""
        ladder_code = ""
        for s in range(stages):
            ladder_code += f"""
        else if (val % {stages+1} == {s}) {{
            res = (val * {s+2}) ^ 0x{s:02X};
        }}"""
        return CFGKernel(
            name=f"cfg_ladder_{stages}way",
            description=f"Multi-way ({stages} branches) cascade ladder",
            setup_code=f"""
    const int N = {n};
    vector<int> a(N);
    for(int i=0; i<N; ++i) a[i] = rand() % 1000;
    long long sum = 0;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        int val = a[i];
        int res = 0;
        if (val % {stages+1} == 0) {{
            res = val + 1;
        }}{ladder_code}
        else {{
            res = val ^ 0xDEAD;
        }}
        sum += res;
    }}
""",
            result_expr="sum",
            cfg_type="ladder",
        )

    def gen_jump_table_switch(self, n: int, cases: int = 8) -> CFGKernel:
        """Multi-case switch statement (tests lower-switch and jump-table creation)."""
        case_clauses = ""
        for c in range(cases):
            case_clauses += f"""
            case {c}: acc += (val * {c+3}) ^ 0x{c+1:02X}; break;"""

        return CFGKernel(
            name=f"cfg_switch_{cases}cases",
            description=f"Switch statement with {cases} cases for jump-table analysis",
            setup_code=f"""
    const int N = {n};
    vector<int> a(N);
    for(int i=0; i<N; ++i) a[i] = rand() % {cases};
    long long acc = 0;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        int val = a[i];
        switch(val) {{{case_clauses}
            default: acc += val; break;
        }}
    }}
""",
            result_expr="acc",
            cfg_type="switch_jump_table",
        )

    def gen_correlated_jump_threading(self, n: int) -> CFGKernel:
        """Correlated conditional branches (ideal target for LLVM jump-threading pass)."""
        return CFGKernel(
            name="cfg_correlated_jump_threading",
            description="Correlated consecutive branches (jump-threading opportunity)",
            setup_code=f"""
    const int N = {n};
    vector<int> a(N);
    for(int i=0; i<N; ++i) a[i] = rand() % 100;
    long long score = 0;
""",
            kernel_code=f"""
    for(int i=0; i<N; ++i) {{
        int x = a[i];
        int y = 0;
        if (x > 50) {{
            y = 10;
        }} else {{
            y = 20;
        }}
        // Correlated second branch: jump-threading can bypass testing y
        if (y == 10) {{
            score += x * 2;
        }} else {{
            score += x / 2;
        }}
    }}
""",
            result_expr="score",
            cfg_type="jump_threading",
        )

    def gen_recursive_callgraph(self, depth: int) -> CFGKernel:
        """Recursive call graph with function prologue/epilogue overhead (inlining / tail call target)."""
        return CFGKernel(
            name=f"cfg_recursive_d{depth}",
            description=f"Recursive function call graph (depth {depth}) for inlining analysis",
            setup_code=f"""
    auto rec_fn = [](auto self, int d, int val) -> long long {{
        if (d <= 0) return val;
        return self(self, d - 1, val * 2 + 1) + self(self, d - 1, val - 1);
    }};
    long long total = 0;
""",
            kernel_code=f"""
    for(int i=0; i<5000; ++i) {{
        total += rec_fn(rec_fn, {depth}, i % 10);
    }}
""",
            result_expr="total",
            cfg_type="recursive_callgraph",
        )
