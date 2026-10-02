"""
autoopt.synthesizer.generator
=============================
Combinatorial Procedural Kernel Synthesizer.
Composes Memory Patterns x Loop Shapes x Control Flow Topologies x Data Types
into up to 1,000,000+ unique, compilable, self-contained C++ benchmark programs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from autoopt.synthesizer.memory_patterns import MemoryPatternGenerator, MemoryKernel
from autoopt.synthesizer.loop_shapes import LoopShapeGenerator, LoopKernel
from autoopt.synthesizer.control_flows import ControlFlowGenerator, CFGKernel

CPP_HEADER = """\
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <climits>
#include <numeric>
using namespace std;
"""


@dataclass
class SynthesizedProgram:
    id: int
    name: str
    category: str
    dimension: str
    source_code: str


class KernelSynthesizer:
    """Procedurally synthesizes millions of unique C++ benchmark programs."""

    DATA_TYPES = ["int", "long long", "float", "double", "unsigned int"]
    SIZES_SMALL  = [100, 256, 500, 1000]
    SIZES_MEDIUM = [2000, 5000, 10000, 25000]
    SIZES_LARGE  = [50000, 100000, 200000]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.mem_gen = MemoryPatternGenerator(self.rng)
        self.loop_gen = LoopShapeGenerator(self.rng)
        self.cfg_gen = ControlFlowGenerator(self.rng)

    def synthesize_one(self, program_id: int) -> SynthesizedProgram:
        """
        Synthesizes a unique program with deterministically randomized
        memory patterns, loop shapes, or control-flow topologies.
        """
        rng = random.Random(program_id * 1000003 + 17)
        dimension = rng.choice(["memory", "loop", "control_flow"])
        dtype = rng.choice(self.DATA_TYPES)

        if dimension == "memory":
            pattern = rng.choice([
                "sequential", "strided", "indirect", "pointer_chasing",
                "dependence_raw", "redundant_gvn", "stencil", "triangular"
            ])
            n = rng.choice(self.SIZES_MEDIUM if dtype != "double" else self.SIZES_SMALL)

            if pattern == "sequential":
                k = self.mem_gen.gen_sequential(n, dtype=dtype)
            elif pattern == "strided":
                s = rng.choice([2, 3, 4, 8, 16])
                k = self.mem_gen.gen_strided(max(100, n // s), s, dtype=dtype)
            elif pattern == "indirect":
                k = self.mem_gen.gen_indirect_gather_scatter(n, dtype="int")
            elif pattern == "pointer_chasing":
                k = self.mem_gen.gen_pointer_chasing(max(500, n // 2))
            elif pattern == "dependence_raw":
                k = self.mem_gen.gen_loop_carried_dependence(n, dtype=dtype)
            elif pattern == "redundant_gvn":
                k = self.mem_gen.gen_redundant_loads_stores(n, dtype="int")
            elif pattern == "stencil":
                dim = rng.choice([64, 128, 256])
                steps = rng.randint(5, 25)
                k = self.mem_gen.gen_2d_stencil(dim, steps, dtype=dtype)
            else:
                dim = rng.choice([100, 256, 500])
                k = self.mem_gen.gen_triangular_access(dim, dtype="int")

            name = f"prog_{program_id:07d}_{k.name}"
            category = f"mem_{pattern}"
            body_setup = k.setup_code
            body_kernel = k.kernel_code
            res_expr = k.result_expr

        elif dimension == "loop":
            loop_kind = rng.choice([
                "canonical", "geometric", "nested_3d", "licm", "multi_exit", "loop_unswitch"
            ])
            n = rng.choice(self.SIZES_MEDIUM)

            if loop_kind == "canonical":
                k = self.loop_gen.gen_canonical_unroll(n)
            elif loop_kind == "geometric":
                mult = rng.choice([2, 3, 4])
                k = self.loop_gen.gen_geometric_induction(10000000, mult)
            elif loop_kind == "nested_3d":
                dim = rng.choice([32, 64, 96])
                k = self.loop_gen.gen_nested_3d(dim)
            elif loop_kind == "licm":
                k = self.loop_gen.gen_licm_candidate(n)
            elif loop_kind == "multi_exit":
                target = rng.randint(0, n * 2)
                k = self.loop_gen.gen_multi_exit_loop(n, target)
            else:
                k = self.loop_gen.gen_loop_with_branches(n)

            name = f"prog_{program_id:07d}_{k.name}"
            category = f"loop_{loop_kind}"
            body_setup = k.setup_code
            body_kernel = k.kernel_code
            res_expr = k.result_expr

        else:  # control_flow
            cfg_kind = rng.choice([
                "diamond", "ladder", "switch", "jump_threading", "recursion"
            ])
            n = rng.choice(self.SIZES_MEDIUM)

            if cfg_kind == "diamond":
                k = self.cfg_gen.gen_diamond_branches(n)
            elif cfg_kind == "ladder":
                stages = rng.choice([3, 5, 7, 10])
                k = self.cfg_gen.gen_cascade_ladder(n, stages=stages)
            elif cfg_kind == "switch":
                cases = rng.choice([4, 8, 16])
                k = self.cfg_gen.gen_jump_table_switch(n, cases=cases)
            elif cfg_kind == "jump_threading":
                k = self.cfg_gen.gen_correlated_jump_threading(n)
            else:
                depth = rng.choice([10, 12, 14, 16])
                k = self.cfg_gen.gen_recursive_callgraph(depth)

            name = f"prog_{program_id:07d}_{k.name}"
            category = f"cfg_{cfg_kind}"
            body_setup = k.setup_code
            body_kernel = k.kernel_code
            res_expr = k.result_expr

        # Wrap into full compilable program
        source = f"""{CPP_HEADER}
int main() {{
    srand(42);
{body_setup}
    clock_t _t0 = clock();
{body_kernel}
    long long _result = (long long)({res_expr});
    clock_t _t1 = clock();
    cout << "time=" << (double)(_t1-_t0)/CLOCKS_PER_SEC << " result=" << _result << endl;
    return 0;
}}
"""
        return SynthesizedProgram(
            id=program_id,
            name=name,
            category=category,
            dimension=dimension,
            source_code=source,
        )

    def synthesize_batch(
        self,
        count: int,
        start_id: int = 1,
        output_dir: Optional[Path] = None,
    ) -> List[SynthesizedProgram]:
        """Synthesizes a batch of programs, optionally persisting to disk."""
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        programs = []
        for i in range(count):
            pid = start_id + i
            prog = self.synthesize_one(pid)
            if output_dir:
                file_path = output_dir / f"{prog.name}.cpp"
                file_path.write_text(prog.source_code, encoding="utf-8")
            programs.append(prog)

        return programs
