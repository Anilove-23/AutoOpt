"""
autoopt.engine.sequences
========================
Defines compiler optimization passes, presets, and transformations.
Maps target compiler passes to specific optimization recipes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass(frozen=True)
class OptimizationSequence:
    id: int
    name: str
    description: str
    gcc_flags: str
    clang_flags: str
    llvm_passes: str
    primary_benefit: str


OPTIMIZATION_SEQUENCES: List[OptimizationSequence] = [
    OptimizationSequence(
        id=0,
        name="O0_baseline",
        description="Unoptimized baseline. Fast compile time, zero dead-code elimination.",
        gcc_flags="-O0",
        clang_flags="-O0",
        llvm_passes="",
        primary_benefit="Debugging and compilation speed",
    ),
    OptimizationSequence(
        id=2,
        name="O2_standard",
        description="Standard production optimization. Balances speed and code size without aggressive unrolling.",
        gcc_flags="-O2",
        clang_flags="-O2",
        llvm_passes="default<O2>",
        primary_benefit="General-purpose production default",
    ),
    OptimizationSequence(
        id=3,
        name="O3_aggressive",
        description="Aggressive optimization with vectorization and loop transformations.",
        gcc_flags="-O3",
        clang_flags="-O3",
        llvm_passes="default<O3>",
        primary_benefit="High-throughput compute kernels",
    ),
    OptimizationSequence(
        id=4,
        name="Os_size",
        description="Optimize for binary size. Minimizes instruction cache pressure and branch target buffer bloat.",
        gcc_flags="-Os",
        clang_flags="-Os",
        llvm_passes="default<Os>",
        primary_benefit="Instruction cache locality in pointer-chasing and branch-heavy code",
    ),
    OptimizationSequence(
        id=5,
        name="O2_unroll_vec",
        description="Level 2 optimization with aggressive loop unrolling and tree vectorization.",
        gcc_flags="-O2 -funroll-loops -ftree-vectorize",
        clang_flags="-O2 -funroll-loops -fvectorize",
        llvm_passes="default<O2>,loop-unroll,loop-vectorize",
        primary_benefit="Countable loops with high arithmetic-to-control ratio",
    ),
    OptimizationSequence(
        id=8,
        name="O2_no_inline",
        description="Level 2 optimization with function inlining disabled.",
        gcc_flags="-O2 -fno-inline",
        clang_flags="-O2 -fno-inline",
        llvm_passes="default<O2>",
        primary_benefit="Deep recursion avoiding stack frame bloat and cache eviction",
    ),
    OptimizationSequence(
        id=10,
        name="O3_combine",
        description="Aggressive O3 coupled with explicit loop unrolling, vectorization, and inlining.",
        gcc_flags="-O3 -funroll-loops -finline-functions -ftree-vectorize",
        clang_flags="-O3 -funroll-loops -finline-functions -fvectorize",
        llvm_passes="default<O3>,inline,loop-unroll,loop-vectorize",
        primary_benefit="Compute kernels with high call density and nested loops",
    ),
    OptimizationSequence(
        id=12,
        name="O3_no_vec",
        description="O3 with vectorization disabled to avoid scalar peel/remainder loops.",
        gcc_flags="-O3 -fno-tree-vectorize",
        clang_flags="-O3 -fno-vectorize",
        llvm_passes="default<O3>",
        primary_benefit="Control-heavy loops with irregular strides where vectorization overhead hurts",
    ),
    OptimizationSequence(
        id=13,
        name="O2_fast_math",
        description="Level 2 optimization enabling aggressive IEEE-754 relaxations and algebraic simplifications.",
        gcc_flags="-O2 -ffast-math",
        clang_flags="-O2 -ffast-math",
        llvm_passes="default<O2>,reassociate,instcombine",
        primary_benefit="Floating-point matrix math, stencils, and DSP filters",
    ),
    OptimizationSequence(
        id=17,
        name="O1_unroll",
        description="Lightweight O1 pipeline with loop unrolling, preserving tight instruction scheduling.",
        gcc_flags="-O1 -funroll-loops",
        clang_flags="-O1 -funroll-loops",
        llvm_passes="default<O1>,loop-unroll",
        primary_benefit="Small tight loops where register allocator pressure in O2/O3 causes spills",
    ),
    OptimizationSequence(
        id=18,
        name="O2_gcse_pre",
        description="O2 with Global Common Subexpression Elimination and Partial Redundancy Elimination.",
        gcc_flags="-O2 -fgcse",
        clang_flags="-O2",
        llvm_passes="default<O2>,gvn,licm",
        primary_benefit="Redundant memory loads and invariant expression elimination",
    ),
    OptimizationSequence(
        id=19,
        name="full_aggressive",
        description="Maximum optimization: O3 + fast-math + loop unrolling + inlining + vectorization.",
        gcc_flags="-O3 -ffast-math -funroll-loops -finline-functions -ftree-vectorize",
        clang_flags="-O3 -ffast-math -funroll-loops -finline-functions -fvectorize",
        llvm_passes="default<O3>,reassociate,inline,loop-unroll,loop-vectorize",
        primary_benefit="Pure numerical throughput in compute-bound matrix & vector kernels",
    ),
]

SEQUENCE_MAP: Dict[int, OptimizationSequence] = {s.id: s for s in OPTIMIZATION_SEQUENCES}


def get_sequence_by_id(seq_id: int) -> OptimizationSequence:
    return SEQUENCE_MAP.get(seq_id, SEQUENCE_MAP[3])
