"""
autoopt.analyzer.inst_counter
=============================
Instruction Mix, Arithmetic Profiling, and Call Graph Density Analyzer.
Inspired by LLVM passes:
  - instcount: Counts of various instruction types (arithmetic, float, calls, jumps)
  - basiccg / print-callgraph: Call graph construction and call-site frequency
  - inline & always-inline: Inlining profitability heuristics
  - tailcallelim: Tail recursion elimination analysis
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class InstructionMetrics:
    total_instructions: int
    arithmetic_instructions: int
    floating_point_instructions: int
    bitwise_instructions: int
    call_instructions: int
    recursive_calls: int
    float_ratio: float           # float_ops / total
    arithmetic_mix_ratio: float  # arith_ops / total
    call_density: float          # call_ops / total
    is_pure_function_candidate: bool  # no global writes or external calls


class InstructionCounter:
    """Analyzes opcode distributions and call graph density from source and assembly."""

    # Assembly opcodes
    ARITH_OPS = re.compile(r'^\s*(add|sub|imul|mul|idiv|div|neg|inc|dec|lea)\b', re.IGNORECASE)
    FLOAT_OPS = re.compile(r'^\s*(fld|fst|fadd|fsub|fmul|fdiv|movss|movsd|addss|addsd|subss|subsd|mulss|mulsd|divss|divsd|vadd|vmul|vsub|vdiv|sqrtss|sqrtsd)\b', re.IGNORECASE)
    BITWISE_OPS = re.compile(r'^\s*(xor|and|or|not|shl|shr|sar|sal|rol|ror)\b', re.IGNORECASE)
    CALL_OPS = re.compile(r'^\s*call\b', re.IGNORECASE)

    def analyze(self, source_code: str, assembly_code: str = "") -> InstructionMetrics:
        src = re.sub(r'/\*.*?\*/', ' ', source_code, flags=re.DOTALL)
        src = re.sub(r'//[^\n]*', ' ', src)

        # Detect recursive calls in source
        func_names = re.findall(r'\b([A-Za-z0-9_]+)\s*\([^;{]*\)\s*\{', src)
        recursive_count = 0
        for fn in set(func_names):
            if fn in ("main", "if", "for", "while"):
                continue
            calls_to_self = len(re.findall(rf'\b{fn}\s*\(', src))
            if calls_to_self >= 2:  # Declaration + at least 1 recursive call
                recursive_count += (calls_to_self - 1)

        if assembly_code:
            total, arith, flt, bitwise, calls = self._parse_assembly_insts(assembly_code)
        else:
            loc = max(1, len([l for l in src.splitlines() if l.strip()]))
            flt_types = len(re.findall(r'\b(float|double)\b', src))
            arith = len(re.findall(r'[\+\-\*\/%]', src))
            bitwise = len(re.findall(r'[&\|\^~]|<<|>>', src))
            calls = len(re.findall(r'\b[A-Za-z0-9_]+\s*\(', src))
            flt = flt_types * 4
            total = max(1, arith + flt + bitwise + calls + loc * 2)

        t = max(total, 1)
        float_ratio = flt / t
        arith_mix = arith / t
        call_dens = calls / t

        # Pure function candidate: no global writes, no heap allocations
        is_pure = (calls <= 1) and ("new " not in src) and ("malloc" not in src)

        return InstructionMetrics(
            total_instructions=total,
            arithmetic_instructions=arith,
            floating_point_instructions=flt,
            bitwise_instructions=bitwise,
            call_instructions=calls,
            recursive_calls=recursive_count,
            float_ratio=float_ratio,
            arithmetic_mix_ratio=arith_mix,
            call_density=call_dens,
            is_pure_function_candidate=is_pure,
        )

    def _parse_assembly_insts(self, asm: str) -> tuple[int, int, int, int, int]:
        total = 0
        arith = 0
        flt = 0
        bitwise = 0
        calls = 0

        for line in asm.splitlines():
            st = line.strip()
            if not st or st.startswith(('#', '.', '@')) or st.endswith(':'):
                continue
            total += 1
            if self.ARITH_OPS.match(st):
                arith += 1
            elif self.FLOAT_OPS.match(st):
                flt += 1
            elif self.BITWISE_OPS.match(st):
                bitwise += 1
            elif self.CALL_OPS.match(st):
                calls += 1

        return max(1, total), arith, flt, bitwise, calls
