"""
autoopt.analyzer.loop_analyzer
==============================
Loop Structure and Scalar Evolution (SCEV) Analyzer.
Inspired by LLVM passes:
  - loops: Natural Loop Information
  - scalar-evolution: Scalar expressions, induction variable detection
  - indvars: Canonicalize induction variables
  - licm: Loop Invariant Code Motion candidates
  - loop-unroll: Loop trip-count determinism
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class LoopMetrics:
    total_loops: int
    max_nesting_depth: int
    canonical_loops: int      # i++ / i+=1 (prime loop unroll target)
    strided_loops: int        # i+=2, i+=stride (stride vectorization target)
    geometric_loops: int      # i*=2, i/=2 (logarithmic trip count)
    nested_loop_ratio: float  # loops with depth >= 2 / total
    multi_exit_loops: int     # loops containing 'break' or early return
    loop_invariant_ratio: float  # estimate of loop invariant calculations


class LoopAnalyzer:
    """Analyzes natural loop structure, nesting geometry, and induction variables."""

    LOOP_KEYWORD = re.compile(r'\b(for|while|do)\b')
    CANONICAL_INC = re.compile(r'\+\+|\+=\s*1\b')
    STRIDED_INC   = re.compile(r'\+=\s*[2-9]\b|\+=\s*\w+')
    GEOMETRIC_INC = re.compile(r'\*=\s*\d+|/=\s*\d+|<<=\s*\d+|>>=\s*\d+')
    BREAK_RETURN  = re.compile(r'\b(break|return)\b')

    def analyze(self, source_code: str) -> LoopMetrics:
        src = re.sub(r'/\*.*?\*/', ' ', source_code, flags=re.DOTALL)
        src = re.sub(r'//[^\n]*', ' ', src)

        loop_matches = self.LOOP_KEYWORD.findall(src)
        total_loops = len(loop_matches)

        if total_loops == 0:
            return LoopMetrics(
                total_loops=0,
                max_nesting_depth=0,
                canonical_loops=0,
                strided_loops=0,
                geometric_loops=0,
                nested_loop_ratio=0.0,
                multi_exit_loops=0,
                loop_invariant_ratio=0.0,
            )

        # Detect maximum nesting depth
        max_depth = 0
        depth_ge_2 = 0
        current_depth = 0

        # Scan curly braces and loop keywords
        tokens = re.findall(r'(\bfor\b|\bwhile\b|\bdo\b|\{|\})', src)
        loop_context_stack = []

        for tok in tokens:
            if tok in ("for", "while", "do"):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                if current_depth >= 2:
                    depth_ge_2 += 1
                loop_context_stack.append(current_depth)
            elif tok == "{":
                pass
            elif tok == "}":
                if loop_context_stack:
                    current_depth = max(0, current_depth - 1)
                    loop_context_stack.pop()

        max_depth = max(1, max_depth) if total_loops > 0 else 0

        # Induction variable step types
        for_loops = re.findall(r'for\s*\([^;]*;[^;]*;([^)]*)\)', src)
        canonical_cnt = 0
        strided_cnt = 0
        geometric_cnt = 0

        for inc_expr in for_loops:
            if self.CANONICAL_INC.search(inc_expr):
                canonical_cnt += 1
            elif self.STRIDED_INC.search(inc_expr):
                strided_cnt += 1
            elif self.GEOMETRIC_INC.search(inc_expr):
                geometric_cnt += 1
            else:
                canonical_cnt += 1

        # Check for multi-exit loops (break statements)
        breaks = len(self.BREAK_RETURN.findall(src))
        multi_exit = min(total_loops, breaks)

        # Estimate invariant calculations (proxy: expressions before loop or const variables)
        const_vars = len(re.findall(r'\bconst\b', src))
        inv_ratio = min(1.0, const_vars / max(1, total_loops * 3))

        return LoopMetrics(
            total_loops=total_loops,
            max_nesting_depth=max_depth,
            canonical_loops=canonical_cnt,
            strided_loops=strided_cnt,
            geometric_loops=geometric_cnt,
            nested_loop_ratio=depth_ge_2 / max(1, total_loops),
            multi_exit_loops=multi_exit,
            loop_invariant_ratio=inv_ratio,
        )
