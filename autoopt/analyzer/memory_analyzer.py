"""
autoopt.analyzer.memory_analyzer
================================
Memory Access Pattern, Memory Dependence, and Alias Analysis Proxy.
Inspired by LLVM passes:
  - aa-eval: Alias Analysis Precision Evaluator
  - basic-aa, scev-aa: Pointer aliasing and induction pointer arithmetic
  - memdep: Memory Dependence Analysis (redundant loads, dead stores)
  - da: Dependence Analysis (loop-carried dependencies)
  - sroa / mem2reg: Aggregate scalarization and register promotion
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MemoryMetrics:
    total_memory_ops: int
    load_operations: int
    store_operations: int
    load_store_ratio: float
    memory_intensity: float       # Memory operations / total instructions
    pointer_dereferences: int
    pointer_arguments: int        # Target for argpromotion
    aliasing_risk_score: float    # 0.0 (strictly no-alias) to 1.0 (high alias likelihood)
    contiguous_access_ratio: float # A[i] vs indirect A[idx[i]]
    indirect_access_count: int    # Scatter/gather operations
    dynamic_allocations: int      # new / malloc calls


class MemoryAnalyzer:
    """Analyzes memory access locality, alias likelihood, and memory intensity."""

    POINTER_DECL = re.compile(r'\b[A-Za-z0-9_:]+\s*\*\s*([A-Za-z0-9_]+)')
    POINTER_ARROW = re.compile(r'->|\*\([A-Za-z0-9_]+\)')
    ARRAY_SUBSCRIPT = re.compile(r'\[([^\]]+)\]')
    INDIRECT_INDEX = re.compile(r'\[[A-Za-z0-9_]+\[[^\]]+\]\]')  # e.g. A[idx[i]]
    ALLOCATION = re.compile(r'\b(new\s+|malloc\s*\(|calloc\s*\(|vector<|string\b)')

    # Assembly load/store patterns
    MEM_LOAD = re.compile(r'^\s*(mov|ldr|ld|movss|movsd|vmovss|vmovsd)\s+[^,]+,\s*\[?[A-Za-z0-9_$.+-]+\]?', re.IGNORECASE)
    MEM_STORE = re.compile(r'^\s*(mov|str|st|movss|movsd|vmovss|vmovsd)\s+\[?[A-Za-z0-9_$.+-]+\]?,\s*[^,]+', re.IGNORECASE)

    def analyze(self, source_code: str, assembly_code: str = "") -> MemoryMetrics:
        src = re.sub(r'/\*.*?\*/', ' ', source_code, flags=re.DOTALL)
        src = re.sub(r'//[^\n]*', ' ', src)

        # Pointer analysis
        ptr_matches = self.POINTER_DECL.findall(src)
        ptr_count = len(ptr_matches)
        derefs = len(self.POINTER_ARROW.findall(src)) + len(re.findall(r'\*\s*[A-Za-z0-9_]+', src))

        # Pointer arguments in function signatures
        ptr_args = len(re.findall(r'\b[A-Za-z0-9_:]+\s*\*\s*[A-Za-z0-9_]+\s*[,)]', src))

        # Array indexing patterns
        all_subscripts = self.ARRAY_SUBSCRIPT.findall(src)
        indirect_subscripts = self.INDIRECT_INDEX.findall(src)
        indirect_count = len(indirect_subscripts)
        total_subscripts = max(1, len(all_subscripts))
        contiguous_ratio = max(0.0, 1.0 - (indirect_count / total_subscripts))

        # Dynamic allocations
        alloc_count = len(self.ALLOCATION.findall(src))

        # Alias Risk Score (inspired by aa-eval):
        # Multiple pointers of the same type in the same scope increase alias ambiguity
        if ptr_count >= 2 and ptr_args >= 1:
            alias_risk = min(1.0, 0.3 + 0.15 * ptr_count)
        elif ptr_count >= 1:
            alias_risk = 0.2
        else:
            alias_risk = 0.05

        # Memory instruction counts from assembly if available
        if assembly_code:
            loads, stores, total_insts = self._parse_assembly_memory(assembly_code)
        else:
            # Estimate from source syntax
            loads = max(1, total_subscripts + derefs)
            stores = max(1, len(re.findall(r'=[^=;]+;', src)))
            total_insts = loads + stores + ptr_count * 2

        mem_ops = loads + stores
        mem_intensity = mem_ops / max(1, total_insts)
        load_store_ratio = loads / max(1, stores)

        return MemoryMetrics(
            total_memory_ops=mem_ops,
            load_operations=loads,
            store_operations=stores,
            load_store_ratio=load_store_ratio,
            memory_intensity=mem_intensity,
            pointer_dereferences=derefs,
            pointer_arguments=ptr_args,
            aliasing_risk_score=alias_risk,
            contiguous_access_ratio=contiguous_ratio,
            indirect_access_count=indirect_count,
            dynamic_allocations=alloc_count,
        )

    def _parse_assembly_memory(self, asm: str) -> tuple[int, int, int]:
        loads = 0
        stores = 0
        total = 0

        for line in asm.splitlines():
            st = line.strip()
            if not st or st.startswith(('#', '.', '@')) or st.endswith(':'):
                continue
            total += 1
            # Check for memory operand indicators in x86 AT&T or Intel syntax
            has_mem = '(' in st or '[' in st or 'PTR' in st
            if has_mem:
                if self.MEM_STORE.match(st):
                    stores += 1
                else:
                    loads += 1

        return max(loads, 1), max(stores, 1), max(total, 1)
