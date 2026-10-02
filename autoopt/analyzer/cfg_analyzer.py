"""
autoopt.analyzer.cfg_analyzer
=============================
Control Flow Graph (CFG) and Dominance Structure Analyzer.
Inspired by LLVM passes:
  - domtree: Dominator Tree Construction
  - domfrontier: Dominance Frontier Construction
  - dot-cfg: Control Flow Graph structure
  - jump-threading & simplifycfg: branch topology analysis
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class CFGMetrics:
    basic_blocks: int
    edges_estimate: int
    cyclomatic_complexity: int
    branch_count: int
    branch_density: float
    conditional_jumps: int
    unconditional_jumps: int
    switch_statements: int
    diamond_branches: int
    loop_backedges: int


class CFGAnalyzer:
    """Analyzes control flow graph topology from C/C++ source and unoptimized assembly."""

    BRANCH_KEYWORDS = re.compile(r'\b(if|else\s+if|else|switch|case|default)\b')
    COND_JUMPS = re.compile(r'^\s*j(e|ne|g|ge|l|le|a|ae|b|be|z|nz)\b', re.IGNORECASE)
    UNCOND_JUMPS = re.compile(r'^\s*jmp\b', re.IGNORECASE)
    LABELS = re.compile(r'^\s*([A-Za-z0-9_$.]+):')

    def analyze(self, source_code: str, assembly_code: str = "") -> CFGMetrics:
        # Strip comments
        src = re.sub(r'/\*.*?\*/', ' ', source_code, flags=re.DOTALL)
        src = re.sub(r'//[^\n]*', ' ', src)

        # Source-level branch features
        branch_matches = self.BRANCH_KEYWORDS.findall(src)
        branch_count = len(branch_matches)
        switch_count = len(re.findall(r'\bswitch\s*\(', src))

        # Diamond pattern proxy: 'if (...) { ... } else { ... }'
        diamond_matches = len(re.findall(r'\bif\s*\([^)]+\)\s*\{[^}]*\}\s*else\s*\{', src))

        # Assembly-level CFG features if available
        if assembly_code:
            bbs, cond_j, uncond_j, backedges = self._parse_assembly_cfg(assembly_code)
        else:
            bbs = max(1, branch_count * 2 + 1)
            cond_j = branch_count
            uncond_j = max(0, branch_count - 1)
            backedges = len(re.findall(r'\b(for|while|do)\b', src))

        # Cyclomatic complexity: M = E - N + 2P (approx: conditional branches + 1)
        cyclomatic = max(1, cond_j + 1)
        total_jumps = cond_j + uncond_j
        edges = bbs + total_jumps

        total_lines = max(1, len([l for l in src.splitlines() if l.strip()]))
        branch_density = branch_count / total_lines

        return CFGMetrics(
            basic_blocks=bbs,
            edges_estimate=edges,
            cyclomatic_complexity=cyclomatic,
            branch_count=branch_count,
            branch_density=branch_density,
            conditional_jumps=cond_j,
            unconditional_jumps=uncond_j,
            switch_statements=switch_count,
            diamond_branches=diamond_matches,
            loop_backedges=backedges,
        )

    def _parse_assembly_cfg(self, asm: str) -> tuple[int, int, int, int]:
        labels: List[str] = []
        cond_jumps = 0
        uncond_jumps = 0
        backedges = 0

        lines = asm.splitlines()
        for idx, line in enumerate(lines):
            st = line.strip()
            if not st or st.startswith(('#', '.', '@')):
                continue

            lbl_match = self.LABELS.match(line)
            if lbl_match:
                labels.append(lbl_match.group(1))
                continue

            if self.COND_JUMPS.match(st):
                cond_jumps += 1
                # Check for loop backedge (jump to an earlier label)
                parts = st.split()
                if len(parts) >= 2 and parts[1] in labels:
                    backedges += 1
            elif self.UNCOND_JUMPS.match(st):
                uncond_jumps += 1

        basic_blocks = max(1, len(labels))
        return basic_blocks, cond_jumps, uncond_jumps, backedges
