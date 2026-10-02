"""
features/extractor.py
─────────────────────
Static program-feature extractor that works WITHOUT LLVM on the machine.
It compiles each .c file with GCC at -O0, disassembles the object, and
extracts a rich set of structural features from the assembly / source.

Features extracted
──────────────────
Source-level (parsed from .c):
  - num_functions      : number of function definitions
  - num_loops          : for/while/do loops (rough count)
  - max_loop_nesting   : estimated maximum loop nesting depth
  - num_branches       : if/else/switch/case statements
  - num_recursive_calls: obvious recursive call patterns
  - num_pointers       : pointer declarations (* occurrences in declarations)
  - lines_of_code      : non-blank, non-comment lines

Assembly-level (from GCC -O0 -S):
  - num_instructions   : total asm instruction count (non-directive lines)
  - num_memory_ops     : mov/lea/load/store-class instructions
  - num_jump_insns     : jmp/jne/je/jl/jg/… conditional + unconditional jumps
  - num_call_insns     : call instructions
  - num_arithmetic     : add/sub/mul/imul/div/idiv/xor/and/or/shl/shr
  - num_float_ops      : fld/fst/fadd/fsub/fmul/fdiv/movss/movsd/addss etc.
  - num_basic_blocks   : approximate BB count (jump targets + entry)
  - cyclomatic_complexity: McCabe = num_jumps − num_calls + 2
                           (approximation from asm branch count)
  - instruction_mix_ratio: arithmetic / total_instructions
  - memory_intensity   : memory_ops / total_instructions
  - branch_intensity   : jump_insns / total_instructions
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

# ──────────────────────────────────────────────
# Source-level feature extraction
# ──────────────────────────────────────────────

def _extract_source_features(src: str) -> Dict[str, Any]:
    """Parse C source text for structural features."""
    # Strip block comments
    src_nc = re.sub(r'/\*.*?\*/', ' ', src, flags=re.DOTALL)
    # Strip line comments
    src_nc = re.sub(r'//[^\n]*', ' ', src_nc)
    lines   = src_nc.splitlines()

    # Non-blank lines
    loc = sum(1 for l in lines if l.strip())

    # Function definitions: type name( ...  )  {
    num_functions = len(re.findall(r'\b\w[\w\s\*]+\s+\w+\s*\([^;]*\)\s*\{', src_nc))

    # Loop keywords
    num_loops = len(re.findall(r'\b(for|while|do)\b', src_nc))

    # Estimate nesting: count maximum depth of { following loop keyword
    max_nesting = 0
    depth = 0
    in_loop_stack: list[int] = []
    for ch in src_nc:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth = max(depth - 1, 0)
    # crude nesting: scan for consecutive 'for(' patterns before {
    nesting_cands = re.findall(r'(?:(?:for|while|do)\s*\([^)]*\)\s*\{?\s*)+', src_nc)
    for cand in nesting_cands:
        cnt = len(re.findall(r'\b(?:for|while|do)\b', cand))
        max_nesting = max(max_nesting, cnt)

    # Branch statements
    num_branches = len(re.findall(r'\b(if|else|switch|case|default)\b', src_nc))

    # Pointer declarations (heuristic)
    num_pointers = len(re.findall(r'\*\s*\w+\s*[,;=\)]', src_nc))

    # Recursive calls: function name appears inside its own body
    func_names = re.findall(r'\b(\w+)\s*\(', src_nc)
    name_counts: Dict[str, int] = {}
    for n in func_names:
        name_counts[n] = name_counts.get(n, 0) + 1
    # Functions called ≥2 times include at least one recursive call
    num_recursive = sum(1 for n, c in name_counts.items() if c >= 2 and len(n) > 2)

    return {
        'lines_of_code':       loc,
        'num_functions':       max(num_functions, 1),
        'num_loops':           num_loops,
        'max_loop_nesting':    max_nesting,
        'num_branches':        num_branches,
        'num_pointers':        num_pointers,
        'num_recursive_calls': num_recursive,
    }


# ──────────────────────────────────────────────
# Assembly-level feature extraction
# ──────────────────────────────────────────────

# Regex sets for x86/x64 assembly mnemonics
_MEM_RE    = re.compile(r'^\s*(mov|lea|push|pop|ldr|str|ld|st)', re.I)
_JUMP_RE   = re.compile(r'^\s*(j[a-z]+|br|b\.)', re.I)
_CALL_RE   = re.compile(r'^\s*call', re.I)
_ARITH_RE  = re.compile(r'^\s*(add|sub|imul|mul|idiv|div|xor|and|or|shl|shr|sar|neg|not|inc|dec)', re.I)
_FLOAT_RE  = re.compile(r'^\s*(fld|fst|fadd|fsub|fmul|fdiv|fild|movss|movsd|addss|addsd|subss|mulss|mulsd|divss|divsd|vcvt)', re.I)
_DIRECTIVE = re.compile(r'^\s*[.#@]')
_LABEL_RE  = re.compile(r'^\s*\w[\w.]+:')


def _extract_asm_features(asm_text: str) -> Dict[str, Any]:
    """Parse GCC -O0 assembly text for instruction-level features."""
    total = mem = jumps = calls = arith = floats = bbs = 0

    for line in asm_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _DIRECTIVE.match(line):
            continue
        if _LABEL_RE.match(line):
            bbs += 1
            continue
        # It's an instruction
        total += 1
        if _MEM_RE.match(line):   mem   += 1
        if _JUMP_RE.match(line):  jumps += 1
        if _CALL_RE.match(line):  calls += 1
        if _ARITH_RE.match(line): arith += 1
        if _FLOAT_RE.match(line): floats += 1

    # McCabe approximation: E - N + 2 ≈ branches (conditional jumps) + 1
    cond_jumps = jumps  # conservative — treat all jumps as potentially conditional
    cyclomatic = cond_jumps + 1

    safe_total = max(total, 1)
    return {
        'num_instructions':      total,
        'num_memory_ops':        mem,
        'num_jump_insns':        jumps,
        'num_call_insns':        calls,
        'num_arithmetic':        arith,
        'num_float_ops':         floats,
        'num_basic_blocks':      max(bbs, 1),
        'cyclomatic_complexity': cyclomatic,
        'instruction_mix_ratio': arith / safe_total,
        'memory_intensity':      mem   / safe_total,
        'branch_intensity':      jumps / safe_total,
        'float_ratio':           floats / safe_total,
        'call_density':          calls / safe_total,
    }


# ──────────────────────────────────────────────
# Top-level: compile + extract
# ──────────────────────────────────────────────

def extract_features(c_file: str | Path, gcc: str = 'gcc') -> Dict[str, Any]:
    """
    Extract static features from a C source file.

    Parameters
    ----------
    c_file : path to .c source
    gcc    : path to gcc executable

    Returns
    -------
    dict of feature_name → value
    """
    c_file = Path(c_file).resolve()
    if not c_file.exists():
        raise FileNotFoundError(f"Source file not found: {c_file}")

    src = c_file.read_text(encoding='utf-8', errors='replace')
    feats: Dict[str, Any] = {'program': c_file.stem}

    # 1. Source-level features
    feats.update(_extract_source_features(src))

    # 2. Compile to assembly with -O0 (no optimisation so features describe raw program)
    with tempfile.NamedTemporaryFile(suffix='.s', delete=False) as tf:
        asm_path = tf.name

    try:
        result = subprocess.run(
            [gcc, '-O0', '-S', '-o', asm_path, str(c_file)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            # Fallback: return only source-level features with zeros for asm
            feats.update({k: 0 for k in [
                'num_instructions', 'num_memory_ops', 'num_jump_insns',
                'num_call_insns', 'num_arithmetic', 'num_float_ops',
                'num_basic_blocks', 'cyclomatic_complexity',
                'instruction_mix_ratio', 'memory_intensity',
                'branch_intensity', 'float_ratio', 'call_density',
            ]})
            feats['asm_error'] = result.stderr[:200]
            return feats

        asm_text = Path(asm_path).read_text(encoding='utf-8', errors='replace')
        feats.update(_extract_asm_features(asm_text))

    finally:
        try:
            os.unlink(asm_path)
        except OSError:
            pass

    return feats


# ──────────────────────────────────────────────
# Batch extraction
# ──────────────────────────────────────────────

def extract_all(benchmark_dir: str | Path, gcc: str = 'gcc') -> list[Dict[str, Any]]:
    """Extract features for every .c file under *benchmark_dir*."""
    benchmark_dir = Path(benchmark_dir)
    c_files = sorted(benchmark_dir.rglob('*.c'))
    records = []
    for cf in c_files:
        print(f"  Extracting: {cf.name} ...", end=' ', flush=True)
        try:
            rec = extract_features(cf, gcc=gcc)
            records.append(rec)
            print("OK")
        except Exception as exc:
            print(f"ERROR: {exc}")
    return records


if __name__ == '__main__':
    import sys
    bench_dir = sys.argv[1] if len(sys.argv) > 1 else 'benchmarks/custom'
    records = extract_all(bench_dir)
    out = Path('data/features.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(f"\nSaved {len(records)} records to {out}")
