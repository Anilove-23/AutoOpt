"""
scripts/extract_features_parallel.py
=====================================
Parallel feature extraction for large program sets.
Extracts 20 static features per .cpp program using g++ -O0 -S.
Uses multiprocessing for speed — 10,000 programs in ~5-10 minutes.

Output: data/features_generated.json (list of feature dicts)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Assembly feature extraction (same as extractor.py but C++-aware)
# ─────────────────────────────────────────────────────────────────────────────

_MEM_RE   = re.compile(r'^\s*(mov|lea|push|pop|ldr|str|ld|st)', re.I)
_JUMP_RE  = re.compile(r'^\s*(j[a-z]+|br|b\.)', re.I)
_CALL_RE  = re.compile(r'^\s*call', re.I)
_ARITH_RE = re.compile(r'^\s*(add|sub|imul|mul|idiv|div|xor|and|or|shl|shr|sar|neg|not|inc|dec)', re.I)
_FLOAT_RE = re.compile(r'^\s*(fld|fst|fadd|fsub|fmul|fdiv|movss|movsd|addss|addsd|subss|mulss|mulsd|divss|divsd|vcvt|sqrtss|sqrtsd)', re.I)
_DIRECT   = re.compile(r'^\s*[.#@]')
_LABEL    = re.compile(r'^\s*\w[\w.$]+:')


def _asm_features(asm: str) -> Dict[str, Any]:
    total = mem = jumps = calls = arith = floats = bbs = 0
    for line in asm.splitlines():
        s = line.strip()
        if not s or _DIRECT.match(line): continue
        if _LABEL.match(line): bbs += 1; continue
        total += 1
        if _MEM_RE.match(line):   mem   += 1
        if _JUMP_RE.match(line):  jumps += 1
        if _CALL_RE.match(line):  calls += 1
        if _ARITH_RE.match(line): arith += 1
        if _FLOAT_RE.match(line): floats += 1
    t = max(total, 1)
    return {
        "num_instructions":      total,
        "num_memory_ops":        mem,
        "num_jump_insns":        jumps,
        "num_call_insns":        calls,
        "num_arithmetic":        arith,
        "num_float_ops":         floats,
        "num_basic_blocks":      max(bbs, 1),
        "cyclomatic_complexity": jumps + 1,
        "instruction_mix_ratio": arith / t,
        "memory_intensity":      mem   / t,
        "branch_intensity":      jumps / t,
        "float_ratio":           floats / t,
        "call_density":          calls / t,
    }


def _src_features(src: str) -> Dict[str, Any]:
    # Strip comments
    s = re.sub(r'/\*.*?\*/', ' ', src, flags=re.DOTALL)
    s = re.sub(r'//[^\n]*', ' ', s)
    loc       = sum(1 for l in s.splitlines() if l.strip())
    num_funcs = len(re.findall(r'\b\w[\w\s*<>:]+\s+\w+\s*\([^;{]*\)\s*(?:const\s*)?\{', s))
    num_loops = len(re.findall(r'\b(for|while|do)\b', s))
    max_nest  = 0
    for cand in re.findall(r'(?:(?:for|while|do)\s*\([^)]*\)\s*\{?\s*)+', s):
        max_nest = max(max_nest, len(re.findall(r'\b(?:for|while|do)\b', cand)))
    num_branches   = len(re.findall(r'\b(if|else|switch|case|default)\b', s))
    num_pointers   = len(re.findall(r'\*\s*\w+\s*[,;=)\[]', s))
    name_cnts: Dict[str, int] = {}
    for nm in re.findall(r'\b(\w+)\s*\(', s):
        name_cnts[nm] = name_cnts.get(nm, 0) + 1
    num_recursive = sum(1 for nm, c in name_cnts.items() if c >= 2 and len(nm) > 2)
    return {
        "lines_of_code":       loc,
        "num_functions":       max(num_funcs, 1),
        "num_loops":           num_loops,
        "max_loop_nesting":    max_nest,
        "num_branches":        num_branches,
        "num_pointers":        num_pointers,
        "num_recursive_calls": num_recursive,
    }


def extract_one(args: tuple) -> Dict[str, Any] | None:
    cpp_file, gxx = args
    cpp_file = Path(cpp_file)
    src = cpp_file.read_text(encoding="utf-8", errors="replace")

    feats: Dict[str, Any] = {
        "program":  cpp_file.stem,
        "category": "_".join(cpp_file.stem.split("_")[2:]) if cpp_file.stem.count("_") >= 2 else cpp_file.stem,
    }
    feats.update(_src_features(src))

    with tempfile.NamedTemporaryFile(suffix=".s", delete=False) as tf:
        asm_path = tf.name
    try:
        r = subprocess.run(
            [gxx, "-std=c++17", "-O0", "-S", "-o", asm_path, str(cpp_file)],
            capture_output=True, timeout=60
        )
        if r.returncode == 0:
            asm = Path(asm_path).read_text(encoding="utf-8", errors="replace")
            feats.update(_asm_features(asm))
        else:
            feats.update({k: 0 for k in [
                "num_instructions","num_memory_ops","num_jump_insns","num_call_insns",
                "num_arithmetic","num_float_ops","num_basic_blocks","cyclomatic_complexity",
                "instruction_mix_ratio","memory_intensity","branch_intensity","float_ratio","call_density"
            ]})
    except Exception:
        feats.update({k: 0 for k in [
            "num_instructions","num_memory_ops","num_jump_insns","num_call_insns",
            "num_arithmetic","num_float_ops","num_basic_blocks","cyclomatic_complexity",
            "instruction_mix_ratio","memory_intensity","branch_intensity","float_ratio","call_density"
        ]})
    finally:
        try: os.unlink(asm_path)
        except: pass

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_parallel(
    benchmark_dir: str | Path,
    out_json:      str | Path = "data/features_generated.json",
    gxx:           str        = "g++",
    workers:       int        = 4,
    report_every:  int        = 500,
) -> None:
    benchmark_dir = Path(benchmark_dir)
    out_json      = Path(out_json)
    out_json.parent.mkdir(exist_ok=True)

    cpp_files = sorted(benchmark_dir.rglob("*.cpp"))
    print(f"Extracting features from {len(cpp_files)} programs with {workers} workers ...")

    t0      = time.perf_counter()
    records = []
    done    = 0

    args_list = [(str(f), gxx) for f in cpp_files]

    with mp.Pool(processes=workers) as pool:
        for rec in pool.imap_unordered(extract_one, args_list, chunksize=10):
            if rec is not None:
                records.append(rec)
            done += 1
            if done % report_every == 0 or done == len(cpp_files):
                elapsed = time.perf_counter() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                print(f"  [{done:>6}/{len(cpp_files)}]  {rate:.0f} prog/s  ETA {(len(cpp_files)-done)/rate/60:.1f} min")

    out_json.write_text(json.dumps(records, indent=None), encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(f"\nExtracted {len(records)} programs in {elapsed:.1f}s -> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir", nargs="?", default="benchmarks/generated")
    parser.add_argument("--out",     default="data/features_generated.json")
    parser.add_argument("--gxx",     default="g++")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report",  type=int, default=500)
    args = parser.parse_args()

    mp.freeze_support()
    extract_all_parallel(
        benchmark_dir=args.benchmark_dir,
        out_json=args.out,
        gxx=args.gxx,
        workers=args.workers,
        report_every=args.report,
    )
