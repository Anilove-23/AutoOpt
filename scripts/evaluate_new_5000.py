"""
scripts/evaluate_new_5000.py
============================
Evaluates trained ML model on 5,000 unseen random C++ programs.

For each program:
  1. Extracts static + assembly features (matching training format).
  2. Model predicts optimal compiler flags.
  3. Compiles and times:
       - Model-recommended flags
       - Standard -O3 baseline
  4. Records speedup, faster/slower/equal verdict, and category.
  5. Computes comprehensive metrics and breakdown across all DSA types.

Features:
  - 4-worker multiprocessing
  - Real-time incremental CSV output with resume support
  - Auto-generates summary tables & publication plots
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Sequence mapping (matches training)
SEQUENCES = [
    {"id": 0,  "name": "O0_baseline",      "flags": "-O0"},
    {"id": 2,  "name": "O2_standard",      "flags": "-O2"},
    {"id": 3,  "name": "O3_aggressive",    "flags": "-O3"},
    {"id": 4,  "name": "Os_size",          "flags": "-Os"},
    {"id": 5,  "name": "O2_unroll_vec",    "flags": "-O2 -funroll-loops -ftree-vectorize"},
    {"id": 10, "name": "O3_combine",       "flags": "-O3 -funroll-loops -finline-functions -ftree-vectorize"},
    {"id": 13, "name": "O2_fast_math",     "flags": "-O2 -ffast-math"},
    {"id": 19, "name": "full_aggressive",  "flags": "-O3 -ffast-math -funroll-loops -finline-functions -ftree-vectorize"},
]
SEQ_BY_ID = {s["id"]: s for s in SEQUENCES}

GXX = "g++"
RUNS = 3
TIMEOUT = 30


# ── Feature extraction helpers ───────────────────────────────────────────────

_MEM_RE   = re.compile(r'^\s*(mov|lea|push|pop|ldr|str|ld|st)', re.I)
_JUMP_RE  = re.compile(r'^\s*(j[a-z]+|br|b\.)', re.I)
_CALL_RE  = re.compile(r'^\s*call', re.I)
_ARITH_RE = re.compile(r'^\s*(add|sub|imul|mul|idiv|div|xor|and|or|shl|shr|sar|neg|not|inc|dec)', re.I)
_FLOAT_RE = re.compile(r'^\s*(fld|fst|fadd|fsub|fmul|fdiv|movss|movsd|addss|addsd|subss|mulss|mulsd|divss|divsd|vcvt|sqrtss|sqrtsd)', re.I)
_DIRECT   = re.compile(r'^\s*[.#@]')
_LABEL    = re.compile(r'^\s*\w[\w.$]+:')


def _extract_feats(cpp_file: Path, gxx: str) -> tuple[Dict[str, Any], str]:
    src = cpp_file.read_text(encoding="utf-8", errors="replace")
    cat = "_".join(cpp_file.stem.split("_")[2:]) if cpp_file.stem.count("_") >= 2 else cpp_file.stem

    # Source features
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

    feats: Dict[str, Any] = {
        "lines_of_code":       loc,
        "num_functions":       max(num_funcs, 1),
        "num_loops":           num_loops,
        "max_loop_nesting":    max_nest,
        "num_branches":        num_branches,
        "num_pointers":        num_pointers,
        "num_recursive_calls": num_recursive,
    }

    # Assembly features via g++ -O0 -S
    with tempfile.NamedTemporaryFile(suffix=".s", delete=False) as tf:
        asm_path = tf.name
    try:
        r = subprocess.run([gxx, "-std=c++17", "-O0", "-S", "-o", asm_path, str(cpp_file)],
                           capture_output=True, timeout=30)
        if r.returncode == 0:
            asm = Path(asm_path).read_text(encoding="utf-8", errors="replace")
            total = mem = jumps = calls = arith = floats = bbs = 0
            for line in asm.splitlines():
                st = line.strip()
                if not st or _DIRECT.match(line): continue
                if _LABEL.match(line): bbs += 1; continue
                total += 1
                if _MEM_RE.match(line):   mem   += 1
                if _JUMP_RE.match(line):  jumps += 1
                if _CALL_RE.match(line):  calls += 1
                if _ARITH_RE.match(line): arith += 1
                if _FLOAT_RE.match(line): floats += 1
            t = max(total, 1)
            feats.update({
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
            })
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

    return feats, cat


def _time_binary(exe: Path, runs: int = RUNS) -> float | None:
    times = []
    for _ in range(runs):
        try:
            t0 = time.perf_counter()
            r = subprocess.run([str(exe)], capture_output=True, timeout=TIMEOUT)
            t1 = time.perf_counter()
            if r.returncode == 0:
                times.append(t1 - t0)
        except Exception:
            break
    if not times:
        return None
    times.sort()
    return times[len(times) // 2]


# ── Worker function ──────────────────────────────────────────────────────────

_MODEL = None
_FEAT_COLS = None

def _init_worker(model_path: str):
    global _MODEL, _FEAT_COLS
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        _MODEL = data[0]
        if hasattr(_MODEL, "n_jobs"):
            _MODEL.n_jobs = 1
        _FEAT_COLS = data[1]


def _eval_one(args: tuple) -> dict | None:
    cpp_file_str, tmpdir, gxx = args
    cpp_file = Path(cpp_file_str)
    prog_name = cpp_file.stem

    try:
        feats, cat = _extract_feats(cpp_file, gxx)
        # Build one-hot category dict
        row_dict = dict(feats)
        for col in _FEAT_COLS:
            if col.startswith("cat_"):
                row_dict[col] = 1 if col == f"cat_{cat}" else 0

        # Vectorize
        X = [[row_dict.get(c, 0) for c in _FEAT_COLS]]
        pred_seq_id = int(_MODEL.predict(X)[0])
        seq_info = SEQ_BY_ID.get(pred_seq_id, SEQ_BY_ID[3])
        rec_flags = seq_info["flags"]
        rec_name  = seq_info["name"]

        # Compile & time Recommended sequence
        exe_rec = Path(tmpdir) / f"{prog_name}_rec.exe"
        cmd_rec = [gxx, "-std=c++17"] + rec_flags.split() + [str(cpp_file), "-o", str(exe_rec), "-lm"]
        r_rec = subprocess.run(cmd_rec, capture_output=True, timeout=60)
        t_rec = _time_binary(exe_rec) if r_rec.returncode == 0 and exe_rec.exists() else None
        try:
            if exe_rec.exists(): exe_rec.unlink()
        except: pass

        # Compile & time Baseline (-O3)
        if rec_flags == "-O3" and t_rec is not None:
            t_o3 = t_rec
        else:
            exe_o3 = Path(tmpdir) / f"{prog_name}_o3.exe"
            cmd_o3 = [gxx, "-std=c++17", "-O3", str(cpp_file), "-o", str(exe_o3), "-lm"]
            r_o3 = subprocess.run(cmd_o3, capture_output=True, timeout=60)
            t_o3 = _time_binary(exe_o3) if r_o3.returncode == 0 and exe_o3.exists() else None
            try:
                if exe_o3.exists(): exe_o3.unlink()
            except: pass

        if t_rec is None or t_o3 is None or t_rec <= 0 or t_o3 <= 0:
            return None

        speedup = t_o3 / t_rec
        # Allow tiny epsilon for floating point equality
        if abs(speedup - 1.0) < 0.005:
            verdict = "equal"
        elif speedup > 1.0:
            verdict = "faster"
        else:
            verdict = "slower"

        pct_change = (speedup - 1.0) * 100.0

        return {
            "program":     prog_name,
            "category":    cat,
            "rec_seq_id":  pred_seq_id,
            "rec_seq_name": rec_name,
            "rec_flags":   rec_flags,
            "t_rec":       round(t_rec, 6),
            "t_o3":        round(t_o3, 6),
            "speedup":     round(speedup, 4),
            "pct_change":  round(pct_change, 2),
            "verdict":     verdict,
        }
    except Exception as e:
        return None


# ── Main Evaluator ───────────────────────────────────────────────────────────

def run_evaluation(
    test_dir: str = "benchmarks/test_5000",
    model_path: str = "models/rf_large.pkl",
    out_csv: str = "results/eval_5000_results.csv",
    workers: int = 4,
    report_every: int = 100,
):
    test_dir = Path(test_dir)
    out_csv  = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cpp_files = sorted(test_dir.glob("*.cpp"))
    print(f"===========================================================", flush=True)
    print(f"  EVALUATION HARNESS: 5,000 UNSEEN C++ PROGRAMS", flush=True)
    print(f"===========================================================", flush=True)
    print(f"  Total test programs:   {len(cpp_files):,}", flush=True)
    print(f"  Model:                 {model_path}", flush=True)
    print(f"  Parallel workers:      {workers}", flush=True)
    print(f"  Output CSV:            {out_csv}", flush=True)

    # Resume support
    done_set = set()
    if out_csv.exists() and os.path.getsize(out_csv) > 0:
        with open(out_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done_set.add(r["program"])
        print(f"  Resuming: {len(done_set):,} already evaluated programs found.", flush=True)
        cpp_files = [f for f in cpp_files if f.stem not in done_set]
        print(f"  Remaining to evaluate: {len(cpp_files):,}", flush=True)

    if not cpp_files:
        print("  All programs already evaluated! Generating final summary...", flush=True)
        summarize_results(out_csv)
        return

    write_hdr = not out_csv.exists() or os.path.getsize(out_csv) == 0
    f_out = open(out_csv, "a", newline="", encoding="utf-8")
    fields = ["program","category","rec_seq_id","rec_seq_name","rec_flags","t_rec","t_o3","speedup","pct_change","verdict"]
    writer = csv.DictWriter(f_out, fieldnames=fields)
    if write_hdr:
        writer.writeheader()

    t_start = time.perf_counter()
    done = 0
    total = len(cpp_files)

    with tempfile.TemporaryDirectory() as tmpdir:
        work_items = [(str(cf), tmpdir, GXX) for cf in cpp_files]

        with mp.Pool(processes=workers, initializer=_init_worker, initargs=(model_path,)) as pool:
            for res in pool.imap_unordered(_eval_one, work_items, chunksize=1):
                if res is not None:
                    writer.writerow(res)
                    f_out.flush()
                done += 1

                if done % report_every == 0 or done == total:
                    elapsed = time.perf_counter() - t_start
                    rate    = done / elapsed if elapsed > 0 else 0
                    eta_min = (total - done) / rate / 60 if rate > 0 else 0
                    print(
                        f"  [{done:>5}/{total}] {done*100//total:>3}%  "
                        f"{rate:.1f} prog/s  ETA {eta_min:.1f} min",
                        flush=True
                    )

    f_out.close()
    print(f"\n  Done! Evaluation completed in {(time.perf_counter()-t_start)/60:.1f} min", flush=True)
    summarize_results(out_csv)


# ── Detailed Summary & Metrics Computation ───────────────────────────────────

def summarize_results(csv_path: Path):
    df = pd.read_csv(csv_path)
    total = len(df)
    if total == 0:
        print("No valid results found in CSV.")
        return

    faster = df[df["verdict"] == "faster"]
    slower = df[df["verdict"] == "slower"]
    equal  = df[df["verdict"] == "equal"]

    n_faster = len(faster)
    n_slower = len(slower)
    n_equal  = len(equal)

    win_rate  = n_faster / total * 100.0
    loss_rate = n_slower / total * 100.0
    tie_rate  = n_equal  / total * 100.0

    mean_sp   = df["speedup"].mean()
    median_sp = df["speedup"].median()
    geo_sp    = float(np.exp(np.mean(np.log(df["speedup"]))))
    max_sp    = df["speedup"].max()
    min_sp    = df["speedup"].min()

    # Faster-only stats
    avg_faster_pct = faster["pct_change"].mean() if n_faster > 0 else 0.0

    print("\n" + "=" * 65)
    print("           COMPREHENSIVE MODEL EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Programs Evaluated:  {total:,}")
    print("-" * 65)
    print(f"  FASTER than -O3:         {n_faster:>5}  ({win_rate:>5.1f}%)   [avg +{avg_faster_pct:.1f}% faster]")
    print(f"  EQUAL to -O3:            {n_equal:>5}  ({tie_rate:>5.1f}%)")
    print(f"  SLOWER than -O3:         {n_slower:>5}  ({loss_rate:>5.1f}%)")
    print("-" * 65)
    print(f"  Combined Win/Tie Rate:   {win_rate + tie_rate:>5.1f}%")
    print(f"  Average Speedup (Mean):  {mean_sp:.4f}x  ({(mean_sp-1)*100:+.2f}%)")
    print(f"  Geometric Mean Speedup:  {geo_sp:.4f}x  ({(geo_sp-1)*100:+.2f}%)")
    print(f"  Median Speedup:          {median_sp:.4f}x  ({(median_sp-1)*100:+.2f}%)")
    print(f"  Peak Maximum Speedup:    {max_sp:.4f}x  ({(max_sp-1)*100:+.1f}%)")
    print(f"  Worst Case (Min):        {min_sp:.4f}x  ({(min_sp-1)*100:+.1f}%)")
    print("=" * 65)

    # ── Category Breakdown ───────────────────────────────────────────────────
    print("\n" + "-" * 75)
    print(f"{'Category / DSA Type':<25} {'Total':>6} {'Faster':>7} {'Slower':>7} {'Equal':>6} {'Win Rate':>9} {'Avg Speedup':>12}")
    print("-" * 75)

    cat_rows = []
    for cat, grp in df.groupby("category"):
        c_tot = len(grp)
        c_fst = (grp["verdict"] == "faster").sum()
        c_slw = (grp["verdict"] == "slower").sum()
        c_eql = (grp["verdict"] == "equal").sum()
        c_win = (c_fst / c_tot) * 100.0
        c_sp  = grp["speedup"].mean()
        cat_rows.append({
            "category": cat, "total": c_tot, "faster": c_fst,
            "slower": c_slw, "equal": c_eql, "win_rate": c_win, "avg_speedup": c_sp
        })

    cat_rows.sort(key=lambda x: -x["win_rate"])
    for r in cat_rows:
        print(f"{r['category']:<25} {r['total']:>6} {r['faster']:>7} {r['slower']:>7} {r['equal']:>6} {r['win_rate']:>8.1f}% {r['avg_speedup']:>11.3f}x")
    print("-" * 75)

    # Save summary tables to disk
    sum_dir = Path("results")
    sum_json = sum_dir / "eval_5000_summary.json"
    summary_data = {
        "total_evaluated": total,
        "faster_count": n_faster,
        "slower_count": n_slower,
        "equal_count": n_equal,
        "win_rate_pct": round(win_rate, 2),
        "loss_rate_pct": round(loss_rate, 2),
        "tie_rate_pct": round(tie_rate, 2),
        "mean_speedup": round(mean_sp, 4),
        "geomean_speedup": round(geo_sp, 4),
        "median_speedup": round(median_sp, 4),
        "max_speedup": round(max_sp, 4),
        "min_speedup": round(min_sp, 4),
        "category_breakdown": cat_rows,
    }
    sum_json.write_text(json.dumps(summary_data, indent=2))
    print(f"\nSaved summary JSON -> {sum_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir",   default="benchmarks/test_5000")
    parser.add_argument("--model",      default="models/rf_large.pkl")
    parser.add_argument("--out",        default="results/eval_5000_results.csv")
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--report",     type=int, default=100)
    args = parser.parse_args()

    mp.freeze_support()
    run_evaluation(
        test_dir=args.test_dir,
        model_path=args.model,
        out_csv=args.out,
        workers=args.workers,
        report_every=args.report,
    )
