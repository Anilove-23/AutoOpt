"""
scripts/measure.py
──────────────────
Compile every benchmark with every candidate optimization flag set and
measure execution time + binary size.

Optimization candidates (GCC-based, since no LLVM on the machine):
  Each "sequence" is a GCC flag-set — a combination of individual
  optimization flags that correspond to what the LLVM passes would do:

  seq_id  description                flags
  ──────────────────────────────────────────────────────────────────────
  0       none (baseline)            -O0
  1       basic                      -O1
  2       standard                   -O2
  3       aggressive                 -O3
  4       size                       -Os
  5       unroll+vectorize           -O2 -funroll-loops -ftree-vectorize
  6       inline-heavy               -O2 -finline-functions -finline-limit=500
  7       loop-opt                   -O3 -funroll-loops -floop-interchange
  8       no-inline                  -O2 -fno-inline
  9       branch-opt                 -O2 -fbranch-probabilities (skip if profdata missing)
  10      combine                    -O3 -funroll-loops -finline-functions -ftree-vectorize
  11      size+inline                -Os -finline-functions
  12      no-unroll                  -O3 -fno-unroll-loops
  13      fast-math                  -O2 -ffast-math
  14      fast-math+vec              -O3 -ffast-math -ftree-vectorize
  15      lto-style                  -O2 -fipa-pta
  16      aggressive-no-vec         -O3 -fno-tree-vectorize
  17      unroll-only                -O1 -funroll-loops
  18      gcse+pre                   -O2 -fgcse -fpredictive-commoning
  19      full-aggressive            -O3 -ffast-math -funroll-loops -finline-functions -ftree-vectorize

Output: data/measurements.csv  with columns:
  program, seq_id, seq_name, flags, exec_time_s, binary_size_bytes
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import time
import json
from pathlib import Path

# ─────────────────────────────────────────────────────
# Optimization sequence definitions
# ─────────────────────────────────────────────────────

SEQUENCES: list[dict] = [
    {"id": 0,  "name": "O0_baseline",        "flags": "-O0"},
    {"id": 1,  "name": "O1_basic",            "flags": "-O1"},
    {"id": 2,  "name": "O2_standard",         "flags": "-O2"},
    {"id": 3,  "name": "O3_aggressive",       "flags": "-O3"},
    {"id": 4,  "name": "Os_size",             "flags": "-Os"},
    {"id": 5,  "name": "O2_unroll_vec",       "flags": "-O2 -funroll-loops -ftree-vectorize"},
    {"id": 6,  "name": "O2_inline_heavy",     "flags": "-O2 -finline-functions -finline-limit=500"},
    {"id": 7,  "name": "O3_loop_opt",         "flags": "-O3 -funroll-loops"},
    {"id": 8,  "name": "O2_no_inline",        "flags": "-O2 -fno-inline"},
    {"id": 9,  "name": "O2_no_unroll",        "flags": "-O2 -fno-unroll-loops"},
    {"id": 10, "name": "O3_combine",          "flags": "-O3 -funroll-loops -finline-functions -ftree-vectorize"},
    {"id": 11, "name": "Os_inline",           "flags": "-Os -finline-functions"},
    {"id": 12, "name": "O3_no_vec",           "flags": "-O3 -fno-tree-vectorize"},
    {"id": 13, "name": "O2_fast_math",        "flags": "-O2 -ffast-math"},
    {"id": 14, "name": "O3_fast_math_vec",    "flags": "-O3 -ffast-math -ftree-vectorize"},
    {"id": 15, "name": "O2_ipa_pta",          "flags": "-O2 -fipa-pta"},
    {"id": 16, "name": "O3_no_vec2",          "flags": "-O3 -fno-tree-vectorize -fno-unroll-loops"},
    {"id": 17, "name": "O1_unroll",           "flags": "-O1 -funroll-loops"},
    {"id": 18, "name": "O2_gcse_pre",         "flags": "-O2 -fgcse -fpredictive-commoning"},
    {"id": 19, "name": "full_aggressive",     "flags": "-O3 -ffast-math -funroll-loops -finline-functions -ftree-vectorize"},
]

GCC       = "gcc"
RUNS      = 5          # median of N timed runs
TIMEOUT   = 60         # seconds per run

# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def compile_program(src: Path, exe: Path, flags_str: str, gcc: str = GCC) -> bool:
    """Compile src → exe with the given flags. Return True on success."""
    flags = flags_str.split()
    cmd   = [gcc] + flags + [str(src), "-o", str(exe), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def measure_exec_time(exe: Path, runs: int = RUNS, timeout: int = TIMEOUT) -> float | None:
    """Run exe *runs* times and return the median wall-clock time (seconds)."""
    times: list[float] = []
    for _ in range(runs):
        try:
            t0 = time.perf_counter()
            r  = subprocess.run([str(exe)], capture_output=True, timeout=timeout)
            t1 = time.perf_counter()
            if r.returncode == 0:
                times.append(t1 - t0)
        except subprocess.TimeoutExpired:
            pass
    if not times:
        return None
    times.sort()
    return times[len(times) // 2]


def binary_size(exe: Path) -> int:
    """Return the binary file size in bytes."""
    return exe.stat().st_size


# ─────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────

def measure_all(benchmark_dir: str | Path,
                out_csv: str | Path = "data/measurements.csv",
                gcc: str = GCC) -> None:
    benchmark_dir = Path(benchmark_dir)
    out_csv       = Path(out_csv)
    out_csv.parent.mkdir(exist_ok=True)

    c_files = sorted(benchmark_dir.rglob("*.c"))
    print(f"Found {len(c_files)} benchmark(s), {len(SEQUENCES)} sequence(s) each.")
    print(f"Total compilations: {len(c_files) * len(SEQUENCES)}\n")

    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for cf in c_files:
            prog_name = cf.stem
            print(f"[{prog_name}]")
            for seq in SEQUENCES:
                exe = Path(tmpdir) / f"{prog_name}_{seq['id']}.exe"
                print(f"  seq={seq['id']:2d} {seq['name']:<25} ... ", end='', flush=True)

                # Compile
                try:
                    ok = compile_program(cf, exe, seq['flags'], gcc=gcc)
                except Exception as e:
                    print(f"COMPILE-ERROR: {e}")
                    continue

                if not ok:
                    print("COMPILE-FAILED")
                    continue

                # Measure
                t = measure_exec_time(exe)
                sz = binary_size(exe)

                if t is None:
                    print("RUN-TIMEOUT")
                else:
                    print(f"time={t:.4f}s  size={sz}B")

                rows.append({
                    "program":           prog_name,
                    "seq_id":            seq["id"],
                    "seq_name":          seq["name"],
                    "flags":             seq["flags"],
                    "exec_time_s":       t if t is not None else -1.0,
                    "binary_size_bytes": sz,
                })

    # Write CSV
    fieldnames = ["program","seq_id","seq_name","flags","exec_time_s","binary_size_bytes"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    import sys
    bdir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/custom"
    measure_all(bdir)
