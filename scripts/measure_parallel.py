"""
scripts/measure_parallel.py
===========================
Parallel compile-and-measure for large benchmark sets (10,000+ programs).
Uses multiprocessing.Pool for compilation and timing.

Design choices for scale
------------------------
- 8 key optimization sequences (reduced from 20 for speed)
- 3 timed runs per program per sequence (median taken)
- 4 parallel worker processes (tune --workers for your machine)
- Each worker gets its own temp directory
- Progress is printed every --report-interval programs
- Skips programs that already exist in the CSV (resume support)
- Compiler: g++ (C++17)

Expected runtime for 10,000 programs:
  - 8 sequences * 3 runs = 24 executions per program
  - ~0.2s average per execution = 4.8s per program
  - With 4 workers: ~12,000s / 4 = ~3,300s = ~1 hour
  (varies hugely by benchmark complexity and machine speed)

Output: data/measurements_generated.csv
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Key optimization sequences (8 carefully chosen sequences)
# ─────────────────────────────────────────────────────────────────────────────

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

GXX     = "g++"
RUNS    = 3
TIMEOUT = 30   # seconds per run — generous for recursive programs


# ─────────────────────────────────────────────────────────────────────────────
# Per-program worker function (runs in child process)
# ─────────────────────────────────────────────────────────────────────────────

def _measure_one(args: tuple) -> list[dict]:
    cpp_file, tmpdir, gxx = args
    cpp_file = Path(cpp_file)
    prog_name = cpp_file.stem
    rows: list[dict] = []

    for seq in SEQUENCES:
        exe = Path(tmpdir) / f"{prog_name}_{seq['id']}.exe"
        flags = seq["flags"].split()
        cmd   = [gxx, "-std=c++17"] + flags + [str(cpp_file), "-o", str(exe), "-lm"]

        # Compile
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
            if r.returncode != 0:
                continue
        except Exception:
            continue

        # Measure
        times: list[float] = []
        sz = exe.stat().st_size if exe.exists() else 0

        for _ in range(RUNS):
            try:
                t0 = time.perf_counter()
                r2 = subprocess.run([str(exe)], capture_output=True, timeout=TIMEOUT)
                t1 = time.perf_counter()
                if r2.returncode == 0:
                    times.append(t1 - t0)
            except subprocess.TimeoutExpired:
                break
            except Exception:
                break

        # Remove executable to avoid eating gigabytes of disk space
        try:
            if exe.exists():
                exe.unlink()
        except Exception:
            pass

        if not times:
            exec_t = -1.0
        else:
            times.sort()
            exec_t = times[len(times) // 2]

        rows.append({
            "program":           prog_name,
            "seq_id":            seq["id"],
            "seq_name":          seq["name"],
            "flags":             seq["flags"],
            "exec_time_s":       exec_t,
            "binary_size_bytes": sz,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def measure_all_parallel(
    benchmark_dir: str | Path,
    out_csv:       str | Path  = "data/measurements_generated.csv",
    gxx:           str         = GXX,
    workers:       int         = 4,
    report_every:  int         = 100,
) -> None:
    benchmark_dir = Path(benchmark_dir)
    out_csv       = Path(out_csv)
    out_csv.parent.mkdir(exist_ok=True)

    cpp_files = sorted(benchmark_dir.rglob("*.cpp"))
    print(f"Found {len(cpp_files)} .cpp files | {len(SEQUENCES)} sequences | {workers} workers", flush=True)
    print(f"Estimated executions: {len(cpp_files) * len(SEQUENCES) * RUNS:,}", flush=True)

    # Load already-measured programs (for resume)
    done_progs: set[str] = set()
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            for row in csv.DictReader(f):
                done_progs.add(row["program"])
        print(f"Resuming: {len(done_progs)} programs already measured, skipping them.", flush=True)
        cpp_files = [f for f in cpp_files if f.stem not in done_progs]
        print(f"Remaining: {len(cpp_files)} programs", flush=True)

    if not cpp_files:
        print("Nothing to do.", flush=True)
        return

    # Write CSV header if new file
    fieldnames = ["program", "seq_id", "seq_name", "flags", "exec_time_s", "binary_size_bytes"]
    write_header = not out_csv.exists() or os.path.getsize(out_csv) == 0
    csv_file = open(out_csv, "a", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    t_global   = time.perf_counter()
    done_count = 0
    total      = len(cpp_files)

    with tempfile.TemporaryDirectory() as tmpdir:
        work_args = [(str(f), tmpdir, gxx) for f in cpp_files]

        with mp.Pool(processes=workers) as pool:
            for rows in pool.imap_unordered(_measure_one, work_args, chunksize=1):
                writer.writerows(rows)
                csv_file.flush()
                done_count += 1

                if done_count % report_every == 0 or done_count == total:
                    elapsed  = time.perf_counter() - t_global
                    rate     = done_count / elapsed if elapsed > 0 else 0
                    eta      = (total - done_count) / rate if rate > 0 else 0
                    valid    = sum(1 for r in rows if r["exec_time_s"] > 0)
                    print(
                        f"  [{done_count:>6}/{total}] "
                        f"{done_count*100//total:>3}%  "
                        f"{rate:.1f} prog/s  "
                        f"ETA {eta/60:.1f} min  "
                        f"last_prog={rows[0]['program'] if rows else '?'}  ok={valid}/{len(SEQUENCES)}",
                        flush=True
                    )

    csv_file.close()
    elapsed = time.perf_counter() - t_global
    print(f"\nDone! {done_count} programs in {elapsed/60:.1f} min -> {out_csv}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel optimization measurement")
    parser.add_argument("benchmark_dir", nargs="?", default="benchmarks/generated")
    parser.add_argument("--out",      default="data/measurements_generated.csv")
    parser.add_argument("--gxx",      default="g++")
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--report",   type=int, default=100,
                        help="Print progress every N programs")
    args = parser.parse_args()

    # Windows multiprocessing safety
    mp.freeze_support()
    measure_all_parallel(
        benchmark_dir=args.benchmark_dir,
        out_csv=args.out,
        gxx=args.gxx,
        workers=args.workers,
        report_every=args.report,
    )
