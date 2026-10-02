"""
run_pipeline_large.py
=====================
Master script for the 10,000-program training pipeline.

Steps:
  1. Generate 10,000 C++ DSA programs
  2. Extract features in parallel (g++ -O0 -S)
  3. Compile + measure with 8 optimization sequences (parallel)
  4. Assemble large dataset
  5. Train RF + XGBoost on large dataset
  6. Generate visualization plots

Usage:
  python run_pipeline_large.py                        # full run
  python run_pipeline_large.py --skip-generate        # skip step 1 (already generated)
  python run_pipeline_large.py --skip-measure         # skip steps 1-3 (already measured)
  python run_pipeline_large.py --workers 4            # parallel workers (default: 4)
  python run_pipeline_large.py --count 10000          # number of programs to generate
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def step(n, title):
    bar = "=" * 60
    print(f"\n{bar}\n  STEP {n}: {title}\n{bar}")


def main():
    parser = argparse.ArgumentParser(description="10k-program ML compiler optimization pipeline")
    parser.add_argument("--count",          type=int, default=10000)
    parser.add_argument("--workers",        type=int, default=4)
    parser.add_argument("--bench-dir",      default="benchmarks/generated")
    parser.add_argument("--skip-generate",  action="store_true")
    parser.add_argument("--skip-extract",   action="store_true")
    parser.add_argument("--skip-measure",   action="store_true")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    bench_dir    = Path(args.bench_dir)
    feat_json    = Path("data/features_generated.json")
    meas_csv     = Path("data/measurements_generated.csv")
    dataset_csv  = Path("data/dataset_large.csv")
    t_total      = time.perf_counter()

    # ── Step 1: Generate programs ──────────────────────────────────────────────
    if not args.skip_generate and not args.skip_measure:
        step(1, f"Generating {args.count} C++ DSA programs")
        t0 = time.perf_counter()
        from scripts.generate_programs import generate_all
        generate_all(args.count, bench_dir, seed=args.seed)
        print(f"\n  [OK] Generated in {time.perf_counter()-t0:.1f}s")
    else:
        cpp_count = len(list(bench_dir.rglob("*.cpp"))) if bench_dir.exists() else 0
        print(f"\n  [SKIP] Using existing {cpp_count} programs in {bench_dir}")

    # ── Step 2: Feature extraction ─────────────────────────────────────────────
    if not args.skip_extract and not args.skip_measure:
        step(2, "Extracting features (parallel)")
        t0 = time.perf_counter()
        from scripts.extract_features_parallel import extract_all_parallel
        extract_all_parallel(bench_dir, out_json=feat_json,
                             workers=args.workers, report_every=500)
        print(f"\n  [OK] Features extracted in {time.perf_counter()-t0:.1f}s")
    else:
        recs = len(json.loads(feat_json.read_text())) if feat_json.exists() else 0
        print(f"\n  [SKIP] Using existing {recs} feature records")

    # ── Step 3: Measurement ────────────────────────────────────────────────────
    if not args.skip_measure:
        step(3, "Compiling & measuring optimization sequences (parallel)")
        print("  NOTE: This step runs overnight for 10,000 programs.")
        print("        It supports resume — safe to Ctrl+C and restart.\n")
        t0 = time.perf_counter()
        from scripts.measure_parallel import measure_all_parallel
        measure_all_parallel(bench_dir, out_csv=meas_csv,
                             workers=args.workers, report_every=100)
        print(f"\n  [OK] Measurement complete in {(time.perf_counter()-t0)/60:.1f} min")
    else:
        rows = 0
        if meas_csv.exists():
            import pandas as pd
            rows = len(pd.read_csv(meas_csv))
        print(f"\n  [SKIP] Using existing measurements ({rows} rows)")

    # ── Step 4: Dataset assembly ───────────────────────────────────────────────
    step(4, "Assembling large ML dataset")
    t0 = time.perf_counter()
    from scripts.build_dataset_large import build
    df = build()
    print(f"\n  [OK] Dataset built in {time.perf_counter()-t0:.1f}s  ({len(df)} rows)")

    # ── Step 5: Model training ─────────────────────────────────────────────────
    step(5, "Training RF + XGBoost on large dataset")
    t0 = time.perf_counter()
    from ml.train_large import main as train_main
    train_main()
    print(f"\n  [OK] Models trained in {time.perf_counter()-t0:.1f}s")

    # ── Step 6: Plots ─────────────────────────────────────────────────────────
    step(6, "Generating result plots")
    t0 = time.perf_counter()
    try:
        from ml.visualize_large import main as viz_main
        viz_main()
    except ImportError:
        from ml.visualize import main as viz_main
        viz_main()
    print(f"\n  [OK] Plots in {time.perf_counter()-t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Dataset    -> {dataset_csv}")
    print(f"  Models     -> models/rf_large.pkl, models/xgb_large.pkl")
    print(f"  Results    -> results/evaluation_large.csv")
    print(f"  Recommend  -> python cli/recommend.py <file.cpp> --model-dir large")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
