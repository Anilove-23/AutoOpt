"""
run_pipeline.py
---------------
Master orchestration script -- runs the entire pipeline end-to-end:

  Step 1: Extract features from all benchmarks
  Step 2: Compile + measure all optimization sequences
  Step 3: Assemble dataset (merge features + labels)
  Step 4: Train ML models
  Step 5: Generate all visualisation plots

Run:
  python run_pipeline.py [--benchmark-dir benchmarks/custom] [--gcc gcc]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _step(n: int, title: str):
    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  STEP {n}: {title}")
    print(f"{bar}")


def main():
    parser = argparse.ArgumentParser(description="ML Compiler Optimization -- full pipeline")
    parser.add_argument("--benchmark-dir", default="benchmarks/custom",
                        help="Directory containing benchmark .c files")
    parser.add_argument("--gcc", default="gcc",
                        help="GCC executable (default: gcc)")
    parser.add_argument("--skip-measure", action="store_true",
                        help="Skip measurement if measurements.csv already exists")
    args = parser.parse_args()

    bench_dir = Path(args.benchmark_dir)
    t_total   = time.perf_counter()

    # ------------------------------------------------------------
    # Step 1 -- Feature Extraction
    # ------------------------------------------------------------
    _step(1, "Extracting program features")
    t0 = time.perf_counter()

    from features.extractor import extract_all
    import json

    records = extract_all(bench_dir, gcc=args.gcc)
    out = Path("data/features.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(f"\n  OK {len(records)} programs extracted -> {out}  [{time.perf_counter()-t0:.1f}s]")

    # ------------------------------------------------------------
    # Step 2 -- Measurement
    # ------------------------------------------------------------
    meas_csv = Path("data/measurements.csv")
    if args.skip_measure and meas_csv.exists():
        print(f"\n  >> Skipping measurement (--skip-measure, using existing {meas_csv})")
    else:
        _step(2, "Compiling & measuring optimization sequences")
        t0 = time.perf_counter()
        from scripts.measure import measure_all
        measure_all(bench_dir, out_csv=meas_csv, gcc=args.gcc)
        print(f"\n  OK Measurements complete -> {meas_csv}  [{time.perf_counter()-t0:.1f}s]")

    # ------------------------------------------------------------
    # Step 3 -- Dataset Assembly
    # ------------------------------------------------------------
    _step(3, "Assembling ML dataset")
    t0 = time.perf_counter()
    from scripts.build_dataset import build_dataset
    df = build_dataset()
    print(f"\n  OK Dataset: {len(df)} rows  [{time.perf_counter()-t0:.1f}s]")

    # ------------------------------------------------------------
    # Step 4 -- Model Training
    # ------------------------------------------------------------
    _step(4, "Training ML models (RF + XGBoost)")
    t0 = time.perf_counter()
    from ml.train import main as train_main
    train_main()
    print(f"\n  OK Models trained  [{time.perf_counter()-t0:.1f}s]")

    # ------------------------------------------------------------
    # Step 5 -- Visualisation
    # ------------------------------------------------------------
    _step(5, "Generating result plots")
    t0 = time.perf_counter()
    from ml.visualize import main as viz_main
    viz_main()
    print(f"\n  OK Plots saved to results/plots/  [{time.perf_counter()-t0:.1f}s]")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*55}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Models    -> models/")
    print(f"  Results   -> results/evaluation.csv")
    print(f"  Plots     -> results/plots/")
    print(f"  Recommend -> python cli/recommend.py <file.c>")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
