"""
scripts/build_dataset_large.py
==============================
Builds the ML dataset from large-scale generated program measurements.
Merges features_generated.json + measurements_generated.csv.
Adds category as an extra feature (one-hot encoded).
Saves to data/dataset_large.csv
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

FEAT_JSON  = Path("data/features_generated.json")
MEAS_CSV   = Path("data/measurements_generated.csv")
OUT_CSV    = Path("data/dataset_large.csv")

O3_ID = 3    # must match SEQUENCES in measure_parallel.py

def build():
    # ── Load ─────────────────────────────────────────────────────────────────
    feats = json.loads(FEAT_JSON.read_text())
    df_feat = pd.DataFrame(feats)
    print(f"Features:     {len(df_feat)} programs, {df_feat.shape[1]} cols")

    df_meas = pd.read_csv(MEAS_CSV)
    df_meas = df_meas[df_meas["exec_time_s"] > 0].copy()
    print(f"Measurements: {len(df_meas)} rows (valid)")

    # ── Labels ────────────────────────────────────────────────────────────────
    records = []
    for prog, grp in df_meas.groupby("program"):
        best_t = grp.loc[grp["exec_time_s"].idxmin()]
        best_s = grp.loc[grp["binary_size_bytes"].idxmin()]

        o3_rows = grp[grp["seq_id"] == O3_ID]
        exec_o3 = float(o3_rows["exec_time_s"].iloc[0]) if not o3_rows.empty else np.nan
        size_o3 = float(o3_rows["binary_size_bytes"].iloc[0]) if not o3_rows.empty else np.nan

        best_time = float(best_t["exec_time_s"])
        best_size = float(best_s["binary_size_bytes"])

        speedup = exec_o3 / best_time if (not np.isnan(exec_o3) and best_time > 0) else np.nan
        szratio = size_o3 / best_size if (not np.isnan(size_o3) and best_size > 0) else np.nan

        records.append({
            "program":            prog,
            "best_time_seq":      int(best_t["seq_id"]),
            "best_size_seq":      int(best_s["seq_id"]),
            "best_time_seq_name": best_t["seq_name"],
            "best_size_seq_name": best_s["seq_name"],
            "exec_time_O3":       exec_o3,
            "best_exec_time":     best_time,
            "best_size_bytes":    best_size,
            "speedup_vs_O3":      speedup,
            "size_ratio_vs_O3":   szratio,
        })

    df_labels = pd.DataFrame(records)
    print(f"Label records: {len(df_labels)}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = df_feat.merge(df_labels, on="program", how="inner")
    print(f"Dataset rows: {len(df)}")

    # ── Category encoding ─────────────────────────────────────────────────────
    if "category" in df.columns:
        cat_dummies = pd.get_dummies(df["category"], prefix="cat").astype(int)
        df = pd.concat([df, cat_dummies], axis=1)
        df = df.drop(columns=["category"])
        print(f"Category one-hot cols: {cat_dummies.shape[1]}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    label_cols = ["program","best_time_seq","best_size_seq","best_time_seq_name",
                  "best_size_seq_name","exec_time_O3","best_exec_time",
                  "best_size_bytes","speedup_vs_O3","size_ratio_vs_O3"]
    feat_cols = [c for c in df.columns if c not in label_cols]
    df[feat_cols] = df[feat_cols].fillna(0)

    # Drop programs where best sequence is same as O3 (no benefit to learn)
    # Keep them — they are valid training signal (model learns when O3 IS best)

    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nBest-sequence distribution:")
    print(df["best_time_seq_name"].value_counts().to_string())
    print(f"\nSpeedup vs O3 stats:")
    print(df["speedup_vs_O3"].describe().round(4).to_string())
    print(f"\nWin rate (beats O3): {(df['speedup_vs_O3']>1.0).mean():.1%}")
    return df


if __name__ == "__main__":
    build()
