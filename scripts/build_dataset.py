"""
scripts/build_dataset.py
------------------------
Merge features (data/features.json) with measurements (data/measurements.csv)
to produce the final ML-ready dataset: data/dataset.csv

Label strategy
--------------
For each program, find the optimization sequence (seq_id) that achieves:
  - lowest median exec_time_s  (primary target: "best_time_seq")
  - smallest binary_size_bytes (secondary target: "best_size_seq")

Also compute:
  - speedup_vs_O3   : exec_time(O3) / exec_time(best_seq)   (>1 = better than O3)
  - size_ratio_vs_O3: size(O3) / size(best_seq)             (>1 = smaller than O3)
  - exec_time_O2, exec_time_O3  for reference

Output columns:
  program, <feature columns>, best_time_seq, best_size_seq,
  speedup_vs_O3, size_ratio_vs_O3, exec_time_O2, exec_time_O3,
  best_time_seq_name, best_size_seq_name
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np


FEATURES_JSON = Path("data/features.json")
MEASUREMENTS_CSV = Path("data/measurements.csv")
DATASET_CSV = Path("data/dataset.csv")

# seq_id for O2 and O3 (must match scripts/measure.py)
O2_ID = 2
O3_ID = 3


def build_dataset(
    features_json: Path = FEATURES_JSON,
    measurements_csv: Path = MEASUREMENTS_CSV,
    out_csv: Path = DATASET_CSV,
) -> pd.DataFrame:

    # -- Load features ----------------------------------------------------------
    feats = json.loads(features_json.read_text())
    df_feat = pd.DataFrame(feats)
    print(f"Features: {len(df_feat)} programs, {df_feat.shape[1]} columns")

    # -- Load measurements -----------------------------------------------------
    df_meas = pd.read_csv(measurements_csv)
    # Remove timed-out / failed runs
    df_meas = df_meas[df_meas["exec_time_s"] > 0].copy()
    print(f"Measurements (valid): {len(df_meas)} rows")

    # -- Per-program labels ----------------------------------------------------
    records = []
    for prog, grp in df_meas.groupby("program"):
        if len(grp) == 0:
            continue

        best_time_row = grp.loc[grp["exec_time_s"].idxmin()]
        best_size_row = grp.loc[grp["binary_size_bytes"].idxmin()]

        o2_rows = grp[grp["seq_id"] == O2_ID]
        o3_rows = grp[grp["seq_id"] == O3_ID]

        exec_o2 = float(o2_rows["exec_time_s"].iloc[0]) if len(o2_rows) > 0 else np.nan
        exec_o3 = float(o3_rows["exec_time_s"].iloc[0]) if len(o3_rows) > 0 else np.nan
        size_o3 = float(o3_rows["binary_size_bytes"].iloc[0]) if len(o3_rows) > 0 else np.nan

        best_time = float(best_time_row["exec_time_s"])
        best_size = float(best_size_row["binary_size_bytes"])

        speedup_vs_o3   = exec_o3 / best_time if (not np.isnan(exec_o3) and best_time > 0) else np.nan
        size_ratio_vs_o3 = size_o3 / best_size if (not np.isnan(size_o3) and best_size > 0) else np.nan

        records.append({
            "program":           prog,
            "best_time_seq":     int(best_time_row["seq_id"]),
            "best_size_seq":     int(best_size_row["seq_id"]),
            "best_time_seq_name": best_time_row["seq_name"],
            "best_size_seq_name": best_size_row["seq_name"],
            "exec_time_O2":      exec_o2,
            "exec_time_O3":      exec_o3,
            "best_exec_time":    best_time,
            "best_size_bytes":   best_size,
            "speedup_vs_O3":     speedup_vs_o3,
            "size_ratio_vs_O3":  size_ratio_vs_o3,
        })

    df_labels = pd.DataFrame(records)
    print(f"Label records: {len(df_labels)}")

    # -- Merge -----------------------------------------------------------------
    df = df_feat.merge(df_labels, on="program", how="inner")
    print(f"Dataset shape: {df.shape}")

    # Drop any asm_error column if present
    if "asm_error" in df.columns:
        df = df.drop(columns=["asm_error"])

    # Fill NaN in feature columns with 0
    feature_cols = [c for c in df.columns if c not in
                    ["program", "best_time_seq", "best_size_seq",
                     "best_time_seq_name", "best_size_seq_name",
                     "exec_time_O2", "exec_time_O3", "best_exec_time",
                     "best_size_bytes", "speedup_vs_O3", "size_ratio_vs_O3"]]
    df[feature_cols] = df[feature_cols].fillna(0)

    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nDataset saved -> {out_csv}")
    print(df[["program", "best_time_seq", "best_time_seq_name", "speedup_vs_O3"]].to_string(index=False))
    return df


if __name__ == "__main__":
    df = build_dataset()
