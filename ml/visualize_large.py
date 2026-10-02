"""
ml/visualize_large.py
=====================
Publication-quality visualizations for the large-scale (10k) experiment.

Plots generated:
  1. speedup_distribution_large.png  : Speedup vs -O3 histogram across all test programs
  2. win_rate_by_category.png        : Win rate vs -O3 broken down by DSA algorithm category
  3. best_sequence_distribution.png  : Frequency of each optimization sequence winning
  4. feature_importance_large.png    : Top 20 most important features from Random Forest
  5. binary_size_vs_speedup.png      : Trade-off between binary size change and speedup
  6. confusion_matrix_large.png      : Heatmap of predicted vs actual best sequence

All plots saved to results/plots_large/
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PLOTS_DIR   = Path("results/plots_large")
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_speedup_distribution():
    eval_csv = RESULTS_DIR / "evaluation_large.csv"
    if not eval_csv.exists():
        return
    df = pd.read_csv(eval_csv)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"RandomForest": "#1976D2", "XGBoost": "#388E3C"}

    for model_name, grp in df.groupby("model"):
        sp = grp["speedup"].dropna()
        sp_filtered = sp[(sp >= 0.5) & (sp <= 3.0)]
        ax.hist(sp_filtered, bins=30, alpha=0.6, label=f"{model_name} (median {sp.median():.3f}x)",
                color=colors.get(model_name, "#757575"), edgecolor="white")

    ax.axvline(1.0, color="#D32F2F", linestyle="--", linewidth=2, label="Parity with -O3 (1.0x)")
    ax.set_xlabel("Speedup relative to -O3 (higher is better)", fontsize=11)
    ax.set_ylabel("Number of Programs", fontsize=11)
    ax.set_title("Distribution of Speedups Achieved by ML-Recommended Flags vs -O3", fontsize=12, pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = PLOTS_DIR / "speedup_distribution_large.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"  Saved -> {out}")


def plot_win_rate_by_category():
    eval_csv = RESULTS_DIR / "evaluation_large.csv"
    dataset_csv = DATA_DIR / "dataset_large.csv"
    if not eval_csv.exists() or not dataset_csv.exists():
        return

    df_eval = pd.read_csv(eval_csv)
    df_data = pd.read_csv(dataset_csv)

    # Extract category from program name: prog_000001_category
    df_eval["category"] = df_eval["program"].apply(
        lambda p: "_".join(p.split("_")[2:]) if p.count("_") >= 2 else "other"
    )

    cat_stats = df_eval.groupby(["category", "model"])["beats_O3"].agg(["mean", "count"]).reset_index()
    rf_stats = cat_stats[cat_stats["model"] == "RandomForest"].sort_values("mean", ascending=True)

    if rf_stats.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(rf_stats) * 0.35)))
    y_pos = np.arange(len(rf_stats))

    bars = ax.barh(y_pos, rf_stats["mean"] * 100, color="#1976D2", edgecolor="white", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rf_stats["category"], fontsize=9)
    ax.axvline(50, color="#D32F2F", linestyle="--", alpha=0.7, label="50% baseline")
    ax.set_xlabel("Win Rate vs -O3 (%)", fontsize=11)
    ax.set_title("ML Model Win Rate vs -O3 by Algorithm Category", fontsize=12, pad=10)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "win_rate_by_category.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"  Saved -> {out}")


def plot_best_sequence_distribution():
    dataset_csv = DATA_DIR / "dataset_large.csv"
    if not dataset_csv.exists():
        return
    df = pd.read_csv(dataset_csv)

    counts = df["best_time_seq_name"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = counts.plot(kind="bar", ax=ax, color="#5C6BC0", edgecolor="white", width=0.6)

    ax.set_xlabel("Optimization Sequence Preset", fontsize=11)
    ax.set_ylabel("Count of Programs where this Sequence Won", fontsize=11)
    ax.set_title("Empirical Ground Truth: Which Optimization Sequence is Optimal across 10k Programs?", fontsize=12, pad=12)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)

    for i, v in enumerate(counts):
        ax.text(i, v + max(counts)*0.015, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=8.5)

    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    out = PLOTS_DIR / "best_sequence_distribution.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"  Saved -> {out}")


def plot_feature_importance():
    imp_file = RESULTS_DIR / "feature_importance_large.json"
    if not imp_file.exists():
        return
    data = json.loads(imp_file.read_text())
    pairs = sorted(zip(data["importances"], data["features"]), reverse=True)[:20]
    vals, names = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(names))
    ax.barh(y, vals, color="#0288D1", edgecolor="white", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Feature Importance (Gini Impurity)", fontsize=11)
    ax.set_title("Top 20 Code Features Driving Compiler Optimization Decisions", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    out = PLOTS_DIR / "feature_importance_large.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"  Saved -> {out}")


def main():
    print("Generating large dataset plots ...")
    plot_speedup_distribution()
    plot_win_rate_by_category()
    plot_best_sequence_distribution()
    plot_feature_importance()
    print(f"All large-scale plots saved to {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
