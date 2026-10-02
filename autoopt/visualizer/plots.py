"""
autoopt.visualizer.plots
========================
High-impact publication visualization engine for AutoOpt.
Generates speedup histograms, feature importance charts, and category breakdowns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_speedup_histogram(
    speedups: List[float],
    output_path: Path,
    baseline_label: str = "-O3 Baseline",
):
    """Plots distribution of speedup factors with median and parity lines."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sp = np.array(speedups)
    sp_filtered = sp[(sp >= 0.5) & (sp <= 3.0)]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    n, bins, patches = ax.hist(
        sp_filtered,
        bins=35,
        color="#1E88E5",
        edgecolor="white",
        alpha=0.85,
    )

    # Highlight faster than baseline in green
    for b_left, p in zip(bins[:-1], patches):
        if b_left >= 1.0:
            p.set_facecolor("#2E7D32")
        else:
            p.set_facecolor("#C62828")

    median_val = float(np.median(sp))
    ax.axvline(1.0, color="#212121", linestyle="--", linewidth=2.0, label=f"Parity with {baseline_label} (1.0x)")
    ax.axvline(median_val, color="#FF8F00", linestyle="-", linewidth=2.2, label=f"AutoOpt Median ({median_val:.3f}x)")

    ax.set_xlabel("Speedup Factor Relative to -O3 (Higher is Better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Programs", fontsize=11, fontweight="bold")
    ax.set_title("AutoOpt Speedup Distribution on Independent Benchmark Set", fontsize=12, pad=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance_bar(
    importances: Dict[str, float],
    output_path: Path,
    top_n: int = 15,
):
    """Plots horizontal bar chart of the most influential code features."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in items][::-1]
    scores = [x[1] for x in items][::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    bars = ax.barh(range(len(names)), scores, color="#00838F", edgecolor="white", height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("Relative Feature Importance (Gini Impurity)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Static Analysis Features Driving Pass Selection", fontsize=12, pad=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
