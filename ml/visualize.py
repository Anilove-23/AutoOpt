"""
ml/visualize.py
───────────────
Generate all result plots:
  1. Bar chart: execution time per program — model vs -O2 vs -O3
  2. Speedup distribution histogram
  3. Feature importance bar chart (RF + XGBoost)
  4. Confusion matrix (predicted vs actual best sequence)
  5. Binary size comparison bar chart

All plots saved to results/plots/
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RESULTS_DIR  = Path("results")
PLOTS_DIR    = RESULTS_DIR / "plots"
MODELS_DIR   = Path("models")
DATA_DIR     = Path("data")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# O2 / O3 seq ids must match measure.py
O2_ID = 2
O3_ID = 3


# ─────────────────────────────────────────────────────
# 1. Execution-time bar chart (model vs O2 vs O3)
# ─────────────────────────────────────────────────────

def plot_exec_time_comparison(eval_csv: Path = RESULTS_DIR / "evaluation.csv",
                               meas_csv: Path = DATA_DIR / "measurements.csv"):
    if not eval_csv.exists():
        print("  evaluation.csv not found — skipping exec-time plot")
        return

    df_eval = pd.read_csv(eval_csv)
    df_meas = pd.read_csv(meas_csv)
    df_meas = df_meas[df_meas["exec_time_s"] > 0]

    models = df_eval["model"].unique()
    programs = df_eval["program"].unique()

    for model_name in models:
        sub = df_eval[df_eval["model"] == model_name]
        progs   = sub["program"].tolist()
        t_model = sub["t_model"].tolist()
        t_o3    = sub["t_o3"].tolist()

        # Get O2 times
        t_o2 = []
        for p in progs:
            row = df_meas[(df_meas["program"] == p) & (df_meas["seq_id"] == O2_ID)]
            t_o2.append(float(row["exec_time_s"].iloc[0]) if not row.empty else float("nan"))

        x    = np.arange(len(progs))
        w    = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(progs) * 1.5), 5))
        ax.bar(x - w,     t_model, w, label=f"{model_name} recommended", color="#2196F3")
        ax.bar(x,         t_o2,    w, label="-O2 baseline",               color="#FF9800")
        ax.bar(x + w,     t_o3,    w, label="-O3 baseline",               color="#F44336")

        ax.set_xticks(x)
        ax.set_xticklabels(progs, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Execution time (s)")
        ax.set_title(f"Execution Time: {model_name} vs -O2 / -O3")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = PLOTS_DIR / f"exec_time_{model_name.lower()}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────
# 2. Speedup histogram
# ─────────────────────────────────────────────────────

def plot_speedup_histogram(eval_csv: Path = RESULTS_DIR / "evaluation.csv"):
    if not eval_csv.exists():
        return
    df = pd.read_csv(eval_csv)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ["#2196F3", "#4CAF50", "#FF9800"]
    for idx, (model, grp) in enumerate(df.groupby("model")):
        sp = grp["speedup"].dropna()
        ax.hist(sp, bins=15, alpha=0.7,
                label=model, color=colors[idx % len(colors)],
                edgecolor="white")

    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="-O3 parity (1.0×)")
    ax.set_xlabel("Speedup vs -O3")
    ax.set_ylabel("Count")
    ax.set_title("Speedup Distribution of Recommended Sequences vs -O3")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "speedup_histogram.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────
# 3. Feature importance
# ─────────────────────────────────────────────────────

def plot_feature_importance(imp_json: Path = RESULTS_DIR / "feature_importance.json",
                             top_n: int = 15):
    if not imp_json.exists():
        print("  feature_importance.json not found — skipping")
        return
    data = json.loads(imp_json.read_text())
    features = data["features"]
    imps     = data["importances"]
    pairs    = sorted(zip(imps, features), reverse=True)[:top_n]
    vals, names = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(names))
    ax.barh(y, vals, color="#2196F3", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance score")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "feature_importance_rf.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # XGBoost importance
    try:
        rf_pkl  = MODELS_DIR / "xgb_model.pkl"
        if rf_pkl.exists():
            with open(rf_pkl, "rb") as f:
                xgb_model, le, feat_cols = pickle.load(f)
            xgb_imp = xgb_model.feature_importances_
            pairs2  = sorted(zip(xgb_imp, feat_cols), reverse=True)[:top_n]
            vals2, names2 = zip(*pairs2)
            fig, ax = plt.subplots(figsize=(9, 5))
            y2 = np.arange(len(names2))
            ax.barh(y2, vals2, color="#4CAF50", edgecolor="white")
            ax.set_yticks(y2)
            ax.set_yticklabels(names2, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Importance score")
            ax.set_title(f"Top {top_n} Feature Importances (XGBoost)")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            out2 = PLOTS_DIR / "feature_importance_xgb.png"
            plt.savefig(out2, dpi=150)
            plt.close()
            print(f"  Saved: {out2}")
    except Exception as e:
        print(f"  XGBoost importance skipped: {e}")


# ─────────────────────────────────────────────────────
# 4. Win-rate summary bar chart
# ─────────────────────────────────────────────────────

def plot_win_rate_summary(eval_csv: Path = RESULTS_DIR / "evaluation.csv"):
    if not eval_csv.exists():
        return
    df = pd.read_csv(eval_csv)
    summary = df.groupby("model")["beats_O3"].agg(["mean","sum","count"]).reset_index()
    summary.columns = ["model","win_rate","wins","total"]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors  = ["#2196F3", "#4CAF50"]
    bars    = ax.bar(summary["model"], summary["win_rate"] * 100,
                     color=colors[:len(summary)], edgecolor="white", width=0.4)

    for bar, (_, r) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{r['wins']}/{r['total']}\n({r['win_rate']:.0%})",
                ha="center", va="bottom", fontsize=10)

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% parity")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Win rate vs -O3 (%)")
    ax.set_title("Model Win Rate: % of Programs Beating -O3")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "win_rate_summary.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────
# 5. Binary size comparison
# ─────────────────────────────────────────────────────

def plot_binary_size(dataset_csv: Path = DATA_DIR / "dataset.csv",
                     meas_csv:    Path = DATA_DIR / "measurements.csv"):
    if not dataset_csv.exists():
        return
    df_ds   = pd.read_csv(dataset_csv)
    df_meas = pd.read_csv(meas_csv)

    programs   = df_ds["program"].tolist()
    best_sizes = df_ds["best_size_bytes"].tolist()
    o3_sizes   = []
    for p in programs:
        row = df_meas[(df_meas["program"] == p) & (df_meas["seq_id"] == O3_ID)]
        o3_sizes.append(int(row["binary_size_bytes"].iloc[0]) if not row.empty else 0)

    x = np.arange(len(programs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(programs) * 1.4), 4))
    ax.bar(x - w/2, [s/1024 for s in best_sizes], w, label="Best sequence", color="#2196F3")
    ax.bar(x + w/2, [s/1024 for s in o3_sizes],   w, label="-O3 baseline",  color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(programs, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Binary size (KB)")
    ax.set_title("Binary Size: Best Sequence vs -O3")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "binary_size_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────
# 6. Optimization sequence distribution
# ─────────────────────────────────────────────────────

def plot_seq_distribution(dataset_csv: Path = DATA_DIR / "dataset.csv"):
    if not dataset_csv.exists():
        return
    df = pd.read_csv(dataset_csv)
    counts = df["best_time_seq_name"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(kind="bar", ax=ax, color="#9C27B0", edgecolor="white")
    ax.set_xlabel("Optimization sequence")
    ax.set_ylabel("# programs where this is best")
    ax.set_title("Which optimization sequence wins per program?")
    ax.tick_params(axis='x', rotation=40)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "seq_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    print("Generating plots …")
    plot_exec_time_comparison()
    plot_speedup_histogram()
    plot_feature_importance()
    plot_win_rate_summary()
    plot_binary_size()
    plot_seq_distribution()
    print(f"\nAll plots saved in {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
