"""
ml/train.py
-----------
Train ML models (Random Forest + XGBoost) to predict the best
optimization sequence (best_time_seq) from program features.

Pipeline
--------
1. Load data/dataset.csv
2. Split: 70% train / 15% val / 15% test  (program-level, no leakage)
3. Train RandomForestClassifier (baseline)
4. Train XGBoostClassifier with Optuna hyperparameter search
5. Evaluate both models vs -O3 baseline on the held-out test set
6. Save trained models to models/rf_model.pkl and models/xgb_model.pkl
7. Save evaluation results to results/evaluation.csv
8. Print classification report + feature importances
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# -----------------------------------------------------
# Config
# -----------------------------------------------------

DATASET_CSV = Path("data/dataset.csv")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")

# Feature columns (everything except metadata / label columns)
META_COLS  = ["program"]
LABEL_COLS = [
    "best_time_seq", "best_size_seq",
    "best_time_seq_name", "best_size_seq_name",
    "exec_time_O2", "exec_time_O3", "best_exec_time",
    "best_size_bytes", "speedup_vs_O3", "size_ratio_vs_O3",
]

O3_SEQ_ID = 3   # must match scripts/measure.py

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------

def load_data(path: Path = DATASET_CSV):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS + LABEL_COLS]
    X = df[feat_cols].values.astype(np.float32)
    y = df["best_time_seq"].values.astype(int)
    return df, feat_cols, X, y


def train_test_split_programs(df, X, y, train_frac=0.70, val_frac=0.15, seed=42):
    """Program-level split to prevent data leakage."""
    rng     = np.random.default_rng(seed)
    programs = df["program"].unique()
    rng.shuffle(programs)
    n       = len(programs)
    n_train = max(1, int(n * train_frac))
    n_val   = max(1, int(n * val_frac))

    train_progs = set(programs[:n_train])
    val_progs   = set(programs[n_train: n_train + n_val])
    test_progs  = set(programs[n_train + n_val:])

    # Fallback: if splits leave test empty, put everything in train/test
    if not test_progs:
        test_progs = val_progs
        val_progs  = set()

    idx_train = df["program"].isin(train_progs).values
    idx_val   = df["program"].isin(val_progs).values
    idx_test  = df["program"].isin(test_progs).values

    return (
        X[idx_train], y[idx_train],
        X[idx_val],   y[idx_val],
        X[idx_test],  y[idx_test],
        df[idx_test],
    )


# -----------------------------------------------------
# Random Forest
# -----------------------------------------------------

def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n-- Random Forest -----------------------------------------")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, rf.predict(X_val)) if len(y_val) > 0 else float('nan')
    print(f"  Val accuracy: {val_acc:.3f}")
    return rf


# -----------------------------------------------------
# XGBoost (with simple grid search when optuna unavailable)
# -----------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val, n_classes: int):
    print("\n-- XGBoost -----------------------------------------------")

    # Try Optuna first; fallback to a quick manual grid
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _use_optuna = True
    except ImportError:
        _use_optuna = False

    le = LabelEncoder()
    # Fit on the union of train + val labels to avoid unseen-label errors
    # when the dataset is small and not all classes appear in training split
    all_labels = np.concatenate([y_train, y_val]) if len(y_val) > 0 else y_train
    le.fit(all_labels)
    y_tr = le.transform(y_train)
    y_vl = le.transform(y_val) if len(y_val) > 0 else np.array([])

    num_class = len(le.classes_)

    def _make_model(params):
        return xgb.XGBClassifier(
            num_class=num_class if num_class > 2 else None,
            objective="multi:softmax" if num_class > 2 else "binary:logistic",
            eval_metric="merror" if num_class > 2 else "error",
            random_state=42,
            verbosity=0,
            **params,
        )

    if _use_optuna and len(y_val) > 0:
        def objective(trial):
            params = {
                "n_estimators":   trial.suggest_int("n_estimators", 100, 500),
                "max_depth":      trial.suggest_int("max_depth", 3, 8),
                "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            m = _make_model(params)
            m.fit(X_train, y_tr, eval_set=[(X_val, y_vl)], verbose=False)
            return accuracy_score(y_vl, m.predict(X_val))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        best_params = study.best_params
        print(f"  Optuna best params: {best_params}")
        xgb_model = _make_model(best_params)
    else:
        # Simple default
        best_params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
                       "subsample": 0.8, "colsample_bytree": 0.8}
        xgb_model = _make_model(best_params)

    xgb_model.fit(X_train, y_tr, verbose=False)
    val_acc = accuracy_score(y_vl, xgb_model.predict(X_val)) if len(y_vl) > 0 else float('nan')
    print(f"  Val accuracy: {val_acc:.3f}")

    # Wrap with label encoder so predict() returns original seq_ids
    return xgb_model, le


# -----------------------------------------------------
# Evaluation
# -----------------------------------------------------

def evaluate(models_dict, X_test, y_test, df_test, feat_cols):
    """
    Compare each model's recommendations against the -O3 baseline.
    Metrics:
      - accuracy      : how often the model picks the true best sequence
      - win_rate_time : fraction of programs where model-seq time ? O3 time
      - avg_speedup   : geometric mean of (O3_time / model_seq_time)
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    results_rows = []

    meas = pd.read_csv("data/measurements.csv")
    meas = meas[meas["exec_time_s"] > 0]

    def get_time(prog, seq_id):
        row = meas[(meas["program"] == prog) & (meas["seq_id"] == seq_id)]
        if row.empty:
            return None
        return float(row["exec_time_s"].iloc[0])

    for model_name, predict_fn in models_dict.items():
        preds = predict_fn(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"\n-- {model_name} -----------------------------------------")
        print(f"  Test accuracy (exact seq match): {acc:.3f}")

        wins = 0; total = 0; speedups = []
        per_prog_rows = []

        for i, (_, row) in enumerate(df_test.iterrows()):
            prog = row["program"]
            pred_seq = int(preds[i])
            t_model  = get_time(prog, pred_seq)
            t_o3     = get_time(prog, O3_SEQ_ID)
            if t_model is None or t_o3 is None:
                continue
            total  += 1
            win     = t_model <= t_o3
            speedup = t_o3 / t_model if t_model > 0 else 1.0
            wins   += int(win)
            speedups.append(speedup)
            per_prog_rows.append({
                "model":       model_name,
                "program":     prog,
                "true_seq":    int(row["best_time_seq"]),
                "pred_seq":    pred_seq,
                "t_model":     t_model,
                "t_o3":        t_o3,
                "speedup":     speedup,
                "beats_O3":    win,
            })

        win_rate = wins / total if total > 0 else 0.0
        avg_sp   = float(np.exp(np.mean(np.log(speedups)))) if speedups else 1.0
        print(f"  Win rate vs -O3:  {win_rate:.2%}  ({wins}/{total})")
        print(f"  Geomean speedup:  {avg_sp:.4f}x")
        results_rows.extend(per_prog_rows)

    pd.DataFrame(results_rows).to_csv(RESULTS_DIR / "evaluation.csv", index=False)
    print(f"\nDetailed results -> {RESULTS_DIR / 'evaluation.csv'}")
    return results_rows


# -----------------------------------------------------
# Feature importance
# -----------------------------------------------------

def print_feature_importance(rf_model, feat_cols, top_n=10):
    imp = sorted(zip(feat_cols, rf_model.feature_importances_),
                 key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\n-- Top {top_n} Feature Importances (Random Forest) ---------")
    for name, score in imp:
        bar = "#" * int(score * 80)
        print(f"  {name:<30} {score:.4f}  {bar}")
    return imp


# -----------------------------------------------------
# Leave-One-Out Cross-Validation evaluation
# (correct strategy for n=10 programs)
# -----------------------------------------------------

def loocv_evaluate(df, X, y, feat_cols):
    """
    Leave-One-Out CV: train on n-1 programs, predict on 1, repeat.
    Returns per-program prediction results.
    """
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    rf_preds, xgb_preds, true_labels = [], [], []
    programs = []

    all_classes = np.unique(y)
    n_classes   = len(all_classes)

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                    min_samples_leaf=1, random_state=42)
        rf.fit(X_tr, y_tr)
        rf_pred = int(rf.predict(X_te)[0])

        # XGBoost: re-encode labels within this fold so they are contiguous 0..k-1
        # XGBoost strictly requires y in [0, num_class) with no gaps
        fold_classes   = np.unique(y_tr)       # classes actually in this fold's training
        fold_le        = LabelEncoder()
        fold_le.fit(fold_classes)
        y_tr_enc       = fold_le.transform(y_tr)
        fold_n_classes = len(fold_classes)

        xgb_m = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softmax" if fold_n_classes > 2 else "binary:logistic",
            num_class=fold_n_classes if fold_n_classes > 2 else None,
            eval_metric="merror" if fold_n_classes > 2 else "error",
            random_state=42, verbosity=0,
        )
        xgb_m.fit(X_tr, y_tr_enc, verbose=False)

        # Predict: if test class not seen in training, XGBoost can still output
        # a valid class index — we map it back to a seq_id via fold_le
        xgb_pred_enc = int(xgb_m.predict(X_te)[0])
        # Clamp to valid range in case of out-of-bound prediction
        xgb_pred_enc = min(xgb_pred_enc, fold_n_classes - 1)
        xgb_pred     = int(fold_le.inverse_transform([xgb_pred_enc])[0])

        rf_preds.append(rf_pred)
        xgb_preds.append(xgb_pred)
        true_labels.append(int(y_te[0]))
        programs.append(df.iloc[test_idx[0]]["program"])


    return programs, true_labels, rf_preds, xgb_preds


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading dataset ...")
    df, feat_cols, X, y = load_data()
    print(f"  Programs : {len(df)}")
    print(f"  Features : {len(feat_cols)}")
    print(f"  Classes  : {len(np.unique(y))}  {sorted(np.unique(y).tolist())}")

    if len(df) < 3:
        print("\n[!]  Need at least 3 programs. Run the full pipeline first.")
        return

    # -------------------------------------------------------------------
    # Leave-One-Out CV evaluation (correct for small n)
    # -------------------------------------------------------------------
    print(f"\nRunning Leave-One-Out CV ({len(df)} folds) ...")
    programs, true_labels, rf_preds, xgb_preds = loocv_evaluate(df, X, y, feat_cols)

    print("\n  Program            True   RF    XGB")
    print("  " + "-"*45)
    for prog, true, rf_p, xgb_p in zip(programs, true_labels, rf_preds, xgb_preds):
        rf_ok  = "[OK]" if rf_p  == true else "    "
        xgb_ok = "[OK]" if xgb_p == true else "    "
        print(f"  {prog:<20} {true:>4}   {rf_p:>3}{rf_ok}  {xgb_p:>3}{xgb_ok}")

    rf_acc  = sum(r == t for r, t in zip(rf_preds,  true_labels)) / len(true_labels)
    xgb_acc = sum(r == t for r, t in zip(xgb_preds, true_labels)) / len(true_labels)
    print(f"\n  LOO-CV Accuracy  RF: {rf_acc:.2%}   XGBoost: {xgb_acc:.2%}")

    # -------------------------------------------------------------------
    # Evaluate win-rate vs -O3
    # -------------------------------------------------------------------
    meas = pd.read_csv("data/measurements.csv")
    meas = meas[meas["exec_time_s"] > 0]

    def get_time(prog, seq_id):
        row = meas[(meas["program"] == prog) & (meas["seq_id"] == seq_id)]
        return float(row["exec_time_s"].iloc[0]) if not row.empty else None

    rows_eval = []
    for model_name, preds in [("RandomForest", rf_preds), ("XGBoost", xgb_preds)]:
        wins = 0; total = 0; speedups = []
        for prog, pred_seq in zip(programs, preds):
            t_model = get_time(prog, pred_seq)
            t_o3    = get_time(prog, O3_SEQ_ID)
            if t_model is None or t_o3 is None:
                continue
            total += 1
            win    = t_model <= t_o3
            sp     = t_o3 / t_model if t_model > 0 else 1.0
            wins  += int(win)
            speedups.append(sp)
            rows_eval.append({
                "model": model_name, "program": prog,
                "pred_seq": pred_seq,
                "t_model": t_model, "t_o3": t_o3,
                "speedup": sp, "beats_O3": win,
            })
        win_rate = wins / total if total > 0 else 0.0
        avg_sp   = float(np.exp(np.mean(np.log(speedups)))) if speedups else 1.0
        print(f"\n  [{model_name}]  Win rate vs -O3: {win_rate:.0%} ({wins}/{total})  Geomean speedup: {avg_sp:.4f}x")

    pd.DataFrame(rows_eval).to_csv(RESULTS_DIR / "evaluation.csv", index=False)
    print(f"\n  Detailed results -> {RESULTS_DIR / 'evaluation.csv'}")

    # -------------------------------------------------------------------
    # Train FINAL models on ALL data
    # -------------------------------------------------------------------
    print("\nTraining final models on full dataset ...")

    # -- Random Forest --
    rf_final = RandomForestClassifier(n_estimators=300, max_depth=None,
                                      min_samples_leaf=1, n_jobs=-1, random_state=42)
    rf_final.fit(X, y)
    with open(MODELS_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump((rf_final, feat_cols), f)
    print(f"  RF saved   -> {MODELS_DIR / 'rf_model.pkl'}")

    # -- XGBoost --
    all_classes = np.unique(y)
    n_classes   = len(all_classes)
    le_final    = LabelEncoder()
    le_final.fit(all_classes)
    y_enc = le_final.transform(y)

    xgb_final = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softmax", num_class=n_classes,
        eval_metric="merror", random_state=42, verbosity=0,
    )
    xgb_final.fit(X, y_enc, verbose=False)
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump((xgb_final, le_final, feat_cols), f)
    print(f"  XGB saved  -> {MODELS_DIR / 'xgb_model.pkl'}")

    # -- Feature importance --
    print_feature_importance(rf_final, feat_cols)

    imp_data = {
        "features": feat_cols,
        "importances": rf_final.feature_importances_.tolist(),
    }
    (RESULTS_DIR / "feature_importance.json").write_text(json.dumps(imp_data, indent=2))
    print(f"\nFeature importances saved -> {RESULTS_DIR / 'feature_importance.json'}")

    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()
