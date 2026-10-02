"""
ml/train_large.py
=================
Train ML models on the large (10,000-program) dataset.

With enough data we can do a proper train/val/test split
(unlike the 10-program case where we needed LOO-CV).

Strategy:
  - 70% train / 15% val / 15% test  (program-level split)
  - Models: Random Forest, XGBoost with Optuna tuning
  - Evaluation: accuracy + win-rate vs -O3 + geomean speedup
  - Feature importance via RF + SHAP (on a sample)
  - Saves models to models/rf_large.pkl, models/xgb_large.pkl
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

DATASET_CSV = Path("data/dataset_large.csv")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
O3_ID       = 3

META_COLS  = ["program"]
LABEL_COLS = ["best_time_seq","best_size_seq","best_time_seq_name","best_size_seq_name",
              "exec_time_O3","best_exec_time","best_size_bytes","speedup_vs_O3","size_ratio_vs_O3"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df        = pd.read_csv(DATASET_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS + LABEL_COLS]
    X = df[feat_cols].values.astype(np.float32)
    y = df["best_time_seq"].values.astype(int)
    return df, feat_cols, X, y


def split(df, X, y, train_f=0.70, val_f=0.15, seed=42):
    rng      = np.random.default_rng(seed)
    progs    = df["program"].unique()
    rng.shuffle(progs)
    n        = len(progs)
    n_tr     = int(n * train_f)
    n_vl     = int(n * val_f)
    tr_p     = set(progs[:n_tr])
    vl_p     = set(progs[n_tr:n_tr+n_vl])
    te_p     = set(progs[n_tr+n_vl:])
    idx_tr   = df["program"].isin(tr_p).values
    idx_vl   = df["program"].isin(vl_p).values
    idx_te   = df["program"].isin(te_p).values
    return (X[idx_tr], y[idx_tr],
            X[idx_vl], y[idx_vl],
            X[idx_te], y[idx_te],
            df[idx_te])


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_rf(X_tr, y_tr, X_vl, y_vl):
    print("\n[Random Forest] training ...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                min_samples_leaf=1, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    val_acc = accuracy_score(y_vl, rf.predict(X_vl))
    print(f"  Val accuracy: {val_acc:.3f}")
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost with Optuna
# ─────────────────────────────────────────────────────────────────────────────

def train_xgb(X_tr, y_tr, X_vl, y_vl):
    print("\n[XGBoost] training ...")
    all_classes = np.unique(np.concatenate([y_tr, y_vl]))
    le = LabelEncoder(); le.fit(all_classes)
    y_tr_e = le.transform(y_tr)
    y_vl_e = le.transform(y_vl)
    n_cls  = len(all_classes)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
                "max_depth":        trial.suggest_int("max_depth", 4, 10),
                "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            m = xgb.XGBClassifier(
                **params,
                objective="multi:softmax", num_class=n_cls,
                eval_metric="merror", random_state=42, verbosity=0,
            )
            m.fit(X_tr, y_tr_e, eval_set=[(X_vl, y_vl_e)], verbose=False)
            return accuracy_score(y_vl_e, m.predict(X_vl))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=40, show_progress_bar=False)
        best_p = study.best_params
        print(f"  Optuna best: {best_p}  val_acc={study.best_value:.3f}")
    except ImportError:
        best_p = {"n_estimators":400,"max_depth":7,"learning_rate":0.1,
                  "subsample":0.8,"colsample_bytree":0.8,"min_child_weight":1}

    model = xgb.XGBClassifier(
        **best_p,
        objective="multi:softmax", num_class=n_cls,
        eval_metric="merror", random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr_e, verbose=False)
    val_acc = accuracy_score(y_vl_e, model.predict(X_vl))
    print(f"  Final val accuracy: {val_acc:.3f}")
    return model, le


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation vs -O3
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vs_o3(models_preds: dict, df_test: pd.DataFrame,
                   meas_csv: Path = Path("data/measurements_generated.csv")):
    meas = pd.read_csv(meas_csv)
    meas = meas[meas["exec_time_s"] > 0]

    def get_time(prog, seq_id):
        r = meas[(meas["program"]==prog)&(meas["seq_id"]==seq_id)]
        return float(r["exec_time_s"].iloc[0]) if not r.empty else None

    rows = []
    for model_name, preds in models_preds.items():
        wins=0; total=0; speedups=[]
        for i, (_, row) in enumerate(df_test.iterrows()):
            prog     = row["program"]
            pred_seq = int(preds[i])
            tm       = get_time(prog, pred_seq)
            to3      = get_time(prog, O3_ID)
            if tm is None or to3 is None: continue
            total += 1
            sp = to3/tm if tm>0 else 1.0
            wins += int(tm<=to3)
            speedups.append(sp)
            rows.append({"model":model_name,"program":prog,
                         "pred_seq":pred_seq,"t_model":tm,"t_o3":to3,
                         "speedup":sp,"beats_O3":tm<=to3})
        win_rate = wins/total if total>0 else 0
        avg_sp   = float(np.exp(np.mean(np.log(speedups)))) if speedups else 1.0
        print(f"  [{model_name}] Win rate: {win_rate:.1%} ({wins}/{total})  Geomean speedup: {avg_sp:.4f}x")

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR/"evaluation_large.csv", index=False)
    print(f"  Saved -> {RESULTS_DIR/'evaluation_large.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance(rf, feat_cols, X_sample=None, top_n=20):
    pairs = sorted(zip(feat_cols, rf.feature_importances_), key=lambda x:-x[1])[:top_n]
    print(f"\n  Top {top_n} Features (RF):")
    for name, imp in pairs:
        bar = "#"*int(imp*100)
        print(f"  {name:<35} {imp:.4f}  {bar}")

    # SHAP (optional, on a sample)
    if X_sample is not None:
        try:
            import shap
            sample = X_sample[:min(200, len(X_sample))]
            explainer = shap.TreeExplainer(rf)
            sv        = explainer.shap_values(sample)
            mean_abs  = np.abs(np.array(sv)).mean(axis=(0,2)) if sv.ndim==3 else np.abs(sv).mean(0)
            shap_imp  = dict(zip(feat_cols, [float(x) for x in mean_abs]))
            (RESULTS_DIR/"shap_importance.json").write_text(json.dumps(shap_imp, indent=2))
            print(f"  SHAP importances saved -> {RESULTS_DIR/'shap_importance.json'}")
        except Exception as e:
            print(f"  SHAP skipped: {e}")

    return dict(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading large dataset ...")
    df, feat_cols, X, y = load_data()
    unique_classes = np.unique(y)
    print(f"  Programs : {len(df)}")
    print(f"  Features : {len(feat_cols)}")
    print(f"  Classes  : {len(unique_classes)}  {sorted(unique_classes.tolist())}")

    X_tr, y_tr, X_vl, y_vl, X_te, y_te, df_te = split(df, X, y)
    print(f"  Split    : train={len(y_tr)}  val={len(y_vl)}  test={len(y_te)}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = train_rf(X_tr, y_tr, X_vl, y_vl)
    with open(MODELS_DIR/"rf_large.pkl","wb") as f:
        pickle.dump((rf, feat_cols), f)
    print(f"  Saved -> {MODELS_DIR/'rf_large.pkl'}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_model, le = train_xgb(X_tr, y_tr, X_vl, y_vl)
    with open(MODELS_DIR/"xgb_large.pkl","wb") as f:
        pickle.dump((xgb_model, le, feat_cols), f)
    print(f"  Saved -> {MODELS_DIR/'xgb_large.pkl'}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nTest set evaluation ({len(y_te)} programs):")
    rf_preds  = rf.predict(X_te)
    xgb_preds = le.inverse_transform(xgb_model.predict(X_te))

    rf_acc  = accuracy_score(y_te, rf_preds)
    xgb_acc = accuracy_score(y_te, xgb_preds)
    print(f"  RF  test accuracy: {rf_acc:.3f}")
    print(f"  XGB test accuracy: {xgb_acc:.3f}")

    evaluate_vs_o3(
        {"RandomForest": rf_preds, "XGBoost": xgb_preds},
        df_te
    )

    # ── Feature importance ────────────────────────────────────────────────────
    imp = feature_importance(rf, feat_cols, X_sample=X_tr)
    imp_data = {"features": feat_cols, "importances": rf.feature_importances_.tolist()}
    (RESULTS_DIR/"feature_importance_large.json").write_text(json.dumps(imp_data, indent=2))

    print("\n[DONE] Large model training complete.")


if __name__ == "__main__":
    main()
