"""
cli/recommend.py
----------------
CLI tool: given a .c source file, extract features and output
the recommended optimization flag-set predicted by the trained model.

Usage
-----
  python cli/recommend.py path/to/program.c
  python cli/recommend.py path/to/program.c --model xgboost
  python cli/recommend.py path/to/program.c --model rf --json

Output (default):
  Recommended flags:  -O3 -ffast-math -funroll-loops ...
  Predicted seq id:   19
  Seq name:           full_aggressive

Output (--json):
  {
    "program": "my_prog",
    "recommended_flags": "-O3 -ffast-math ...",
    "seq_id": 19,
    "seq_name": "full_aggressive",
    "model": "xgboost",
    "features": { ... }
  }
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# -- Ensure project root is on sys.path ----------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.extractor import extract_features
from scripts.measure import SEQUENCES   # sequence id -> flags mapping

MODELS_DIR = ROOT / "models"

SEQ_MAP: dict[int, dict] = {s["id"]: s for s in SEQUENCES}

# -----------------------------------------------------
# Model loaders
# -----------------------------------------------------

def load_rf(use_large: bool = False):
    pkl_large = MODELS_DIR / "rf_large.pkl"
    pkl_std   = MODELS_DIR / "rf_model.pkl"
    pkl = pkl_large if (use_large and pkl_large.exists()) or (not pkl_std.exists() and pkl_large.exists()) else pkl_std
    if not pkl.exists():
        raise FileNotFoundError(f"RF model not found at {pkl}. Run: python ml/train.py or ml/train_large.py")
    with open(pkl, "rb") as f:
        model, feat_cols = pickle.load(f)
    return model, feat_cols


def load_xgb(use_large: bool = False):
    pkl_large = MODELS_DIR / "xgb_large.pkl"
    pkl_std   = MODELS_DIR / "xgb_model.pkl"
    pkl = pkl_large if (use_large and pkl_large.exists()) or (not pkl_std.exists() and pkl_large.exists()) else pkl_std
    if not pkl.exists():
        raise FileNotFoundError(f"XGBoost model not found at {pkl}. Run: python ml/train.py or ml/train_large.py")
    with open(pkl, "rb") as f:
        model, le, feat_cols = pickle.load(f)
    return model, le, feat_cols


# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

def predict_rf(c_file: Path, model, feat_cols) -> int:
    feats = extract_features(c_file)
    X = [[feats.get(col, 0) for col in feat_cols]]
    return int(model.predict(X)[0])


def predict_xgb(c_file: Path, model, le, feat_cols) -> int:
    feats = extract_features(c_file)
    X = [[feats.get(col, 0) for col in feat_cols]]
    pred_enc = model.predict(X)[0]
    return int(le.inverse_transform([pred_enc])[0])


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ML-based compiler optimization recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/recommend.py benchmarks/custom/matrix_mult.c
  python cli/recommend.py benchmarks/custom/fibonacci.c --model rf
  python cli/recommend.py my_program.c --json
        """,
    )
    parser.add_argument("source", help="Path to C source file")
    parser.add_argument(
        "--model", choices=["rf", "xgboost"], default="xgboost",
        help="Which trained model to use (default: xgboost)"
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output result as JSON"
    )
    parser.add_argument(
        "--gcc", default="gcc",
        help="GCC executable to use for feature extraction (default: gcc)"
    )
    args = parser.parse_args()

    c_file = Path(args.source).resolve()
    if not c_file.exists():
        print(f"Error: file not found: {c_file}", file=sys.stderr)
        sys.exit(1)

    # Extract features
    feats = extract_features(c_file, gcc=args.gcc)

    # Predict
    if args.model == "rf":
        model, feat_cols = load_rf()
        seq_id = predict_rf(c_file, model, feat_cols)
    else:
        model, le, feat_cols = load_xgb()
        X = [[feats.get(col, 0) for col in feat_cols]]
        pred_enc = model.predict(X)[0]
        seq_id = int(le.inverse_transform([pred_enc])[0])

    seq = SEQ_MAP.get(seq_id, {"name": "unknown", "flags": "-O2"})

    if args.as_json:
        result = {
            "program":             c_file.stem,
            "recommended_flags":   seq["flags"],
            "seq_id":              seq_id,
            "seq_name":            seq["name"],
            "model":               args.model,
            "features":            {k: v for k, v in feats.items() if k != "program"},
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'-'*55}")
        print(f"  Program  : {c_file.name}")
        print(f"  Model    : {args.model}")
        print(f"  Seq ID   : {seq_id}")
        print(f"  Seq Name : {seq['name']}")
        print(f"\n  OK Recommended compilation flags:")
        print(f"    gcc {seq['flags']} {c_file.name} -o {c_file.stem}")
        print(f"{'-'*55}\n")


if __name__ == "__main__":
    main()
