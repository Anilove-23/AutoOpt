"""
autoopt.models.ensemble
=======================
Ensemble Model Combining Random Forest and Gradient Boosted Trees.
Provides probability calibration, confidence estimation, and feature importance.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


@dataclass
class PredictionResult:
    sequence_id: int
    sequence_name: str
    recommended_flags: str
    confidence: float
    probabilities: Dict[str, float]
    model_name: str


class AutoOptEnsemble:
    """Ensemble classifier for compiler optimization prediction."""

    def __init__(
        self,
        rf_estimators: int = 500,
        xgb_estimators: int = 300,
        random_state: int = 42,
    ):
        self.rf = RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
        self.xgb = xgb.XGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            eval_metric="merror",
            random_state=random_state,
            verbosity=0,
        )
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_columns: List[str]) -> AutoOptEnsemble:
        """Trains the ensemble on program feature matrix X and sequence targets y."""
        self.feature_columns = list(feature_columns)
        all_classes = np.unique(y)
        self.label_encoder.fit(all_classes)
        y_enc = self.label_encoder.transform(y)

        # Train Random Forest
        self.rf.fit(X, y)

        # Train XGBoost
        self.xgb.set_params(num_class=len(all_classes))
        self.xgb.fit(X, y_enc)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ensemble class probabilities and class labels."""
        if not self.is_fitted:
            raise RuntimeError("AutoOptEnsemble is not fitted yet.")

        has_le = self.label_encoder is not None and hasattr(self.label_encoder, "classes_")
        target_classes = self.label_encoder.classes_ if has_le else getattr(self.rf, "classes_", None)

        rf_probs = self.rf.predict_proba(X) if hasattr(self.rf, "predict_proba") else None
        xgb_probs = None
        if self.xgb is not None and hasattr(self.xgb, "predict_proba"):
            try:
                xgb_probs = self.xgb.predict_proba(X)
            except Exception:
                xgb_probs = None

        if rf_probs is not None and xgb_probs is not None and rf_probs.shape == xgb_probs.shape:
            ensemble_probs = (rf_probs + xgb_probs) / 2.0
        elif xgb_probs is not None:
            ensemble_probs = xgb_probs
        elif rf_probs is not None:
            ensemble_probs = rf_probs
        else:
            raise RuntimeError("Neither RF nor XGB has valid predict_proba.")

        if target_classes is None:
            target_classes = np.arange(ensemble_probs.shape[1])

        return ensemble_probs, target_classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts winning sequence IDs."""
        probs, classes = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return classes[pred_indices]

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns sorted feature importances from Random Forest."""
        if not self.is_fitted:
            return {}
        pairs = sorted(zip(self.feature_columns, self.rf.feature_importances_), key=lambda x: -x[1])
        return dict(pairs)

    def save(self, file_path: Path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({
                "rf": self.rf,
                "xgb": self.xgb,
                "label_encoder": self.label_encoder,
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted,
            }, f)

    @classmethod
    def load(cls, file_path: Path) -> AutoOptEnsemble:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        ensemble = cls()
        if isinstance(data, dict) and "rf" in data:
            ensemble.rf = data["rf"]
            ensemble.xgb = data["xgb"]
            ensemble.label_encoder = data["label_encoder"]
            ensemble.feature_columns = data["feature_columns"]
            ensemble.is_fitted = data["is_fitted"]
        elif isinstance(data, tuple) and len(data) == 2:
            # Compatible with previous rf_large.pkl format (model, feat_cols)
            ensemble.rf = data[0]
            ensemble.feature_columns = data[1]
            ensemble.is_fitted = True
        return ensemble
