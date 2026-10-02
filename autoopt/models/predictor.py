"""
autoopt.models.predictor
========================
High-level prediction API for end-user programs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from autoopt.analyzer import analyze_program, ProgramProfile
from autoopt.engine.sequences import get_sequence_by_id, SEQUENCE_MAP
from autoopt.models.ensemble import AutoOptEnsemble, PredictionResult


class AutoOptPredictor:
    """Predicts optimal compiler flags for arbitrary C/C++ source code."""

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            # Default model search path
            root = Path(__file__).resolve().parent.parent.parent
            cand1 = root / "models" / "autoopt_ensemble.pkl"
            cand2 = root / "models" / "rf_large.pkl"
            cand3 = root / "models" / "rf_model.pkl"
            model_path = cand1 if cand1.exists() else (cand2 if cand2.exists() else cand3)

        self.model_path = Path(model_path)
        self.ensemble: Optional[AutoOptEnsemble] = None
        if self.model_path.exists():
            self.ensemble = AutoOptEnsemble.load(self.model_path)

    def predict(self, source_file: Path) -> PredictionResult:
        """Analyzes source_file and returns the predicted optimal optimization sequence."""
        profile = analyze_program(source_file)
        return self.predict_from_profile(profile)

    def predict_from_profile(self, profile: ProgramProfile) -> PredictionResult:
        """Predicts optimal optimization flags from an existing ProgramProfile."""
        if self.ensemble is None:
            # Heuristic fallback if model not loaded
            return self._heuristic_fallback(profile)

        # Build feature vector matching model columns
        vec = profile.to_feature_vector(self.ensemble.feature_columns)
        X = np.array([vec], dtype=np.float32)

        try:
            pred_id = int(self.ensemble.predict(X)[0])
            confidence = 0.85
            probs_dict = {}
            if hasattr(self.ensemble, "predict_proba"):
                try:
                    probs, classes = self.ensemble.predict_proba(X)
                    confidence = float(np.max(probs[0]))
                    for cls_id, p in zip(classes, probs[0]):
                        seq_name = get_sequence_by_id(int(cls_id)).name
                        probs_dict[seq_name] = round(float(p), 4)
                except Exception:
                    pass
        except Exception:
            pred_id = 3
            confidence = 0.50
            probs_dict = {}

        seq = get_sequence_by_id(pred_id)
        return PredictionResult(
            sequence_id=seq.id,
            sequence_name=seq.name,
            recommended_flags=seq.gcc_flags,
            confidence=confidence,
            probabilities=probs_dict,
            model_name="AutoOpt-Ensemble",
        )

    def _heuristic_fallback(self, profile: ProgramProfile) -> PredictionResult:
        """Heuristic rule-based fallback when model weights are not loaded."""
        # Floating point heavy -> fast-math
        if profile.inst.float_ratio > 0.15:
            seq = SEQUENCE_MAP[13]  # O2_fast_math
        # High loop count & high arithmetic -> vectorization + unroll
        elif profile.loop.total_loops >= 2 and profile.inst.arithmetic_mix_ratio > 0.3:
            seq = SEQUENCE_MAP[5]   # O2_unroll_vec
        # Pointer chasing / high branch intensity -> Os
        elif profile.cfg.branch_density > 0.10 or profile.memory.pointer_dereferences > 10:
            seq = SEQUENCE_MAP[4]   # Os_size
        else:
            seq = SEQUENCE_MAP[3]   # O3_aggressive

        return PredictionResult(
            sequence_id=seq.id,
            sequence_name=seq.name,
            recommended_flags=seq.gcc_flags,
            confidence=0.70,
            probabilities={},
            model_name="AutoOpt-HeuristicEngine",
        )
