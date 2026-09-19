"""Learned failure-risk model with non-saturating validation calibration.

Training uses TRAIN only. Validation is used only to fit a monotone Platt-style
probability calibrator; test rows are never seen during fitting/calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

FEATURES = [
    "robust_PAT", "isolation_forest", "temporal", "multivariate",
    "slope_normalized", "acceleration_normalized", "ewma_normalized",
    "cusum_normalized", "change_point_strength", "joint_exceedance_count",
    "absolute_limits", "missing_fraction",
]

@dataclass
class RiskArtifact:
    imputer: object
    model: object
    calibrator: object | None
    feature_columns: list[str]
    seed: int
    training_positive_rate: float
    calibration: str

class FailureRiskModel:
    def __init__(self, seed: int = 20260831):
        self.seed = seed
        self.artifact: RiskArtifact | None = None

    def _frame(self, d: pd.DataFrame) -> pd.DataFrame:
        x = d.copy()
        aliases = {
            "robust_PAT": "robust_PAT",
            "isolation_forest": "isolation_forest",
            "temporal": "temporal",
            "multivariate": "multivariate",
            "slope_normalized": "max_slope_normalized",
            "acceleration_normalized": "max_acceleration_normalized",
            "ewma_normalized": "ewma_normalized",
            "cusum_normalized": "cusum_normalized",
            "change_point_strength": "max_change_point_score",
            "joint_exceedance_count": "joint_exceedance_count",
            "absolute_limits": "absolute_violation",
            "missing_fraction": "current_missing_fraction",
        }
        out = pd.DataFrame(index=x.index)
        for target, source in aliases.items():
            out[target] = pd.to_numeric(x.get(source, pd.Series(np.nan, index=x.index)), errors="coerce")
        return out[FEATURES].replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _labels(d: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(d["future_defective"], errors="coerce").fillna(0).astype(int).to_numpy()

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None):
        if "future_defective" not in train.columns:
            raise ValueError("FailureRiskModel requires future_defective in training data")
        X = self._frame(train); y = self._labels(train)
        if np.unique(y).size < 2:
            raise ValueError("FailureRiskModel requires both classes in training data")
        imp = SimpleImputer(strategy="median", add_indicator=True)
        Xi = imp.fit_transform(X)
        pos = max(float(y.mean()), 1e-6)
        weights = np.where(y == 1, 0.5 / pos, 0.5 / max(1.0 - pos, 1e-6))
        weights = weights / np.mean(weights)
        model = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.035, max_leaf_nodes=7,
            min_samples_leaf=10, l2_regularization=1.5,
            random_state=self.seed,
        )
        try:
            model.fit(Xi, y, sample_weight=weights)
        except TypeError:
            model.fit(Xi, y)

        calibrator = None
        calibration = "none"
        if validation is not None and "future_defective" in validation.columns:
            vy = self._labels(validation)
            raw = model.predict_proba(imp.transform(self._frame(validation)))[:, 1]
            # Platt calibration is intentionally preferred to isotonic here:
            # with 80 validation components isotonic commonly maps the upper
            # tail to exactly 1.0, destroying probability resolution.
            if np.unique(vy).size >= 2 and np.unique(raw).size >= 4:
                calibrator = LogisticRegression(C=10.0, solver="lbfgs", random_state=self.seed)
                calibrator.fit(raw.reshape(-1, 1), vy)
                calibration = "platt_validation"
        self.artifact = RiskArtifact(imp, model, calibrator, FEATURES.copy(), self.seed, pos, calibration)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        if self.artifact is None:
            raise RuntimeError("FailureRiskModel is not fitted")
        a = self.artifact
        raw = a.model.predict_proba(a.imputer.transform(a_frame := self._frame(d)))[:, 1]
        if a.calibrator is not None:
            raw = a.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.artifact, path)

    @classmethod
    def load(cls, path: Path):
        x = cls(); x.artifact = joblib.load(path); return x
