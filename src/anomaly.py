from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class AnomalyConfig:
    min_iforest_rows: int = 12
    random_state: int = 42

    robust_weight: float = 0.35
    isolation_weight: float = 0.25
    temporal_weight: float = 0.20
    multivariate_weight: float = 0.20


def empirical_cdf(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    if finite.sum() <= 1:
        return result

    ranks = pd.Series(
        values[finite]
    ).rank(
        method="average",
        pct=True,
    ).to_numpy()

    result[finite] = np.clip(
        ranks,
        0.0,
        1.0,
    )

    return result


def _robust_pat_score(
    features: pd.DataFrame,
) -> np.ndarray:
    z = pd.to_numeric(
        features["robust_z"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    return empirical_cdf(
        np.abs(z)
    )


def _temporal_score(
    features: pd.DataFrame,
) -> np.ndarray:
    slope = pd.to_numeric(
        features["slope"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    acceleration = pd.to_numeric(
        features["acceleration"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    change_point = (
        features["change_point"]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    raw = (
        np.abs(slope)
        + 0.50 * np.abs(acceleration)
        + change_point.astype(float)
    )

    return empirical_cdf(raw)


def _multivariate_score(
    features: pd.DataFrame,
) -> tuple[np.ndarray, str]:
    columns = [
        "robust_z",
        "slope",
        "acceleration",
        "distance_to_limit",
        "rate_of_approach_to_limit",
        "ewma_shift",
        "cusum",
    ]

    available = [
        column
        for column in columns
        if column in features.columns
    ]

    if len(available) < 2 or len(features) < 8:
        return (
            np.zeros(len(features), dtype=float),
            "DEGRADED",
        )

    matrix = (
        features[available]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    matrix = matrix.fillna(
        matrix.median()
    ).fillna(0.0)

    scaler = RobustScaler()
    transformed = scaler.fit_transform(matrix)

    center = np.median(
        transformed,
        axis=0,
    )

    covariance = np.cov(
        transformed,
        rowvar=False,
    )

    if np.ndim(covariance) == 0:
        return (
            np.zeros(len(features), dtype=float),
            "DEGRADED",
        )

    covariance += np.eye(
        covariance.shape[0]
    ) * 1e-6

    try:
        inverse = np.linalg.pinv(
            covariance
        )
    except np.linalg.LinAlgError:
        return (
            np.zeros(len(features), dtype=float),
            "DEGRADED",
        )

    delta = transformed - center

    distance = np.sqrt(
        np.maximum(
            0.0,
            np.einsum(
                "ij,jk,ik->i",
                delta,
                inverse,
                delta,
            ),
        )
    )

    return (
        empirical_cdf(distance),
        "FULL",
    )


def _isolation_score(
    features: pd.DataFrame,
    config: AnomalyConfig,
) -> tuple[np.ndarray, str]:
    if len(features) < config.min_iforest_rows:
        return (
            np.zeros(len(features), dtype=float),
            "DEGRADED",
        )

    columns = [
        "value",
        "robust_z",
        "slope",
        "acceleration",
        "distance_to_limit",
        "rate_of_approach_to_limit",
        "ewma_shift",
        "cusum",
    ]

    available = [
        column
        for column in columns
        if column in features.columns
    ]

    if len(available) < 2:
        return (
            np.zeros(len(features), dtype=float),
            "DEGRADED",
        )

    matrix = (
        features[available]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    matrix = matrix.fillna(
        matrix.median()
    ).fillna(0.0)

    model = IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=config.random_state,
    )

    try:
        model.fit(matrix)

        raw = -model.score_samples(matrix)

        return (
            empirical_cdf(raw),
            "FULL",
        )
    except Exception:
        return (
            np.zeros(len(features), dtype=float),
            "DEGRADED",
        )


def detect_anomalies(
    features: pd.DataFrame,
    config: AnomalyConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = AnomalyConfig()

    if features.empty:
        result = features.copy()

        for column in [
            "robust_pat_score",
            "isolation_forest_score",
            "temporal_score",
            "multivariate_score",
            "anomaly_score",
        ]:
            result[column] = pd.Series(
                dtype=float
            )

        result["capability_mode"] = pd.Series(
            dtype=str
        )

        return result

    result = features.copy()

    robust = _robust_pat_score(result)
    temporal = _temporal_score(result)
    isolation, isolation_mode = _isolation_score(
        result,
        config,
    )
    multivariate, multivariate_mode = _multivariate_score(
        result,
    )

    weights = np.array(
        [
            config.robust_weight,
            config.isolation_weight,
            config.temporal_weight,
            config.multivariate_weight,
        ],
        dtype=float,
    )

    weights /= weights.sum()

    raw_fusion = (
        robust * weights[0]
        + isolation * weights[1]
        + temporal * weights[2]
        + multivariate * weights[3]
    )

    calibrated = empirical_cdf(
        raw_fusion
    )

    result["robust_pat_score"] = robust
    result["isolation_forest_score"] = isolation
    result["temporal_score"] = temporal
    result["multivariate_score"] = multivariate
    result["anomaly_score"] = calibrated

    reference_levels = set(
        result.get(
            "reference_level",
            pd.Series(
                "INSUFFICIENT",
                index=result.index,
            ),
        ).astype(str)
    )

    if (
        isolation_mode == "FULL"
        and multivariate_mode == "FULL"
        and not reference_levels.intersection(
            {"INSUFFICIENT"}
        )
    ):
        capability = "FULL"
    elif reference_levels.intersection(
        {"LOT", "FAMILY", "BATCH", "GLOBAL"}
    ):
        capability = "DEGRADED"
    else:
        capability = "BATCH_FALLBACK"

    result["capability_mode"] = capability

    result["anomaly_evidence_available"] = (
        result[
            [
                "robust_pat_score",
                "isolation_forest_score",
                "temporal_score",
                "multivariate_score",
            ]
        ]
        .gt(0.0)
        .any(axis=1)
    )

    return result
