from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class SafetyConfig:
    risk_reject_threshold: float = 0.75
    risk_review_threshold: float = 0.40
    ood_novelty_threshold: float = 0.70
    max_uncertainty_threshold: float = 0.65
    anomaly_review_threshold: float = 0.60
    rapid_degradation_threshold: float = 0.60
    crossing_confidence_threshold: float = 0.80
    min_reference_rows: int = 12
    known_family_distance_quantile: float = 0.99
    high_ood_blocks_risk_reject: bool = True


def load_policy(
    path: str | Path = "configs/policy.yaml",
) -> dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        result = yaml.safe_load(handle) or {}

    if not isinstance(result, dict):
        raise ValueError(
            f"Invalid policy YAML: {path}"
        )

    return result


def safety_config_from_policy(
    policy: Mapping[str, Any],
) -> SafetyConfig:
    thresholds = policy.get(
        "thresholds",
        {},
    )

    ood = policy.get(
        "ood",
        {},
    )

    decision = policy.get(
        "decision",
        {},
    )

    return SafetyConfig(
        risk_reject_threshold=float(
            thresholds.get(
                "risk_reject_threshold",
                0.75,
            )
        ),
        risk_review_threshold=float(
            thresholds.get(
                "risk_review_threshold",
                0.40,
            )
        ),
        ood_novelty_threshold=float(
            thresholds.get(
                "ood_novelty_threshold",
                0.70,
            )
        ),
        max_uncertainty_threshold=float(
            thresholds.get(
                "max_uncertainty_threshold",
                0.65,
            )
        ),
        anomaly_review_threshold=float(
            thresholds.get(
                "anomaly_review_threshold",
                0.60,
            )
        ),
        rapid_degradation_threshold=float(
            thresholds.get(
                "rapid_degradation_threshold",
                0.60,
            )
        ),
        crossing_confidence_threshold=float(
            thresholds.get(
                "crossing_confidence_threshold",
                0.80,
            )
        ),
        min_reference_rows=int(
            ood.get(
                "min_reference_rows",
                12,
            )
        ),
        known_family_distance_quantile=float(
            ood.get(
                "known_family_distance_quantile",
                0.99,
            )
        ),
        high_ood_blocks_risk_reject=bool(
            decision.get(
                "high_ood_blocks_risk_reject",
                True,
            )
        ),
    )


def _numeric(
    row: pd.Series,
    key: str,
    default: float = 0.0,
) -> float:
    value = pd.to_numeric(
        pd.Series([row.get(key, default)]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return float(default)

    return float(value)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = np.exp(-value)
        return float(
            1.0 / (1.0 + exp_value)
        )

    exp_value = np.exp(value)

    return float(
        exp_value
        / (1.0 + exp_value)
    )


def _ood_vector(
    row: pd.Series,
) -> np.ndarray:
    values = [
        _numeric(row, "robust_z"),
        _numeric(row, "slope"),
        _numeric(row, "acceleration"),
        _numeric(
            row,
            "rate_of_approach_to_limit",
        ),
        _numeric(
            row,
            "ewma_shift",
        ),
        _numeric(
            row,
            "cusum",
        ),
    ]

    return np.asarray(
        values,
        dtype=float,
    )


def compute_ood(
    triage: pd.DataFrame,
    config: SafetyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if triage.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=object),
        )

    families = (
        triage["family"]
        .fillna("UNKNOWN")
        .astype(str)
        .to_numpy()
    )

    matrix = np.vstack(
        [
            _ood_vector(row)
            for _, row in triage.iterrows()
        ]
    )

    scores = np.zeros(
        len(triage),
        dtype=float,
    )

    status = np.full(
        len(triage),
        "LOW",
        dtype=object,
    )

    for family in np.unique(families):
        indices = np.flatnonzero(
            families == family
        )

        family_matrix = matrix[
            indices
        ]

        if family == "UNKNOWN":
            scores[indices] = 1.0
            status[indices] = "SEVERE"
            continue

        if len(family_matrix) < config.min_reference_rows:
            median = np.median(
                family_matrix,
                axis=0,
            )

            mad = (
                np.median(
                    np.abs(
                        family_matrix
                        - median
                    ),
                    axis=0,
                )
                + 1e-6
            )

            distance = np.sqrt(
                np.mean(
                    (
                        (
                            family_matrix
                            - median
                        )
                        / mad
                    )
                    ** 2,
                    axis=1,
                )
            )

            threshold = max(
                float(
                    np.quantile(
                        distance,
                        0.95,
                    )
                ),
                1.0,
            )

            local = np.clip(
                distance / threshold,
                0.0,
                1.0,
            )

            scores[indices] = local

            status[indices] = np.where(
                local >= config.ood_novelty_threshold,
                "MODERATE",
                "LOW",
            )

            continue

        median = np.median(
            family_matrix,
            axis=0,
        )

        mad = (
            np.median(
                np.abs(
                    family_matrix
                    - median
                ),
                axis=0,
            )
            + 1e-6
        )

        distance = np.sqrt(
            np.mean(
                (
                    (
                        family_matrix
                        - median
                    )
                    / mad
                )
                ** 2,
                axis=1,
            )
        )

        reference_threshold = max(
            float(
                np.quantile(
                    distance,
                    config.known_family_distance_quantile,
                )
            ),
            1.0,
        )

        local_score = np.clip(
            distance
            / reference_threshold,
            0.0,
            1.0,
        )

        scores[indices] = local_score

        status[indices] = np.where(
            local_score
            >= config.ood_novelty_threshold,
            "MODERATE",
            "LOW",
        )

    return scores, status


def failure_risk(
    row: pd.Series,
    policy: Mapping[str, Any],
) -> tuple[float, str]:
    risk_config = policy.get(
        "risk",
        {},
    )

    weights = risk_config.get(
        "weights",
        {},
    )

    anomaly_weight = float(
        weights.get(
            "anomaly",
            0.40,
        )
    )

    crossing_weight = float(
        weights.get(
            "crossing",
            0.30,
        )
    )

    acceleration_weight = float(
        weights.get(
            "acceleration",
            0.20,
        )
    )

    proximity_weight = float(
        weights.get(
            "proximity",
            0.10,
        )
    )

    weight_total = (
        anomaly_weight
        + crossing_weight
        + acceleration_weight
        + proximity_weight
    )

    if weight_total <= 0:
        weight_total = 1.0

    anomaly_weight /= weight_total
    crossing_weight /= weight_total
    acceleration_weight /= weight_total
    proximity_weight /= weight_total

    anomaly = float(
        np.clip(
            _numeric(
                row,
                "anomaly_score",
            ),
            0.0,
            1.0,
        )
    )

    crossing = (
        1.0
        if bool(
            row.get(
                "predicted_limit_crossing",
                False,
            )
        )
        else 0.0
    )

    acceleration = float(
        np.clip(
            abs(
                _numeric(
                    row,
                    "acceleration",
                )
            )
            / 5.0,
            0.0,
            1.0,
        )
    )

    value = _numeric(
        row,
        "value",
        np.nan,
    )

    distance = _numeric(
        row,
        "distance_to_limit",
        np.nan,
    )

    if (
        np.isfinite(value)
        and np.isfinite(distance)
    ):
        proximity = float(
            np.clip(
                1.0
                - abs(distance)
                / max(
                    abs(value)
                    + abs(distance),
                    1e-9,
                ),
                0.0,
                1.0,
            )
        )
    else:
        proximity = 0.0

    evidence = (
        anomaly_weight * anomaly
        + crossing_weight * crossing
        + acceleration_weight * acceleration
        + proximity_weight * proximity
    )

    calibration = risk_config.get(
        "calibration",
        {},
    )

    slope = float(
        calibration.get(
            "slope",
            4.0,
        )
    )

    intercept = float(
        calibration.get(
            "intercept",
            -2.0,
        )
    )

    risk = _sigmoid(
        slope * evidence
        + intercept
    )

    crossing_confidence = _numeric(
        row,
        "forecast_confidence",
    )

    threshold = float(
        policy.get(
            "thresholds",
            {},
        ).get(
            "crossing_confidence_threshold",
            0.80,
        )
    )

    if (
        crossing
        and crossing_confidence >= threshold
    ):
        risk = max(
            risk,
            0.76,
        )

    return (
        float(
            np.clip(
                risk,
                0.0,
                1.0,
            )
        ),
        "EVIDENCE_FUSION",
    )


def _hard_limit_exceeded(row: pd.Series) -> bool:
    value = _numeric(
        row,
        "value",
        np.nan,
    )

    if not np.isfinite(value):
        return False

    lower = row.get(
        "lower_limit"
    )

    upper = row.get(
        "upper_limit"
    )

    if lower is not None and pd.notna(lower):
        if value < float(lower):
            return True

    if upper is not None and pd.notna(upper):
        if value > float(upper):
            return True

    return False


def apply_safety_policy(
    triage: pd.DataFrame,
    policy: Mapping[str, Any],
) -> pd.DataFrame:
    if triage.empty:
        result = triage.copy()

        for column in [
            "ood_score",
            "failure_risk",
        ]:
            result[column] = pd.Series(
                dtype=float
            )

        return result

    config = safety_config_from_policy(
        policy
    )

    result = triage.copy()

    ood_score, ood_status = compute_ood(
        result,
        config,
    )

    result["ood_score"] = ood_score
    result["ood_status"] = ood_status

    risks: list[float] = []
    modes: list[str] = []
    dispositions: list[str] = []
    reasons: list[str] = []

    for position, (_, row) in enumerate(
        result.iterrows()
    ):
        risk, risk_mode = failure_risk(
            row,
            policy,
        )

        risks.append(risk)
        modes.append(risk_mode)

        hard_violation = _hard_limit_exceeded(
            row
        )

        high_ood = (
            ood_score[position]
            >= config.ood_novelty_threshold
            or ood_status[position]
            == "SEVERE"
        )

        crossing = bool(
            row.get(
                "predicted_limit_crossing",
                False,
            )
        )

        confidence = _numeric(
            row,
            "forecast_confidence",
        )

        anomaly = _numeric(
            row,
            "anomaly_score",
        )

        acceleration = abs(
            _numeric(
                row,
                "acceleration",
            )
        )

        forecast_mode = str(
            row.get(
                "forecast_mode",
                "INSUFFICIENT",
            )
        )

        uncertainty = float(
            np.clip(
                1.0 - confidence,
                0.0,
                1.0,
            )
        )

        # OOD is intentionally checked before model-based REJECT decisions:
        # a novel domain is a trust problem, not physical-failure evidence.
        if hard_violation:
            disposition = "REJECT"
            reason = (
                "CURRENT_HARD_LIMIT_EXCEEDED"
            )

        elif (
            high_ood
            and config.high_ood_blocks_risk_reject
        ):
            disposition = "UNKNOWN"
            reason = (
                "NOVEL_DOMAIN_OR_FAMILY_INSPECTION"
            )

        elif (
            risk >= config.risk_reject_threshold
            or (
                crossing
                and confidence
                >= config.crossing_confidence_threshold
            )
        ):
            disposition = "REJECT"

            if (
                crossing
                and confidence
                >= config.crossing_confidence_threshold
            ):
                reason = (
                    "PREDICTED_LIMIT_CROSSING_HIGH_CONFIDENCE"
                )
            else:
                reason = (
                    "CALIBRATED_HIGH_FAILURE_RISK"
                )

        elif (
            anomaly
            >= config.anomaly_review_threshold
            or acceleration
            >= config.rapid_degradation_threshold
        ):
            disposition = "REVIEW"
            reason = (
                "SIGNIFICANT_ANOMALY_OR_RAPID_DEGRADATION"
            )

        elif (
            uncertainty
            >= config.max_uncertainty_threshold
            and forecast_mode in {
                "COLD_START",
                "INSUFFICIENT",
            }
        ):
            disposition = (
                "UNKNOWN"
                if forecast_mode
                == "INSUFFICIENT"
                else "REVIEW"
            )

            reason = (
                "HIGH_FORECAST_UNCERTAINTY"
            )

        elif (
            forecast_mode
            == "INSUFFICIENT"
        ):
            disposition = "UNKNOWN"
            reason = (
                "INSUFFICIENT_TIME_HISTORY"
            )

        elif (
            risk
            >= config.risk_review_threshold
        ):
            disposition = "REVIEW"
            reason = (
                "MODERATE_CALIBRATED_FAILURE_RISK"
            )

        else:
            disposition = "PASS"
            reason = (
                "SAFE_IN_DOMAIN_LOW_RISK"
            )

        dispositions.append(
            disposition
        )
        reasons.append(reason)

    result["failure_risk"] = np.asarray(
        risks,
        dtype=float,
    )

    result["risk_mode"] = modes
    result["disposition"] = dispositions
    result["policy_reason_code"] = reasons

    return result
