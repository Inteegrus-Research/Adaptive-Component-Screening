from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    min_reference_group_size: int = 8
    mad_epsilon: float = 1e-9
    ewma_alpha: float = 0.25
    cusum_drift: float = 0.50
    cusum_threshold: float = 2.50


def robust_mad(
    values: np.ndarray | pd.Series,
    epsilon: float = 1e-9,
) -> float:
    array = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if array.size == 0:
        return float("nan")

    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))

    return max(mad, epsilon)


def _reference(
    values: pd.Series,
    minimum_size: int,
    epsilon: float,
) -> tuple[float, float] | None:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(numeric) < minimum_size:
        return None

    median = float(np.median(numeric))
    mad = robust_mad(numeric, epsilon)

    return median, mad


def _reference_for_row(
    frame: pd.DataFrame,
    row: pd.Series,
    parameter: str,
    config: FeatureConfig,
) -> tuple[float, float, str]:
    candidates: list[tuple[str, pd.Series]] = []

    base = frame[frame["parameter"].astype(str).eq(str(parameter))]

    lot_id = str(row.get("lot_id", ""))
    family = str(row.get("family", ""))
    batch_id = str(row.get("batch_id", ""))

    if lot_id:
        candidates.append(
            (
                "LOT",
                base.loc[
                    base["lot_id"].astype(str).eq(lot_id),
                    "value",
                ],
            )
        )

    if family:
        candidates.append(
            (
                "FAMILY",
                base.loc[
                    base["family"].astype(str).eq(family),
                    "value",
                ],
            )
        )

    if batch_id:
        candidates.append(
            (
                "BATCH",
                base.loc[
                    base["batch_id"].astype(str).eq(batch_id),
                    "value",
                ],
            )
        )

    candidates.append(("GLOBAL", base["value"]))

    for level, values in candidates:
        result = _reference(
            values,
            config.min_reference_group_size,
            config.mad_epsilon,
        )

        if result is not None:
            return result[0], result[1], level

    fallback = _reference(base["value"], 2, config.mad_epsilon)

    if fallback is None:
        return np.nan, np.nan, "INSUFFICIENT"

    return fallback[0], fallback[1], "GLOBAL"


def _robust_slope(
    times: np.ndarray,
    values: np.ndarray,
) -> float:
    if len(times) < 2:
        return 0.0

    slopes: list[float] = []

    for i in range(len(times) - 1):
        dt = times[i + 1:] - times[i]
        dy = values[i + 1:] - values[i]

        valid = np.isfinite(dt) & np.isfinite(dy) & (dt != 0)

        if np.any(valid):
            slopes.extend(
                (dy[valid] / dt[valid]).tolist()
            )

    return (
        float(np.median(slopes))
        if slopes
        else 0.0
    )


def _time_features(
    times: np.ndarray,
    values: np.ndarray,
    alpha: float,
    cusum_drift: float,
    cusum_threshold: float,
) -> pd.DataFrame:
    count = len(values)

    delta = np.zeros(count, dtype=float)
    slope = np.zeros(count, dtype=float)
    acceleration = np.zeros(count, dtype=float)
    ewma_shift = np.zeros(count, dtype=float)
    cusum = np.zeros(count, dtype=float)
    change_point = np.zeros(count, dtype=bool)

    if count == 0:
        return pd.DataFrame()

    ewma = float(values[0])
    previous_slope = 0.0
    positive = 0.0
    negative = 0.0

    global_mad = robust_mad(values)

    for i in range(count):
        if i > 0:
            dt = times[i] - times[i - 1]

            if dt != 0:
                delta[i] = values[i] - values[i - 1]
                slope[i] = delta[i] / dt

                if i > 1:
                    acceleration[i] = (
                        slope[i] - previous_slope
                    ) / dt

            ewma = (
                alpha * values[i]
                + (1.0 - alpha) * ewma
            )

        ewma_shift[i] = values[i] - ewma
        previous_slope = slope[i]

        z = (
            (values[i] - np.median(values))
            / max(global_mad, 1e-9)
        )

        positive = max(
            0.0,
            positive + z - cusum_drift,
        )

        negative = min(
            0.0,
            negative + z + cusum_drift,
        )

        cusum[i] = max(
            positive,
            abs(negative),
        )

        change_point[i] = (
            cusum[i] >= cusum_threshold
        )

    return pd.DataFrame(
        {
            "delta": delta,
            "slope": slope,
            "acceleration": acceleration,
            "ewma_shift": ewma_shift,
            "cusum": cusum,
            "change_point": change_point,
        }
    )


def _limit_values(
    parameter: str,
    parameter_config: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None, float | None]:
    spec = parameter_config.get(
        "parameters",
        {},
    ).get(parameter, {})

    engineering = spec.get(
        "engineering_limit",
        {},
    )

    warning = spec.get(
        "warning_limit",
        {},
    )

    return (
        engineering.get("lower"),
        engineering.get("upper"),
        warning.get("lower"),
        warning.get("upper"),
    )


def _signed_distance(
    value: float,
    lower: float | None,
    upper: float | None,
) -> float:
    if lower is not None and value < lower:
        return value - lower

    if upper is not None and value > upper:
        return value - upper

    distances: list[float] = []

    if lower is not None:
        distances.append(value - lower)

    if upper is not None:
        distances.append(upper - value)

    return (
        float(min(distances))
        if distances
        else float("nan")
    )


def _warning_distance(
    value: float,
    lower: float | None,
    upper: float | None,
) -> float:
    return _signed_distance(
        value,
        lower,
        upper,
    )


def add_features(
    canonical: pd.DataFrame,
    parameter_config: Mapping[str, Any],
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = FeatureConfig()

    if canonical.empty:
        return canonical.copy()

    required = {
        "component_id",
        "parameter",
        "value",
        "time_h",
    }

    missing = required - set(canonical.columns)

    if missing:
        raise ValueError(
            f"Missing feature columns: {sorted(missing)}"
        )

    frame = canonical.copy()

    for field in ["lot_id", "batch_id", "family"]:
        if field not in frame.columns:
            frame[field] = ""

        frame[field] = (
            frame[field]
            .fillna("")
            .astype(str)
        )

    frame["value"] = pd.to_numeric(
        frame["value"],
        errors="coerce",
    )

    frame["time_h"] = pd.to_numeric(
        frame["time_h"],
        errors="coerce",
    )

    frame = frame.sort_values(
        [
            "component_id",
            "parameter",
            "time_h",
        ],
        kind="stable",
    )

    records: list[dict[str, Any]] = []

    for (component_id, parameter), group in frame.groupby(
        ["component_id", "parameter"],
        sort=False,
    ):
        group = group.sort_values("time_h")
        times = group["time_h"].to_numpy(dtype=float)
        values = group["value"].to_numpy(dtype=float)

        valid = (
            np.isfinite(times)
            & np.isfinite(values)
        )

        times = times[valid]
        values = values[valid]

        temporal = _time_features(
            times,
            values,
            alpha=config.ewma_alpha,
            cusum_drift=config.cusum_drift,
            cusum_threshold=config.cusum_threshold,
        )

        valid_indices = group.index[valid].tolist()

        for position, source_index in enumerate(valid_indices):
            row = group.loc[source_index]

            value = float(
                row["value"]
            )

            reference_median, reference_mad, reference_level = (
                _reference_for_row(
                    frame,
                    row,
                    str(parameter),
                    config,
                )
            )

            robust_z = (
                abs(value - reference_median)
                / max(reference_mad, config.mad_epsilon)
                if np.isfinite(reference_median)
                else np.nan
            )

            lower, upper, warning_lower, warning_upper = _limit_values(
                str(parameter),
                parameter_config,
            )

            distance_to_limit = _signed_distance(
                value,
                lower,
                upper,
            )

            distance_to_warning = _warning_distance(
                value,
                warning_lower,
                warning_upper,
            )

            current_slope = float(
                temporal.loc[position, "slope"]
            )

            if upper is not None:
                approach = max(
                    0.0,
                    current_slope,
                )
            elif lower is not None:
                approach = max(
                    0.0,
                    -current_slope,
                )
            else:
                approach = abs(current_slope)

            near_warning = False

            if warning_upper is not None:
                near_warning |= value >= warning_upper

            if warning_lower is not None:
                near_warning |= value <= warning_lower

            records.append(
                {
                    "component_id": component_id,
                    "parameter": parameter,
                    "lot_id": str(row.get("lot_id", "")),
                    "batch_id": str(row.get("batch_id", "")),
                    "family": str(row.get("family", "")),
                    "time_h": float(row["time_h"]),
                    "value": value,
                    "unit": row.get("unit", ""),
                    "reference_level": reference_level,
                    "reference_median": reference_median,
                    "reference_mad": reference_mad,
                    "robust_z": robust_z,
                    "delta": float(
                        temporal.loc[position, "delta"]
                    ),
                    "slope": current_slope,
                    "acceleration": float(
                        temporal.loc[position, "acceleration"]
                    ),
                    "ewma_shift": float(
                        temporal.loc[position, "ewma_shift"]
                    ),
                    "cusum": float(
                        temporal.loc[position, "cusum"]
                    ),
                    "change_point": bool(
                        temporal.loc[position, "change_point"]
                    ),
                    "distance_to_limit": distance_to_limit,
                    "distance_to_warning": distance_to_warning,
                    "rate_of_approach_to_limit": approach,
                    "near_warning": near_warning,
                    "lower_limit": lower,
                    "upper_limit": upper,
                    "warning_lower": warning_lower,
                    "warning_upper": warning_upper,
                    "source_index": source_index,
                }
            )

    result = pd.DataFrame(records)

    if result.empty:
        return result

    result["mad_deviation"] = (
        result["value"]
        - result["reference_median"]
    ).abs()

    result["iqr_deviation"] = (
        result
        .groupby("parameter")["value"]
        .transform(
            lambda series: (
                (series - series.median()).abs()
                / max(
                    float(
                        series.quantile(0.75)
                        - series.quantile(0.25)
                    ),
                    config.mad_epsilon,
                )
            )
        )
    )

    return result.sort_values(
        [
            "component_id",
            "parameter",
            "time_h",
        ]
    ).reset_index(drop=True)
