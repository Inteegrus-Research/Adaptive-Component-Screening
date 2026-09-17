from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class ForecastConfig:
    horizon_h: float = 168.0
    min_component_points: int = 2
    min_training_components: int = 4
    min_training_rows: int = 24
    conformal_alpha: float = 0.05
    min_calibration_residuals: int = 8
    cold_start_min_points_for_trend: int = 3
    uncertainty_growth_per_sqrt_hour: float = 0.02
    random_state: int = 42


def _robust_slope(
    time: np.ndarray,
    value: np.ndarray,
) -> float:
    if len(time) < 2:
        return 0.0

    slopes: list[float] = []

    for i in range(len(time) - 1):
        dt = time[i + 1:] - time[i]
        dy = value[i + 1:] - value[i]

        valid = (
            np.isfinite(dt)
            & np.isfinite(dy)
            & (dt != 0)
        )

        if np.any(valid):
            slopes.extend(
                (dy[valid] / dt[valid]).tolist()
            )

    return (
        float(np.median(slopes))
        if slopes
        else 0.0
    )


def _feature_vector(
    time: np.ndarray,
    value: np.ndarray,
    target_time: float,
) -> np.ndarray:
    current_time = float(time[-1])
    current_value = float(value[-1])

    slope = _robust_slope(
        time,
        value,
    )

    recent_start = max(
        0,
        len(time) - 4,
    )

    recent_slope = _robust_slope(
        time[recent_start:],
        value[recent_start:],
    )

    delta = (
        float(value[-1] - value[-2])
        if len(value) >= 2
        else 0.0
    )

    return np.asarray(
        [
            current_value,
            slope,
            recent_slope,
            delta,
            current_time,
            target_time,
            float(len(value)),
        ],
        dtype=float,
    )


def _build_training_table(
    data: pd.DataFrame,
    family: str,
    parameter: str,
    target_component: str,
) -> pd.DataFrame:
    subset = data.loc[
        data["family"].astype(str).eq(str(family))
        & data["parameter"].astype(str).eq(str(parameter))
        & ~data["component_id"].astype(str).eq(
            str(target_component)
        )
    ].copy()

    rows: list[dict[str, Any]] = []

    for component_id, group in subset.groupby(
        "component_id",
        sort=False,
    ):
        group = group.sort_values("time_h")

        time = pd.to_numeric(
            group["time_h"],
            errors="coerce",
        ).to_numpy(dtype=float)

        value = pd.to_numeric(
            group["value"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(time)
            & np.isfinite(value)
        )

        time = time[valid]
        value = value[valid]

        if len(value) < 2:
            continue

        for index in range(1, len(value)):
            x = _feature_vector(
                time[:index],
                value[:index],
                float(time[index]),
            )

            rows.append(
                {
                    "component_id": str(component_id),
                    "x0": x[0],
                    "x1": x[1],
                    "x2": x[2],
                    "x3": x[3],
                    "x4": x[4],
                    "x5": x[5],
                    "x6": x[6],
                    "target": float(value[index]),
                }
            )

    return pd.DataFrame(rows)


def _fit_ml(
    data: pd.DataFrame,
    history: pd.DataFrame,
    family: str,
    parameter: str,
    component_id: str,
    horizon: float,
    config: ForecastConfig,
) -> dict[str, Any] | None:
    training = _build_training_table(
        data,
        family,
        parameter,
        component_id,
    )

    if training.empty:
        return None

    if training["component_id"].nunique() < (
        config.min_training_components
    ):
        return None

    if len(training) < config.min_training_rows:
        return None

    feature_columns = [
        f"x{i}"
        for i in range(7)
    ]

    # Component-wise temporal holdout for conformal calibration.
    calibration_mask = (
        training.groupby(
            "component_id",
            sort=False,
        ).cumcount(ascending=False) == 0
    )

    calibration = training.loc[
        calibration_mask
    ]

    model_training = training.loc[
        ~calibration_mask
    ]

    if len(model_training) < config.min_training_rows:
        model_training = training.copy()
        calibration = training.copy()

    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1e-3,
        random_state=config.random_state,
    )

    model.fit(
        model_training[feature_columns],
        model_training["target"],
    )

    calibration_prediction = model.predict(
        calibration[feature_columns]
    )

    residual = np.abs(
        calibration["target"].to_numpy()
        - calibration_prediction
    )

    if len(residual) >= config.min_calibration_residuals:
        quantile = 1.0 - config.conformal_alpha

        radius = float(
            np.quantile(
                residual,
                quantile,
                method="higher",
            )
        )
    else:
        radius = float(
            np.quantile(
                residual,
                0.95,
            )
            if len(residual)
            else 0.0
        )

    time = pd.to_numeric(
        history["time_h"],
        errors="coerce",
    ).to_numpy(dtype=float)

    value = pd.to_numeric(
        history["value"],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(time)
        & np.isfinite(value)
    )

    time = time[valid]
    value = value[valid]

    x = _feature_vector(
        time,
        value,
        horizon,
    )

    prediction = float(
        model.predict(
            x.reshape(1, -1)
        )[0]
    )

    lower = prediction - radius
    upper = prediction + radius

    confidence = 1.0 / (
        1.0
        + radius
        / max(abs(prediction), 1e-9)
    )

    return {
        "forecast_mode": "ML_FORECAST",
        "predicted_value_at_horizon": prediction,
        "prediction_lower": float(lower),
        "prediction_upper": float(upper),
        "conformal_radius": float(radius),
        "forecast_confidence": float(
            np.clip(confidence, 0.0, 1.0)
        ),
        "training_components": int(
            training["component_id"].nunique()
        ),
        "training_rows": int(
            len(model_training)
        ),
    }


def _cold_start(
    history: pd.DataFrame,
    horizon: float,
    config: ForecastConfig,
) -> dict[str, Any]:
    time = pd.to_numeric(
        history["time_h"],
        errors="coerce",
    ).to_numpy(dtype=float)

    value = pd.to_numeric(
        history["value"],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(time)
        & np.isfinite(value)
    )

    time = time[valid]
    value = value[valid]

    if len(value) < config.min_component_points:
        return {
            "forecast_mode": "INSUFFICIENT",
            "predicted_value_at_horizon": np.nan,
            "prediction_lower": np.nan,
            "prediction_upper": np.nan,
            "conformal_radius": np.nan,
            "forecast_confidence": 0.0,
            "training_components": 0,
            "training_rows": 0,
        }

    current_time = float(time[-1])
    current_value = float(value[-1])

    if len(value) >= config.cold_start_min_points_for_trend:
        slope = _robust_slope(
            time,
            value,
        )

        predicted = (
            current_value
            + slope * (horizon - current_time)
        )

        fitted = (
            current_value
            + slope * (time - current_time)
        )

        residual = value - fitted

        mad = (
            float(
                np.median(
                    np.abs(
                        residual
                        - np.median(residual)
                    )
                )
            )
            if len(residual)
            else 0.0
        )
    else:
        slope = 0.0
        predicted = current_value

        delta = (
            np.diff(value)
            if len(value) > 1
            else np.asarray([0.0])
        )

        mad = float(
            np.median(
                np.abs(
                    delta
                    - np.median(delta)
                )
            )
        )

    elapsed = max(
        horizon - current_time,
        0.0,
    )

    uncertainty = max(
        mad,
        1e-9,
    ) * (
        1.0
        + config.uncertainty_growth_per_sqrt_hour
        * np.sqrt(elapsed)
    )

    lower = predicted - uncertainty
    upper = predicted + uncertainty

    confidence = 1.0 / (
        1.0
        + uncertainty
        / max(abs(predicted), 1e-9)
    )

    return {
        "forecast_mode": "COLD_START",
        "predicted_value_at_horizon": float(predicted),
        "prediction_lower": float(lower),
        "prediction_upper": float(upper),
        "conformal_radius": float(uncertainty),
        "forecast_confidence": float(
            np.clip(confidence, 0.0, 1.0)
        ),
        "training_components": 0,
        "training_rows": 0,
    }


def _crossing(
    history: pd.DataFrame,
    forecast: Mapping[str, Any],
    target_horizon: float,
) -> tuple[bool, float | None, float]:
    time = pd.to_numeric(
        history["time_h"],
        errors="coerce",
    ).to_numpy(dtype=float)

    value = pd.to_numeric(
        history["value"],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(time)
        & np.isfinite(value)
    )

    time = time[valid]
    value = value[valid]

    if len(value) == 0:
        return False, None, 0.0

    current_time = float(time[-1])
    current_value = float(value[-1])

    lower_limit = forecast.get(
        "lower_limit"
    )

    upper_limit = forecast.get(
        "upper_limit"
    )

    predicted = float(
        forecast["predicted_value_at_horizon"]
    )

    lower = float(
        forecast["prediction_lower"]
    )

    upper = float(
        forecast["prediction_upper"]
    )

    confidence = float(
        forecast["forecast_confidence"]
    )

    if upper_limit is not None:
        boundary = float(upper_limit)

        if current_value > boundary:
            return True, current_time, 1.0

        if upper >= boundary:
            denominator = predicted - current_value

            if denominator > 0:
                fraction = (
                    boundary - current_value
                ) / denominator

                estimated = (
                    current_time
                    + fraction
                    * (
                        target_horizon
                        - current_time
                    )
                )

                return (
                    True,
                    float(estimated),
                    max(confidence, 0.80),
                )

            return (
                True,
                None,
                max(confidence, 0.80),
            )

    if lower_limit is not None:
        boundary = float(lower_limit)

        if current_value < boundary:
            return True, current_time, 1.0

        if lower <= boundary:
            denominator = predicted - current_value

            if denominator < 0:
                fraction = (
                    boundary - current_value
                ) / denominator

                estimated = (
                    current_time
                    + fraction
                    * (
                        target_horizon
                        - current_time
                    )
                )

                return (
                    True,
                    float(estimated),
                    max(confidence, 0.80),
                )

            return (
                True,
                None,
                max(confidence, 0.80),
            )

    return False, None, confidence


def forecast_components(
    features: pd.DataFrame,
    canonical: pd.DataFrame,
    parameter_config: Mapping[str, Any],
    config: ForecastConfig | None = None,
    target_horizon_h: float | None = None,
) -> pd.DataFrame:
    if config is None:
        config = ForecastConfig()

    horizon = float(
        target_horizon_h
        if target_horizon_h is not None
        else config.horizon_h
    )

    if features.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    for (component_id, parameter), latest_group in features.groupby(
        ["component_id", "parameter"],
        sort=False,
    ):
        history = canonical.loc[
            canonical["component_id"].astype(str).eq(
                str(component_id)
            )
            & canonical["parameter"].astype(str).eq(
                str(parameter)
            )
        ].sort_values("time_h")

        if history.empty:
            continue

        family_values = (
            history["family"]
            .dropna()
            .astype(str)
        )

        family = (
            family_values.iloc[-1]
            if not family_values.empty
            else "UNKNOWN"
        )

        model_forecast = _fit_ml(
            canonical,
            history,
            family,
            str(parameter),
            str(component_id),
            horizon,
            config,
        )

        if model_forecast is None:
            model_forecast = _cold_start(
                history,
                horizon,
                config,
            )

        parameter_spec = (
            parameter_config
            .get("parameters", {})
            .get(str(parameter), {})
        )

        engineering = parameter_spec.get(
            "engineering_limit",
            {},
        )

        lower_limit = engineering.get(
            "lower"
        )

        upper_limit = engineering.get(
            "upper"
        )

        crossing_forecast = dict(
            model_forecast
        )

        crossing_forecast["lower_limit"] = (
            lower_limit
        )

        crossing_forecast["upper_limit"] = (
            upper_limit
        )

        crossing, crossing_time, crossing_confidence = _crossing(
            history,
            crossing_forecast,
            horizon,
        )

        records.append(
            {
                "component_id": str(component_id),
                "parameter": str(parameter),
                "forecast_mode": model_forecast[
                    "forecast_mode"
                ],
                "predicted_value_at_horizon": model_forecast[
                    "predicted_value_at_horizon"
                ],
                "prediction_lower": model_forecast[
                    "prediction_lower"
                ],
                "prediction_upper": model_forecast[
                    "prediction_upper"
                ],
                "conformal_radius": model_forecast[
                    "conformal_radius"
                ],
                "forecast_confidence": max(
                    float(
                        model_forecast[
                            "forecast_confidence"
                        ]
                    ),
                    crossing_confidence
                    if crossing
                    else 0.0,
                ),
                "predicted_limit_crossing": bool(
                    crossing
                ),
                "estimated_crossing_time_h": (
                    crossing_time
                ),
                "training_components": model_forecast[
                    "training_components"
                ],
                "training_rows": model_forecast[
                    "training_rows"
                ],
                "target_horizon_h": horizon,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit,
            }
        )

    return pd.DataFrame(records)
