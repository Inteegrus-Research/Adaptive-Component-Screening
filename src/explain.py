from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExplanationTrace:
    component_id: str
    parameter: str
    observation: dict[str, Any]
    inference: dict[str, Any]
    risk_synthesis: dict[str, Any]
    disposition: str
    reason_codes: tuple[str, ...]
    narrative: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(
    value: Any,
    digits: int = 3,
) -> str:
    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return "N/A"

    if not np.isfinite(number):
        return "N/A"

    return f"{number:.{digits}f}"


def explain_row(
    row: pd.Series,
) -> ExplanationTrace:
    component_id = str(
        row.get(
            "component_id",
            "UNKNOWN",
        )
    )

    parameter = str(
        row.get(
            "parameter",
            "UNKNOWN_PARAMETER",
        )
    )

    disposition = str(
        row.get(
            "disposition",
            "UNKNOWN",
        )
    )

    observation = {
        "component_id": component_id,
        "parameter": parameter,
        "family": row.get("family", ""),
        "lot_id": row.get("lot_id", ""),
        "batch_id": row.get("batch_id", ""),
        "time_h": row.get("time_h"),
        "value": row.get("value"),
        "unit": row.get("unit", ""),
        "reference_level": row.get(
            "reference_level",
            "INSUFFICIENT",
        ),
        "reference_median": row.get(
            "reference_median"
        ),
        "reference_mad": row.get(
            "reference_mad"
        ),
        "lower_limit": row.get(
            "lower_limit"
        ),
        "upper_limit": row.get(
            "upper_limit"
        ),
    }

    inference = {
        "anomaly_score": row.get(
            "anomaly_score"
        ),
        "robust_pat_score": row.get(
            "robust_pat_score"
        ),
        "isolation_forest_score": row.get(
            "isolation_forest_score"
        ),
        "temporal_score": row.get(
            "temporal_score"
        ),
        "multivariate_score": row.get(
            "multivariate_score"
        ),
        "slope": row.get(
            "slope"
        ),
        "acceleration": row.get(
            "acceleration"
        ),
        "ewma_shift": row.get(
            "ewma_shift"
        ),
        "cusum": row.get(
            "cusum"
        ),
        "change_point": row.get(
            "change_point"
        ),
        "forecast_mode": row.get(
            "forecast_mode"
        ),
        "predicted_value_at_horizon": row.get(
            "predicted_value_at_horizon"
        ),
        "prediction_lower": row.get(
            "prediction_lower"
        ),
        "prediction_upper": row.get(
            "prediction_upper"
        ),
        "predicted_limit_crossing": row.get(
            "predicted_limit_crossing"
        ),
        "estimated_crossing_time_h": row.get(
            "estimated_crossing_time_h"
        ),
        "ood_score": row.get(
            "ood_score"
        ),
        "ood_status": row.get(
            "ood_status"
        ),
    }

    risk_synthesis = {
        "failure_risk": row.get(
            "failure_risk"
        ),
        "risk_mode": row.get(
            "risk_mode"
        ),
        "forecast_confidence": row.get(
            "forecast_confidence"
        ),
        "capability_mode": row.get(
            "capability_mode"
        ),
        "policy_reason": row.get(
            "policy_reason_code"
        ),
    }

    codes: list[str] = []

    policy_code = str(
        row.get(
            "policy_reason_code",
            "",
        )
    )

    if policy_code:
        codes.append(
            policy_code
        )

    if bool(
        row.get(
            "predicted_limit_crossing",
            False,
        )
    ):
        codes.append(
            "ACCELERATING_DRIFT_PROJECTED_CROSSING"
        )

    if bool(
        row.get(
            "change_point",
            False,
        )
    ):
        codes.append(
            "TEMPORAL_CHANGE_POINT"
        )

    if abs(
        float(
            row.get(
                "acceleration",
                0.0,
            )
            or 0.0
        )
    ) > 0:
        codes.append(
            "DEGRADATION_ACCELERATION"
        )

    if str(
        row.get(
            "ood_status",
            "",
        )
    ) in {"MODERATE", "SEVERE"}:
        codes.append(
            "NOVEL_FAMILY_INSPECTION"
        )

    if disposition == "PASS":
        codes.append(
            "STABLE_IN_SPEC"
        )

    if disposition == "UNKNOWN":
        codes.append(
            "ENGINEERING_INSPECTION_REQUIRED"
        )

    reason_codes = tuple(
        dict.fromkeys(
            code
            for code in codes
            if code
        )
    )

    narrative = (
        f"{disposition}: {parameter} on "
        f"component {component_id}. "
        f"Observed value "
        f"{_number(row.get('value'))} "
        f"{row.get('unit', '')} at "
        f"{_number(row.get('time_h'), 2)}h. "
        f"Reference level="
        f"{row.get('reference_level', 'INSUFFICIENT')}; "
        f"anomaly="
        f"{_number(row.get('anomaly_score'))}. "
        f"Forecast mode="
        f"{row.get('forecast_mode', 'INSUFFICIENT')}; "
        f"failure risk="
        f"{_number(row.get('failure_risk'))}; "
        f"OOD="
        f"{row.get('ood_status', 'UNKNOWN')}."
    )

    if bool(
        row.get(
            "predicted_limit_crossing",
            False,
        )
    ):
        narrative += (
            " Predicted engineering-limit "
            "crossing near "
            f"{_number(row.get('estimated_crossing_time_h'), 1)}h."
        )

    narrative += (
        " Reason codes: "
        + (
            ", ".join(reason_codes)
            if reason_codes
            else "NONE"
        )
        + "."
    )

    return ExplanationTrace(
        component_id=component_id,
        parameter=parameter,
        observation=observation,
        inference=inference,
        risk_synthesis=risk_synthesis,
        disposition=disposition,
        reason_codes=reason_codes,
        narrative=narrative,
    )


def explain_dataframe(
    triage: pd.DataFrame,
) -> list[ExplanationTrace]:
    if triage.empty:
        return []

    return [
        explain_row(row)
        for _, row in triage.iterrows()
    ]


def traces_to_dict(
    traces: list[ExplanationTrace],
) -> list[dict[str, Any]]:
    return [
        trace.to_dict()
        for trace in traces
    ]
