"""Read and assemble persisted pipeline results for the frontend API.

No scientific calculations are performed here. This layer only joins the
existing pipeline artifacts into a frontend-friendly representation.
"""
from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.schemas import (
    BatchSummary,
    ComponentIntelligence,
    ComponentSummary,
    EvidenceChannel,
    ForecastPayload,
    MeasurementPoint,
)
from src.utils import PROJECT_ROOT


REPORT_ENV = "ACS_RESULTS_DIR"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / ("demo_report" if (PROJECT_ROOT / "reports" / "demo_report" / "screening.csv").exists() else "demo_screen")


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _record(row: pd.Series) -> dict[str, Any]:
    return _clean(row.to_dict())


def _parse_obj(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    text = value.strip()
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return {}


def _float(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            if str(value) not in {"", "nan", "None"}:
                return value
    return None


def _decision(value: Any) -> str:
    return str(value).upper() if value is not None else "UNKNOWN"


def _trace(screen_row: pd.Series) -> dict[str, Any]:
    trace = _parse_obj(screen_row.get("trace"))
    return trace if isinstance(trace, dict) else {}


def _row_for_part(df: pd.DataFrame, part_id: str, parameter: str | None = None) -> pd.Series | None:
    if df.empty or "part_id" not in df.columns:
        return None
    rows = df[df["part_id"].astype(str).eq(str(part_id))]
    if rows.empty:
        return None
    if parameter and "parameter" in rows.columns:
        exact = rows[rows["parameter"].astype(str).eq(str(parameter))]
        if not exact.empty:
            return exact.iloc[0]
    return rows.iloc[0]


@dataclass
class ResultsRepository:
    report_dir: Path

    @classmethod
    def from_environment(cls) -> "ResultsRepository":
        raw = os.getenv(REPORT_ENV)
        path = Path(raw).expanduser() if raw else DEFAULT_REPORT_DIR
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return cls(path.resolve())

    def _csv(self, name: str) -> pd.DataFrame:
        path = self.report_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    @property
    def screening(self) -> pd.DataFrame:
        return self._csv("screening.csv")

    @property
    def anomaly(self) -> pd.DataFrame:
        return self._csv("anomaly.csv")

    @property
    def forecast(self) -> pd.DataFrame:
        return self._csv("forecast.csv")

    @property
    def canonical(self) -> pd.DataFrame:
        return self._csv("canonical.csv")

    @property
    def raw(self) -> pd.DataFrame:
        return self._csv("input.csv")

    @property
    def explanations(self) -> pd.DataFrame:
        return self._csv("explanations.csv")

    def available(self) -> bool:
        return (self.report_dir / "screening.csv").exists()

    def source_label(self) -> str:
        if self.report_dir == DEFAULT_REPORT_DIR:
            return "bundled_demo_results"
        return str(self.report_dir)

    def summary(self) -> BatchSummary:
        df = self.screening
        if df.empty or "part_id" not in df.columns:
            return BatchSummary(
                total_components=0,
                safe=0,
                review=0,
                reject=0,
                unknown=0,
                decision_counts={},
                report_source=self.source_label(),
            )
        decisions = df["decision"].map(_decision)
        counts = decisions.value_counts(dropna=False).to_dict()
        total = int(df["part_id"].astype(str).nunique())
        # Persisted screening does not contain ground-truth future-failure labels.
        # Do not present a forecast flag as a detected future defect. Benchmark
        # ground truth remains a separate audit artifact.
        future_detected = None
        return BatchSummary(
            total_components=total,
            safe=int(counts.get("SAFE", 0)),
            review=int(counts.get("REVIEW", 0)),
            reject=int(counts.get("REJECT", 0)),
            unknown=int(counts.get("UNKNOWN", 0)),
            decision_counts={str(k): int(v) for k, v in counts.items()},
            future_defects_detected=future_detected,
            report_source=self.source_label(),
        )

    def components(self) -> list[ComponentSummary]:
        screen = self.screening
        if screen.empty:
            return []
        rows: list[ComponentSummary] = []
        for _, srow in screen.iterrows():
            trace = _trace(srow)
            ma = trace.get("module_a", {}) if isinstance(trace, dict) else {}
            mb = trace.get("module_b", {}) if isinstance(trace, dict) else {}
            failure_mode = _first_non_null(srow.get("failure_mode"), mb.get("failure_mode"))
            rows.append(
                ComponentSummary(
                    part_id=str(srow.get("part_id")),
                    lot_id=_first_non_null(srow.get("lot_id"), ma.get("lot_id"), mb.get("lot_id")),
                    component_family=_first_non_null(srow.get("component_family"), ma.get("component_family"), mb.get("component_family")),
                    component_type=_first_non_null(srow.get("component_type"), srow.get("part_type"), mb.get("component_type")),
                    parameter=_first_non_null(srow.get("parameter"), mb.get("parameter")),
                    unit=_first_non_null(srow.get("unit"), mb.get("unit")),
                    decision=_decision(srow.get("decision")),
                    risk_score=_float(srow.get("risk_score")),
                    confidence=str(srow.get("confidence")) if srow.get("confidence") is not None else None,
                    ood_status=str(srow.get("ood_status")) if srow.get("ood_status") is not None else "UNKNOWN",
                    ood_score=_float(srow.get("ood_score")),
                    anomaly_risk=_float(srow.get("anomaly_risk")),
                    failure_risk=_float(srow.get("failure_risk")),
                    uncertainty_score=_float(srow.get("uncertainty_score")),
                    burnin_hours=_float(mb.get("forecast_origin_h")),
                    failure_mode=str(failure_mode) if failure_mode is not None else None,
                )
            )
        return rows

    def intelligence(self, part_id: str) -> ComponentIntelligence:
        screen = self.screening
        row = _row_for_part(screen, part_id)
        if row is None:
            raise KeyError(part_id)
        trace = _trace(row)
        ma = trace.get("module_a", {}) if isinstance(trace, dict) else {}
        mb = trace.get("module_b", {}) if isinstance(trace, dict) else {}
        ood = trace.get("ood", {}) if isinstance(trace, dict) else {}
        dq = trace.get("data_quality", {}) if isinstance(trace, dict) else {}
        fusion = trace.get("fusion", {}) if isinstance(trace, dict) else {}
        policy = trace.get("policy", {}) if isinstance(trace, dict) else {}

        parameter = _first_non_null(mb.get("parameter"), ma.get("parameter"), row.get("parameter"))
        unit = _first_non_null(mb.get("unit"), row.get("unit"))
        fr = _row_for_part(self.forecast, part_id, parameter)
        er = _row_for_part(self.explanations, part_id)

        explanation = _record(er) if er is not None else {}
        if er is not None:
            for source, target in {
                "facts": "facts",
                "model_findings": "model_findings",
                "policy_reasoning": "policy_reasoning",
                "counterfactuals": "counterfactuals",
                "top_reasons": "top_reasons",
                "pattern_attribution_json": "pattern_attribution",
                "audit_trace_json": "audit_trace",
            }.items():
                if source in er.index:
                    explanation[target] = _parse_obj(er.get(source))

        trajectory: list[MeasurementPoint] = []
        can = self.canonical
        if not can.empty and "part_id" in can.columns:
            rows = can[can["part_id"].astype(str).eq(str(part_id))]
            if parameter and "parameter" in rows.columns:
                exact = rows[rows["parameter"].astype(str).eq(str(parameter))]
                if not exact.empty:
                    rows = exact
            if "time_h" in rows.columns and "value" in rows.columns:
                for _, crow in rows.sort_values("time_h").iterrows():
                    t = _float(crow.get("time_h"))
                    if t is not None:
                        trajectory.append(MeasurementPoint(time_h=t, value=_float(crow.get("value"))))

        model_predictions = {}
        for model_name in ("persistence", "linear", "ridge", "gradient_boosting"):
            if fr is not None:
                model_predictions[model_name] = _float(fr.get(f"prediction_{model_name}"))

        forecast = ForecastPayload(
            horizon_h=_float(fr.get("target_horizon_h")) if fr is not None else 168.0,
            origin_h=_float(fr.get("forecast_origin_h")) if fr is not None else _float(mb.get("forecast_origin_h")),
            selected_model=str(fr.get("selected_forecast_model")) if fr is not None and fr.get("selected_forecast_model") is not None else None,
            prediction=_float(fr.get("prediction_168h")) if fr is not None else None,
            lower=_float(fr.get("prediction_lower")) if fr is not None else None,
            upper=_float(fr.get("prediction_upper")) if fr is not None else None,
            interval_width=_float(fr.get("prediction_interval_width")) if fr is not None else None,
            conformal_half_width=_float(fr.get("conformal_half_width")) if fr is not None else None,
            predicted_limit_exceedance=bool(fr.get("predicted_limit_exceedance", 0)) if fr is not None else False,
            limit_exceedance_probability_proxy=_float(fr.get("limit_exceedance_probability_proxy")) if fr is not None else None,
            safety_slope_excess=_float(fr.get("safety_slope_excess")) if fr is not None else None,
            model_predictions=model_predictions,
        )

        evidence = [
            EvidenceChannel(name="Population deviation", score=_float(ma.get("population", ma.get("population_evidence")))),
            EvidenceChannel(name="Temporal drift", score=_float(ma.get("temporal", ma.get("temporal_evidence")))),
            EvidenceChannel(name="Multivariate novelty", score=_float(ma.get("multivariate", ma.get("multivariate_component_score")))),
            EvidenceChannel(name="Absolute limit", score=_float(ma.get("absolute_limits")), detail="Authoritative engineering limit check."),
            EvidenceChannel(name="OOD / domain novelty", score=_float(ood.get("score")), level=str(ood.get("status")) if ood.get("status") is not None else None),
        ]

        limits: dict[str, float | None] = {}
        raw_row = _row_for_part(self.raw, part_id, parameter)
        for key in ("absolute_limit_lower", "absolute_limit_upper", "engineering_limit_lower", "engineering_limit_upper", "upper_limit", "lower_limit", "spec_upper", "spec_lower"):
            source = _first_non_null(
                raw_row.get(key) if raw_row is not None and key in raw_row.index else None,
                row.get(key) if key in row.index else None,
                mb.get(key),
                ma.get(key),
            )
            if source is not None:
                limits[key] = _float(source)

        component = next((x for x in self.components() if x.part_id == str(part_id)), ComponentSummary(part_id=str(part_id), decision=_decision(row.get("decision"))))
        return ComponentIntelligence(
            component=component,
            current_measurements={
                "value_0h": _float(mb.get("value_0h", row.get("value_0h"))),
                "value_24h": _float(mb.get("value_24h", row.get("value_24h"))),
                "value_asof": _float(mb.get("value_asof", row.get("value_asof"))),
                "parameter": parameter,
                "physical_quantity": _first_non_null(mb.get("physical_quantity"), row.get("physical_quantity")),
                "unit": unit,
            },
            historical_trajectory=trajectory,
            forecast=forecast,
            engineering_limits=limits,
            anomaly_evidence=evidence,
            ood=_clean(ood),
            decision={
                "decision": _decision(row.get("decision")),
                "risk_score": _float(row.get("risk_score")),
                "confidence": row.get("confidence"),
                "evidence_state": row.get("evidence_state"),
                "hard_limit_violation": bool(row.get("hard_limit_violation", False)),
                "near_limit": bool(row.get("near_limit", False)),
                "supporting_evidence_count": int(_float(row.get("supporting_evidence_count")) or 0),
                "reasons": _parse_obj(row.get("reasons")),
                "warnings": _parse_obj(row.get("warnings")),
                "policy": _clean(policy),
                "fusion": _clean(fusion),
                "data_quality": _clean(dq),
            },
            explanation=explanation,
            audit_metadata={
                "report_source": self.source_label(),
                "part_id": str(part_id),
                "trace_present": bool(trace),
                "source_split": _first_non_null(mb.get("split"), row.get("split")),
                "failure_mode": _first_non_null(row.get("failure_mode"), mb.get("failure_mode")),
            },
        )
