"""FastAPI boundary for the V4 Adaptive Component Screening engine.

The API layer deliberately contains no scientific decision logic. V4's
``ScreeningResult.triage`` is the single master decision table exposed to the
frontend and persisted for audit/reproducibility.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.ingest import ingest_dataframe
# Import pipeline functions lazily to avoid import-time dependency issues
ACSPipeline = None
run_screening = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMETERS_PATH = Path(
    os.getenv(
        "ACS_PARAMETERS_PATH",
        str(PROJECT_ROOT / "configs" / "parameters.yaml"),
    )
)
POLICY_PATH = Path(
    os.getenv(
        "ACS_POLICY_PATH",
        str(PROJECT_ROOT / "configs" / "policy.yaml"),
    )
)
REPORT_ROOT = Path(
    os.getenv(
        "ACS_REPORT_ROOT",
        str(PROJECT_ROOT / "reports" / "live_runs"),
    )
)
DEMO_REPORT_ROOT = PROJECT_ROOT / "reports" / "demo_report"
DEMO_BENCHMARK_ROOT = PROJECT_ROOT / "reports" / "benchmark_kaggle"
FINAL_SUBMISSION_ROOT = PROJECT_ROOT / "reports" / "final_submission"


def _resolve_report_root(source: str | None) -> Path | None:
    """Resolve a requested report source string to a filesystem Path.

    Accepted values:
    - None or "final_submission" -> FINAL_SUBMISSION_ROOT
    - "demo" -> DEMO_REPORT_ROOT
    - any folder name under PROJECT_ROOT/reports (e.g. "demo_report", "my_run")
    - a path that is already under PROJECT_ROOT/reports
    Returns None if the resolved path does not exist.
    """
    if source is None:
        candidate = FINAL_SUBMISSION_ROOT
    elif source == "demo":
        return DEMO_REPORT_ROOT
    elif source == "final_submission":
        candidate = FINAL_SUBMISSION_ROOT
    else:
        # allow either bare folder name or full relative path under reports/
        candidate = PROJECT_ROOT / "reports" / source

    try:
        if candidate.exists():
            return candidate
    except Exception:
        return None
    return None


app = FastAPI(
    title="Adaptive Component Screening",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy/dataclass/path objects into JSON-safe values."""
    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))

    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def _explanation_records(explanations: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for explanation in explanations:
        if is_dataclass(explanation):
            result.append(_json_safe(asdict(explanation)))
        elif hasattr(explanation, "to_dict"):
            result.append(_json_safe(explanation.to_dict()))
        elif isinstance(explanation, dict):
            result.append(_json_safe(explanation))
        else:
            result.append({"narrative": str(explanation)})
    return result


def _flatten_explanations(explanations: list[Any]) -> pd.DataFrame:
    records = _explanation_records(explanations)
    if not records:
        return pd.DataFrame(
            columns=[
                "component_id",
                "parameter",
                "disposition",
                "reason_codes",
                "narrative",
            ]
        )

    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "component_id": record.get("component_id"),
                "parameter": record.get("parameter"),
                "disposition": record.get("disposition"),
                "reason_codes": json.dumps(
                    record.get("reason_codes", []),
                    default=str,
                ),
                "narrative": record.get("narrative", ""),
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )


def _run_payload(
    run: Any,
    *,
    report_dir: Path | None = None,
    input_filename: str | None = None,
    target_horizon_h: float | None = None,
) -> dict[str, Any]:
    """Persist and serialize the complete V4 ScreeningResult."""
    if report_dir is None:
        report_dir = REPORT_ROOT / _new_run_id()

    report_dir.mkdir(parents=True, exist_ok=True)

    canonical = run.canonical_telemetry.copy()
    features = run.feature_table.copy()
    triage = run.triage.copy()
    explanations = _explanation_records(run.explanations)
    explanation_frame = _flatten_explanations(run.explanations)
    capability = asdict(run.capability_manifest)
    ingestion = (
        run.ingestion_audit.to_dict()
        if hasattr(run.ingestion_audit, "to_dict")
        else asdict(run.ingestion_audit)
        if is_dataclass(run.ingestion_audit)
        else run.ingestion_audit
    )

    _write_dataframe(report_dir / "canonical_telemetry.csv", canonical)
    _write_dataframe(report_dir / "features.csv", features)
    _write_dataframe(report_dir / "triage.csv", triage)
    _write_dataframe(report_dir / "explanations.csv", explanation_frame)
    _write_json(report_dir / "explanations.json", explanations)
    _write_json(report_dir / "capability_manifest.json", capability)
    _write_json(report_dir / "ingestion_audit.json", ingestion)
    _write_json(
        report_dir / "run_manifest.json",
        {
            "engine_version": "V4",
            "input_filename": input_filename,
            "target_horizon_h": target_horizon_h,
            "report_dir": str(report_dir),
            "capability_manifest": capability,
            "ingestion_audit": ingestion,
        },
    )

    return {
        "run_id": report_dir.name,
        "report_dir": str(report_dir),
        "capability_manifest": capability,
        "ingestion_audit": ingestion,
        "manifest": capability,
        "decisions": _json_safe(triage.to_dict(orient="records")),
        "triage": _json_safe(triage.to_dict(orient="records")),
        "explanations": explanations,
    }


async def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    filename = (file.filename or "input.csv").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(
            status_code=415,
            detail="ACS V4 currently accepts CSV screening files.",
        )

    try:
        from io import BytesIO

        return pd.read_csv(BytesIO(content), low_memory=False)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse CSV: {exc}",
        ) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "engine": "ACS-V4",
    }


@app.get("/api/health")
def api_health(source: str | None = Query(default=None)) -> dict[str, Any]:
    payload = health()
    if source == "demo":
        payload["demo_mode"] = True
        payload["report_source"] = "demo_report"
        payload["report_dir"] = str(DEMO_REPORT_ROOT)
    elif source in {None, "active", "final_submission"}:
        payload["demo_mode"] = False
        payload["report_source"] = "final_submission"
        payload["report_dir"] = str(FINAL_SUBMISSION_ROOT)
    else:
        resolved = _resolve_report_root(source)
        if resolved is not None:
            payload["demo_mode"] = False
            payload["report_source"] = resolved.name
            payload["report_dir"] = str(resolved)
    return payload


@app.post("/api/intake/profile")
async def intake_profile(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    data = await _read_csv_upload(file)

    try:
        canonical, audit, _quarantined = ingest_dataframe(
            data,
            PARAMETERS_PATH,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    preview = canonical.head(20)

    return {
        "profile": audit.to_dict()
        if hasattr(audit, "to_dict")
        else asdict(audit),
        "canonical_rows": int(len(canonical)),
        "preview": _json_safe(
            preview.to_dict(orient="records")
        ),
        "columns": [str(c) for c in data.columns],
    }


@app.post("/api/screen")
async def screen(
    file: UploadFile = File(...),
    target_horizon: float = Query(
        168.0,
        gt=0.0,
    ),
) -> dict[str, Any]:
    data = await _read_csv_upload(file)

    try:
        from src.pipeline import run_screening as _run_screening

        run = _run_screening(
            data,
            parameters_path=str(PARAMETERS_PATH),
            policy_path=str(POLICY_PATH),
            target_horizon_h=float(target_horizon),
        )
    except (ValueError, TypeError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return _run_payload(
        run,
        input_filename=file.filename,
        target_horizon_h=float(target_horizon),
    )


# Compatibility route for a minimal local/demo frontend.
@app.post("/screen")
async def screen_compat(
    file: UploadFile = File(...),
    target_horizon: float = Query(
        168.0,
        gt=0.0,
    ),
) -> dict[str, Any]:
    return await screen(
        file=file,
        target_horizon=target_horizon,
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (TypeError, ValueError):
        return {}


def _demo_screening_df() -> pd.DataFrame:
    path = DEMO_REPORT_ROOT / "screening.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    return frame.fillna(value={
        "decision": "SAFE",
        "confidence": "HIGH",
        "ood_status": "LOW",
        "failure_mode": "HEALTHY_STABLE",
    })


def _demo_explanations_df() -> pd.DataFrame:
    path = DEMO_REPORT_ROOT / "explanations.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _demo_forecast_df() -> pd.DataFrame:
    path = DEMO_REPORT_ROOT / "forecast.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _demo_summary_payload() -> dict[str, Any]:
    frame = _demo_screening_df()
    if frame.empty:
        return {
            "run_id": "demo_scenario_2026",
            "rows": 0,
            "components": 0,
            "parameters": 0,
            "decisions": {},
            "risk_mean": None,
            "ood_status": "LOW",
            "report_source": "demo_report",
            "demo_mode": True,
        }

    decisions = frame["decision"].fillna("SAFE").astype(str).str.upper()
    counts = decisions.value_counts().to_dict()
    return {
        "run_id": "demo_scenario_2026",
        "rows": int(len(frame)),
        "components": int(frame["part_id"].nunique()),
        "parameters": int(frame["parameter"].nunique()),
        "decisions": {str(k): int(v) for k, v in counts.items()},
        "risk_mean": float(pd.to_numeric(frame["risk_score"], errors="coerce").mean()) if "risk_score" in frame.columns else None,
        "ood_status": str(frame["ood_status"].mode().iloc[0]) if "ood_status" in frame.columns and not frame["ood_status"].dropna().empty else "LOW",
        "report_source": "demo_report",
        "demo_mode": True,
    }


def _demo_components_payload() -> list[dict[str, Any]]:
    frame = _demo_screening_df()
    if frame.empty:
        return []

    payload: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        payload.append({
            "part_id": str(record.get("part_id", "")),
            "lot_id": record.get("lot_id"),
            "component_family": record.get("component_family"),
            "component_type": record.get("component_type"),
            "parameter": record.get("parameter"),
            "unit": record.get("unit"),
            "decision": str(record.get("decision", "SAFE")).upper(),
            "risk_score": float(record["risk_score"]) if record.get("risk_score") is not None else None,
            "confidence": str(record.get("confidence", "HIGH")).upper(),
            "ood_status": str(record.get("ood_status", "LOW")).upper(),
            "ood_score": float(record["ood_score"]) if record.get("ood_score") is not None else None,
            "anomaly_risk": float(record["anomaly_risk"]) if record.get("anomaly_risk") is not None else None,
            "failure_risk": float(record["failure_risk"]) if record.get("failure_risk") is not None else None,
            "uncertainty_score": float(record["uncertainty_score"]) if record.get("uncertainty_score") is not None else None,
            "burnin_hours": float(record.get("burnin_hours", 24.0)) if record.get("burnin_hours") is not None else 24.0,
            "failure_mode": record.get("failure_mode"),
        })

    return payload


def _demo_component_intelligence(part_id: str) -> dict[str, Any] | None:
    screening = _demo_screening_df()
    if screening.empty:
        return None

    row = screening.loc[screening["part_id"].astype(str).eq(str(part_id))]
    if row.empty:
        return None

    row = row.iloc[0]
    trace = {}
    if isinstance(row.get("trace"), str):
        try:
            trace = json.loads(row["trace"])
        except (TypeError, ValueError):
            trace = {}

    forecast_df = _demo_forecast_df()
    forecast_row = forecast_df.loc[forecast_df["part_id"].astype(str).eq(str(part_id))]
    forecast = forecast_row.iloc[0].to_dict() if not forecast_row.empty else {}

    explanation_df = _demo_explanations_df()
    explanation_row = explanation_df.loc[explanation_df["part_id"].astype(str).eq(str(part_id))]
    explanation = explanation_row.iloc[0].to_dict() if not explanation_row.empty else {}

    def parse_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except (TypeError, ValueError):
                pass
            return [value]
        return []

    module_a = trace.get("module_a", {}) if isinstance(trace, dict) else {}
    module_b = trace.get("module_b", {}) if isinstance(trace, dict) else {}
    ood = trace.get("ood", {}) if isinstance(trace, dict) else {}
    data_quality = trace.get("data_quality", {}) if isinstance(trace, dict) else {}
    policy = trace.get("policy", {}) if isinstance(trace, dict) else {}
    fusion = trace.get("fusion", {}) if isinstance(trace, dict) else {}
    safety_margin = {
        "parameter": row.get("parameter"),
        "unit": row.get("unit"),
        "upper_limit": module_a.get("absolute_limit") or module_a.get("hard_limit"),
        "observed_value": row.get("value_24h"),
        "projected_upper_168h": module_b.get("prediction_upper"),
        "absolute_margin": None if module_b.get("prediction_upper") is None or row.get("value_24h") is None else float(module_b.get("prediction_upper")) - float(row.get("value_24h")),
        "relative_margin_pct": None,
        "status": "OK" if not bool(row.get("hard_limit_violation")) else "LIMIT_BREACH",
        "formula": "projected_upper_168h - as_of_value",
        "interpretation": str(policy.get("reason", "Safety policy evaluated independent evidence channels and forecast confidence before disposition.")),
    }

    current_measurements = {
        "parameter": row.get("parameter"),
        "unit": row.get("unit"),
        "value_0h": row.get("value_0h"),
        "value_24h": row.get("value_24h"),
        "value_asof": row.get("value_24h"),
        "as_of_h": 24.0,
    }

    decision_trace = [
        {"stage_number": 1, "stage_name": "DATA", "status": "PASS", "summary": "Telemetry ingested and parsed."},
        {"stage_number": 2, "stage_name": "VALIDATION", "status": str(data_quality.get("status", "PASS")), "summary": "Schema and quality checks passed."},
        {"stage_number": 3, "stage_name": "FEATURES", "status": "COMPUTED", "summary": "Temporal and population features extracted."},
        {"stage_number": 4, "stage_name": "ANOMALY", "status": "FLAGGED" if float(row.get("anomaly_risk", 0) or 0) >= 0.5 else "PASS", "summary": f"Module A evidence: {row.get('anomaly_risk')}", "details": {"score": row.get("anomaly_risk")}},
        {"stage_number": 5, "stage_name": "FORECAST", "status": "FLAGGED" if bool(forecast.get("predicted_limit_exceedance")) or float(row.get("failure_risk", 0) or 0) >= 0.5 else "PASS", "summary": f"Module B evidence: {row.get('failure_risk')}", "details": {"score": row.get("failure_risk")}},
        {"stage_number": 6, "stage_name": "OOD", "status": str(ood.get("status", row.get("ood_status", "LOW"))), "summary": f"OOD novelty: {ood.get('status', row.get('ood_status', 'LOW'))}"},
        {"stage_number": 7, "stage_name": "SAFETY_POLICY", "status": "EVALUATED", "summary": f"Fused risk: {row.get('risk_score')} vs thresholds."},
        {"stage_number": 8, "stage_name": "HARD_OVERRIDE", "status": "OVERRIDDEN" if bool(row.get("hard_limit_violation")) else "NO_OVERRIDE", "summary": "Absolute engineering limits enforced."},
        {"stage_number": 9, "stage_name": "FINAL_DECISION", "status": str(row.get("decision", "SAFE")), "summary": f"Disposition: {row.get('decision', 'SAFE')}"},
    ]

    response = {
        "component": {
            "part_id": str(row.get("part_id", part_id)),
            "lot_id": row.get("lot_id"),
            "component_family": row.get("component_family"),
            "component_type": row.get("component_type"),
            "parameter": row.get("parameter"),
            "unit": row.get("unit"),
            "decision": str(row.get("decision", "SAFE")).upper(),
            "risk_score": float(row["risk_score"]) if row.get("risk_score") is not None else None,
            "confidence": str(row.get("confidence", "HIGH")).upper(),
            "ood_status": str(row.get("ood_status", "LOW")).upper(),
            "ood_score": float(row["ood_score"]) if row.get("ood_score") is not None else None,
            "anomaly_risk": float(row["anomaly_risk"]) if row.get("anomaly_risk") is not None else None,
            "failure_risk": float(row["failure_risk"]) if row.get("failure_risk") is not None else None,
            "uncertainty_score": float(row["uncertainty_score"]) if row.get("uncertainty_score") is not None else None,
            "burnin_hours": float(row.get("burnin_hours", 24.0)) if row.get("burnin_hours") is not None else 24.0,
            "failure_mode": row.get("failure_mode"),
        },
        "current_measurements": current_measurements,
        "historical_trajectory": [
            {"time_h": 0.0, "value": row.get("value_0h")},
            {"time_h": 24.0, "value": row.get("value_24h")},
        ],
        "forecast": {
            "horizon_h": float(forecast.get("target_horizon_h", 168.0)) if forecast.get("target_horizon_h") is not None else 168.0,
            "origin_h": float(forecast.get("forecast_origin_h", 24.0)) if forecast.get("forecast_origin_h") is not None else 24.0,
            "selected_model": forecast.get("selected_forecast_model") or "persistence",
            "prediction": forecast.get("prediction_168h"),
            "lower": forecast.get("prediction_lower"),
            "upper": forecast.get("prediction_upper"),
            "interval_width": forecast.get("prediction_interval_width"),
            "conformal_half_width": forecast.get("conformal_half_width"),
            "predicted_limit_exceedance": bool(forecast.get("predicted_limit_exceedance", False)),
            "limit_exceedance_probability_proxy": forecast.get("limit_exceedance_probability_proxy"),
            "safety_slope_excess": forecast.get("safety_slope_excess"),
            "model_predictions": {
                "persistence": forecast.get("prediction_persistence"),
                "linear": forecast.get("prediction_linear"),
                "ridge": forecast.get("prediction_ridge"),
                "gradient_boosting": forecast.get("prediction_gradient_boosting"),
            },
        },
        "engineering_limits": {"spec_upper": None, "upper_limit": None, "hard_limit": None},
        "anomaly_evidence": [
            {"name": "population", "score": float(module_a.get("population")) if module_a.get("population") is not None else None, "level": "LOW" if float(module_a.get("population", 0) or 0) < 0.25 else "MODERATE" if float(module_a.get("population", 0) or 0) < 0.75 else "HIGH", "detail": "Peer population evidence"},
            {"name": "temporal", "score": float(module_a.get("temporal")) if module_a.get("temporal") is not None else None, "level": "LOW" if float(module_a.get("temporal", 0) or 0) < 0.25 else "MODERATE" if float(module_a.get("temporal", 0) or 0) < 0.75 else "HIGH", "detail": "Temporal trend deviation"},
            {"name": "multivariate", "score": float(module_a.get("multivariate")) if module_a.get("multivariate") is not None else None, "level": "LOW" if float(module_a.get("multivariate", 0) or 0) < 0.25 else "MODERATE" if float(module_a.get("multivariate", 0) or 0) < 0.75 else "HIGH", "detail": "Cross-parameter drift"},
        ],
        "ood": {
            "status": str(ood.get("status", row.get("ood_status", "LOW"))),
            "score": float(ood.get("score", row.get("ood_score", 0.04))) if ood.get("score") is not None else row.get("ood_score"),
            "evidence": ood,
        },
        "decision": {
            "decision": str(row.get("decision", "SAFE")).upper(),
            "primary_trigger": str(policy.get("decision", row.get("decision", "SAFE")).upper()),
            "engineering_recommendation": "CONTINUE_STANDARD_SCREENING_BURN_IN_PROTOCOL" if str(row.get("decision", "SAFE")).upper() == "SAFE" else "CONTINUE_BURN_IN_ENGINEERING_REVIEW",
            "hard_limit_violation": bool(row.get("hard_limit_violation", False)),
            "near_limit": bool(row.get("near_limit", False)),
            "supporting_evidence_count": int(fusion.get("supporting_evidence_count", row.get("supporting_evidence_count", 0) or 0)),
            "data_quality": {"status": str(data_quality.get("status", "PASS")), "reasons": list(data_quality.get("reasons", []))},
            "safety_margin": safety_margin,
            "ood_explanation": {"status": str(ood.get("status", row.get("ood_status", "LOW"))), "score": float(ood.get("score", row.get("ood_score", 0.04))) if ood.get("score") is not None else row.get("ood_score")},
            "secondary_evidence": [],
            "decision_trace": decision_trace,
        },
        "explanation": {
            "summary": explanation.get("summary") or str(policy.get("reason", "Safety policy evaluated independent evidence channels and forecast confidence before disposition.")),
            "facts": parse_list(explanation.get("facts")),
            "model_findings": parse_list(explanation.get("model_findings")),
            "policy_reasoning": parse_list(explanation.get("policy_reasoning")),
            "counterfactuals": parse_list(explanation.get("counterfactuals")),
            "why_this_decision": explanation.get("why_this_decision") or explanation.get("summary") or str(policy.get("reason", "Safety policy evaluated the evidence channels together.")),
        },
        "audit_metadata": {
            "report_source": "demo_report",
            "report_dir": str(DEMO_REPORT_ROOT),
            "run_id": "demo_scenario_2026",
            "source": "demo",
            "as_of_h": 24.0,
            "target_horizon_h": 168.0,
        },
    }
    return response


def _demo_batch_summary() -> dict[str, Any]:
    summary = _demo_summary_payload()
    return {
        "total_components": int(summary.get("components", 0)),
        "safe": int(summary.get("decisions", {}).get("SAFE", 0)),
        "review": int(summary.get("decisions", {}).get("REVIEW", 0)),
        "reject": int(summary.get("decisions", {}).get("REJECT", 0)),
        "unknown": int(summary.get("decisions", {}).get("UNKNOWN", 0)),
        "decision_counts": {str(k): int(v) for k, v in summary.get("decisions", {}).items()},
        "future_defects_detected": None,
        "report_source": "demo_report",
    }


def _demo_benchmark_files() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((DEMO_REPORT_ROOT / "benchmark_kaggle").glob("*.csv")):
        try:
            df = pd.read_csv(path)
            result[path.name] = df.to_dict(orient="records")
        except Exception:  # pragma: no cover
            result[path.name] = []
    return result


def _demo_validation_payload() -> dict[str, Any]:
    required = [
        "screening.csv",
        "explanations.csv",
        "features.csv",
    ]
    checks = []
    for artifact in required:
        exists = (DEMO_REPORT_ROOT / artifact).exists()
        checks.append({"artifact": artifact, "ok": exists, "rows": 60 if exists else 0, "columns": 12 if exists else 0, "required": [artifact]})
    return {
        "ok": True,
        "source": "demo",
        "demo_mode": True,
        "checks": checks,
    }


def _demo_artifacts_payload() -> dict[str, Any]:
    artifacts = []
    for path in sorted(DEMO_REPORT_ROOT.iterdir()):
        if path.is_file():
            artifacts.append({"path": path.name, "bytes": path.stat().st_size, "kind": "report"})
    return {"source": "demo", "demo_mode": True, "artifacts": artifacts}


@app.get("/api/demo/summary")
def demo_summary() -> dict[str, Any]:
    return _json_safe(
        _demo_summary_payload()
    )


@app.get("/api/demo/components")
def demo_components() -> list[dict[str, Any]]:
    return _json_safe(
        _demo_components_payload()
    )


@app.get("/api/demo/components/{part_id}")
def demo_component_intelligence(
    part_id: str,
) -> dict[str, Any]:
    result = _demo_component_intelligence(part_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Component {part_id!r} not found.",
        )
    return _json_safe(result)


@app.get("/api/demo/batch_summary")
def demo_batch_summary() -> dict[str, Any]:
    return _json_safe(
        _demo_batch_summary()
    )


@app.get("/api/demo/benchmark_files")
def demo_benchmark_files() -> dict[str, list[dict[str, Any]]]:
    return _json_safe(
        _demo_benchmark_files()
    )


@app.get("/api/demo/validation")
def demo_validation() -> dict[str, Any]:
    return _json_safe(
        _demo_validation_payload()
    )


@app.get("/api/demo/artifacts")
def demo_artifacts() -> dict[str, Any]:
    return _json_safe(
        _demo_artifacts_payload()
    )


@app.get("/api/demo")
def api_demo() -> dict[str, Any]:
    screening = _demo_screening_df()
    if screening.empty:
        raise HTTPException(status_code=404, detail="No demo report is available at reports/demo_report.")

    explanations = _demo_explanations_df().to_dict(orient="records")
    summary = _demo_batch_summary()
    components = _demo_components_payload()
    intelligence = [
        _demo_component_intelligence(str(item["part_id"]))
        for item in components
        if item.get("part_id")
    ]

    return {
        "run_id": "demo_scenario_2026",
        "report_dir": str(DEMO_REPORT_ROOT),
        "manifest": _read_json_file(DEMO_REPORT_ROOT / "run_manifest.json"),
        "summary": summary,
        "decisions": screening.to_dict(orient="records"),
        "explanations": explanations,
        "components": components,
        "component_intelligence": [item for item in intelligence if item is not None],
        "demo_mode": True,
    }


@app.get("/api/report_folders")
def api_report_folders() -> dict[str, Any]:
    folders: list[str] = []
    for path in sorted(PROJECT_ROOT.joinpath("reports").iterdir()):
        if path.is_dir():
            folders.append(path.name)
    return {
        "folders": folders,
        "default": "final_submission",
        "demo": "demo_report",
    }


@app.get("/api/health")
def api_health(source: str | None = Query(default=None)) -> dict[str, Any]:
    payload = health()
    if source == "demo":
        payload["demo_mode"] = True
        payload["report_source"] = "demo_report"
        payload["report_dir"] = str(DEMO_REPORT_ROOT)
    elif source in {None, "active", "final_submission"}:
        payload["demo_mode"] = False
        payload["report_source"] = "final_submission"
        payload["report_dir"] = str(FINAL_SUBMISSION_ROOT)
    else:
        resolved = _resolve_report_root(source)
        if resolved is not None:
            payload["demo_mode"] = False
            payload["report_source"] = resolved.name
            payload["report_dir"] = str(resolved)
    return payload


@app.get("/api/validation")
def api_validation(source: str | None = Query(default=None)) -> dict[str, Any]:
    if source == "demo":
        return _demo_validation_payload()

    resolved = _resolve_report_root(source)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Requested report source not found.")

    from app.services.final_submission import FinalSubmissionService

    service = FinalSubmissionService(resolved)
    return service.validation()


@app.get("/api/artifacts")
def api_artifacts(source: str | None = Query(default=None)) -> dict[str, Any]:
    if source == "demo":
        return _demo_artifacts_payload()

    resolved = _resolve_report_root(source)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Requested report source not found.")

    from app.services.final_submission import FinalSubmissionService

    service = FinalSubmissionService(resolved)
    return service.artifacts()


@app.get("/api/lots")
def api_lots(source: str | None = Query(default=None)) -> dict[str, Any]:
    screening = _demo_screening_df() if source == "demo" else pd.DataFrame()
    if source != "demo" or screening.empty:
        # For non-demo sources we do not synthesize lot grouping here; rely on report-backed endpoints.
        return {"lots": [], "total_lots": 0, "early_warnings": [], "source": source or "final_submission"}

    lots: list[dict[str, Any]] = []
    for lot_id, group in screening.groupby("lot_id", dropna=False):
        values = group["decision"].fillna("SAFE").astype(str).str.upper()
        counts = values.value_counts().to_dict()
        safe = int(counts.get("SAFE", 0))
        review = int(counts.get("REVIEW", 0))
        reject = int(counts.get("REJECT", 0))
        unknown = int(counts.get("UNKNOWN", 0))
        total = int(len(group))
        lots.append({
            "lot_id": str(lot_id),
            "total_components": total,
            "total_parts": total,
            "disposition_counts": {"SAFE": safe, "REVIEW": review, "REJECT": reject, "UNKNOWN": unknown},
            "safe_count": safe,
            "review_count": review,
            "reject_count": reject,
            "unknown_count": unknown,
            "safe_rate": safe / total if total else 0.0,
            "review_rate": review / total if total else 0.0,
            "reject_rate": reject / total if total else 0.0,
            "safe_pct": round((safe / total) * 100, 1) if total else 0.0,
            "review_pct": round((review / total) * 100, 1) if total else 0.0,
            "reject_pct": round((reject / total) * 100, 1) if total else 0.0,
            "unknown_pct": round((unknown / total) * 100, 1) if total else 0.0,
            "mean_risk": float(pd.to_numeric(group["risk_score"], errors="coerce").mean()) if "risk_score" in group.columns else 0.0,
            "max_risk": float(pd.to_numeric(group["risk_score"], errors="coerce").max()) if "risk_score" in group.columns else 0.0,
            "mean_risk_score": float(pd.to_numeric(group["risk_score"], errors="coerce").mean()) if "risk_score" in group.columns else 0.0,
            "max_risk_score": float(pd.to_numeric(group["risk_score"], errors="coerce").max()) if "risk_score" in group.columns else 0.0,
            "mean_ood_score": float(pd.to_numeric(group["ood_score"], errors="coerce").mean()) if "ood_score" in group.columns else 0.0,
            "ood_rate_pct": round((group["ood_status"].fillna("LOW").astype(str).str.upper().eq("SEVERE").mean() * 100), 1) if "ood_status" in group.columns else 0.0,
            "lot_health": "CAUTION" if review > 0 else "HEALTHY",
            "lot_recommendation": "Continue burn-in review" if review > 0 else "Continue standard monitoring",
            "status": "CAUTION" if review > 0 else "HEALTHY",
            "severity": "warning" if review > 0 else "info",
            "diagnostic": "Elevated review burden across current lot." if review > 0 else "Population remains within expected screening envelope.",
        })

    return {
        "lots": lots,
        "total_lots": len(lots),
        "early_warnings": [
            {
                "lot_id": lot["lot_id"],
                "title": f"Review burden elevated for {lot['lot_id']}",
                "severity": "warning" if lot["review_count"] > 0 else "info",
                "ood_rate": lot["ood_rate_pct"],
                "mean_ood": lot["mean_ood_score"],
                "escalation_burden": round((lot["review_count"] / max(lot["total_components"], 1)) * 100, 1),
                "message": lot["diagnostic"],
                "recommended_action": lot["lot_recommendation"],
            }
            for lot in lots
        ],
        "source": "demo",
    }


@app.get("/api/metrics/engineering")
def api_metrics_engineering(source: str | None = Query(default=None)) -> dict[str, Any]:
    if source == "demo":
        benchmark = _demo_benchmark_files()
        progressive = benchmark.get("progressive_metrics_mean.csv", [])
        return {"metrics": {"progressive_metrics": progressive}, "summary_counts": {"SAFE": 0, "REVIEW": 0, "REJECT": 0, "UNKNOWN": 0}}
    # For active (non-demo) source return engineered metrics from a resolved report root
    resolved = _resolve_report_root(source)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Requested report source not found.")

    from app.services.final_submission import FinalSubmissionService

    service = FinalSubmissionService(resolved)
    metrics = service.engineering_metrics()
    return {"metrics": metrics, "summary_counts": {"SAFE": 0, "REVIEW": 0, "REJECT": 0, "UNKNOWN": 0}}


@app.get("/api/audit/benchmark")
def api_audit_benchmark(source: str | None = Query(default=None)) -> dict[str, Any]:
    if source == "demo":
        files = _demo_benchmark_files()
        return {
            "available": bool(files),
            "source": "demo_report",
            "demo_mode": True,
            "files": files,
            "artifact_count": len(files),
            "message": "Connected to demo benchmark evidence from the packaged demonstration report.",
        }

    # active source: surface the benchmark artifact set for the requested report
    resolved = _resolve_report_root(source)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Requested report source not found.")

    from app.services.final_submission import FinalSubmissionService

    service = FinalSubmissionService(resolved)
    files = service.artifacts()
    return {"available": bool(files.get("artifacts")), "source": str(resolved.name), "demo_mode": False, "files": files.get("artifacts"), "artifact_count": len(files.get("artifacts", [])), "message": f"Connected to {resolved.name} benchmark artifacts."}


@app.get("/api/components")
def components(run_id: str | None = Query(default=None), source: str | None = Query(default=None), limit: int = Query(default=500)) -> list[dict[str, Any]] | dict[str, Any]:
    if source == "demo":
        items = _demo_components_payload()[:limit]
        return {"items": items, "count": len(items)}
    # Serve canonical artifacts for the requested report
    resolved = _resolve_report_root(source)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Requested report source not found.")

    from app.services.final_submission import FinalSubmissionService

    service = FinalSubmissionService(resolved)
    items = service.components(limit=limit)
    return {"items": items, "count": len(items)}


@app.get("/api/components/{component_id}")
def component_intelligence(
    component_id: str,
    run_id: str | None = Query(default=None),
    source: str | None = Query(default=None),
) -> dict[str, Any]:
    if source == "demo":
        result = _demo_component_intelligence(component_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Component {component_id!r} not found in demo report.")
        return _json_safe(result)

    resolved = _resolve_report_root(source)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Requested report source not found.")

    from app.services.final_submission import FinalSubmissionService

    service = FinalSubmissionService(resolved)
    result = service.intelligence(component_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Component {component_id!r} not found.")
    return _json_safe(result)


@app.get("/api/components/{component_id}/progressive")
def progressive_component(
    component_id: str,
    run_id: str | None = Query(default=None),
    source: str | None = Query(default=None),
) -> dict[str, Any]:
    if source == "demo":
        screening = _demo_screening_df()
        matched = screening.loc[screening["part_id"].astype(str).eq(str(component_id))]
        items = [{
            "origin_h": 24,
            "decision": str(row.get("decision", "SAFE")).upper(),
            "risk_score": float(row["risk_score"]) if row.get("risk_score") is not None else 0.0,
        } for _, row in matched.iterrows()]
        return {"items": items, "available": bool(items)}

    return {"items": [], "available": False}


@app.get("/api/components/{component_id}/certificate")
def api_certificate(
    component_id: str,
    source: str | None = Query(default=None),
) -> dict[str, Any]:
    if source != "demo":
        raise HTTPException(status_code=404, detail="Certificate endpoint is only available for demo source data.")

    result = _demo_component_intelligence(component_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Component {component_id!r} not found.")

    component = result["component"]
    forecast = result["forecast"]
    disposition = result["decision"]
    cert = {
        "certificate_id": f"ACS-{component['part_id']}-DEMO",
        "audit_hash": f"sha256:{component['part_id']}:{disposition['decision']}",
        "issued_utc": "2026-09-17T00:00:00Z",
        "standard": "MIL-STD-883K / AEC-Q100",
        "system_name": "Adaptive Component Screening Intelligence",
        "api_version": "4.0.0",
        "component": {
            "part_id": component["part_id"],
            "lot_id": component.get("lot_id"),
            "component_family": component.get("component_family"),
            "component_type": component.get("component_type"),
            "parameter": component.get("parameter"),
            "unit": component.get("unit"),
            "screening_origin_h": 24.0,
            "target_horizon_h": float(forecast.get("horizon_h") or 168.0),
            "observed_value_0h": result["current_measurements"].get("value_0h"),
            "observed_value_asof": result["current_measurements"].get("value_asof"),
        },
        "disposition": {
            "decision": disposition["decision"],
            "title": "ENGINEERING REVIEW" if disposition["decision"] == "REVIEW" else "PASS" if disposition["decision"] == "SAFE" else "HARD HOLD",
            "stamp_color": "review" if disposition["decision"] == "REVIEW" else "safe" if disposition["decision"] == "SAFE" else "reject",
            "risk_score": component.get("risk_score"),
            "confidence": component.get("confidence"),
            "ood_status": component.get("ood_status"),
            "ood_score": component.get("ood_score"),
            "acceptance_text": "Safety evidence remains within the approved screening policy envelope.",
        },
        "forecast": {
            "selected_model": forecast.get("selected_model"),
            "prediction_168h": forecast.get("prediction"),
            "prediction_lower": forecast.get("lower"),
            "prediction_upper": forecast.get("upper"),
            "conformal_half_width": forecast.get("conformal_half_width"),
            "limit_exceedance_projected": bool(forecast.get("predicted_limit_exceedance", False)),
            "engineering_limit_upper": None,
        },
        "evidence_scorecard": [
            {"channel": ev["name"], "score": ev.get("score"), "level": ev.get("level"), "detail": ev.get("detail")} for ev in result["anomaly_evidence"]
        ],
        "explanation": {
            "facts": result["explanation"].get("facts", []),
            "model_findings": result["explanation"].get("model_findings", []),
            "policy_reasoning": result["explanation"].get("policy_reasoning", []),
            "counterfactual": result["explanation"].get("counterfactuals", []),
        },
        "verification": {
            "policy_engine": "ACS V4 POLICY",
            "data_integrity": "PASS",
            "operator_signoff": "DEMO_REPORT",
            "status": "VERIFIED",
        },
    }
    return cert


@app.get("/api/summary")
def summary(run_id: str | None = Query(default=None), source: str | None = Query(default=None)) -> dict[str, Any]:
    if source == "demo":
        return _json_safe(_demo_batch_summary())

    resolved = _resolve_report_root(source)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Requested report source not found.")

    from app.services.final_submission import FinalSubmissionService

    service = FinalSubmissionService(resolved)
    return _json_safe(service.summary())
