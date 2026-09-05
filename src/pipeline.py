"""Operational screening pipeline orchestration.

Composes data ingestion, universal representation, multi-channel anomaly screening,
horizon-isolated drift forecasting, OOD bounds assessment, safety policy gating,
and hierarchical engineering explanation into a single unified execution pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly import fit_anomaly, load_anomaly_model, score_anomaly
from src.explain import render_report
from src.features import FeatureEngine, validate_features
from src.forecast import fit_forecast, load_forecast_model, predict_forecast
from src.ingest import canonicalize_csv, profile_csv, validate_canonical
from src.safety import assess_screening, fit_ood_profile, load_ood_profile
from src.utils import PROJECT_ROOT, load_yaml, seed_everything


@dataclass(frozen=True)
class PipelineArtifacts:
    anomaly_model: Path
    forecast_model: Path
    ood_profile: Path


@dataclass
class ScreeningRun:
    screening: pd.DataFrame
    explanations: pd.DataFrame
    canonical: pd.DataFrame
    features: pd.DataFrame
    anomaly: pd.DataFrame
    forecast: pd.DataFrame
    ood: dict[str, Any]
    manifest: dict[str, Any]


class PipelineError(RuntimeError):
    pass


def _default_paths() -> PipelineArtifacts:
    return PipelineArtifacts(
        anomaly_model=PROJECT_ROOT / "models" / "anomaly" / "model.joblib",
        forecast_model=PROJECT_ROOT / "models" / "forecast" / "model.joblib",
        ood_profile=PROJECT_ROOT / "models" / "calibration" / "ood_profile.joblib",
    )


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _model_input_from_canonical(canonical: pd.DataFrame, *, as_of_h: float = 24.0) -> pd.DataFrame:
    d = canonical.copy()
    if "time_h" not in d.columns:
        raise PipelineError("Canonical data has no time_h field.")
    if "part_id" not in d.columns or "parameter" not in d.columns:
        raise PipelineError("Canonical input requires part_id and parameter.")

    d["_value"] = pd.to_numeric(d["value"], errors="coerce")
    d["_time"] = pd.to_numeric(d["time_h"], errors="coerce")

    context_keys = [
        c for c in [
            "part_id", "lot_id", "wafer_id", "test_run_id", "component_family",
            "component_type", "parameter", "semantic_type", "physical_quantity",
            "unit", "profile_id", "test_stage", "test_method", "stress_mode",
        ] if c in d.columns and d[c].notna().any()
    ]
    grouped = (
        d.groupby(context_keys + ["_time"], dropna=False, sort=False, observed=True)["_value"]
        .first().unstack("_time").reset_index()
    )
    wide = grouped.copy()
    wide.columns = [str(c) for c in wide.columns]

    rename = {}
    for c in list(wide.columns):
        try:
            t = float(c)
        except (TypeError, ValueError):
            continue
        label = f"value_{int(round(t))}h" if abs(t - round(t)) < 1e-9 else f"value_{t:g}h"
        rename[c] = label
    wide = wide.rename(columns=rename)

    time_cols = []
    for c in wide.columns:
        m = re.fullmatch(r"value_(\d+(?:\.\d+)?)h", str(c))
        if m:
            t = float(m.group(1))
            if t <= float(as_of_h):
                time_cols.append((t, c))
    time_cols.sort()
    if not time_cols:
        raise PipelineError(f"No observations at or before as_of_h={as_of_h}.")

    origin_candidates = pd.DataFrame(
        {t: pd.to_numeric(wide[c], errors="coerce") for t, c in time_cols},
        index=wide.index,
    )
    origin_candidates = origin_candidates.where(origin_candidates.notna(), np.nan)
    wide["forecast_origin_h"] = origin_candidates.apply(
        lambda r: float(r.dropna().index[-1]) if r.notna().any() else np.nan, axis=1
    )
    wide["available_readpoints_h"] = origin_candidates.apply(
        lambda r: ",".join(f"{float(t):g}" for t in r.dropna().index), axis=1
    )

    if "value_0h" not in wide.columns:
        wide["value_0h"] = np.nan
        
    # VARIABLE HORIZON FIX: If exact 24h isn't there, pick latest prior to as_of_h
    if "value_24h" not in wide.columns:
        pre = [(t, c) for t, c in time_cols if 0 < t <= min(24.0, float(as_of_h))]
        if pre:
            latest_t, src = max(pre, key=lambda x: x[0])
            wide["value_24h"] = wide[src]
            wide["adapted_forecast_origin"] = 1
        else:
            wide["value_24h"] = wide["value_0h"]
            wide["adapted_forecast_origin"] = 1
            
    wide["adapted_forecast_origin"] = wide.get("adapted_forecast_origin", 0)
    return wide


def _overlay_source_metadata(source: pd.DataFrame, wide: pd.DataFrame, canonical: pd.DataFrame | None = None) -> pd.DataFrame:
    d = source.copy()
    candidate = [
        "absolute_limit_lower", "absolute_limit_upper",
        "engineering_limit_lower", "engineering_limit_upper",
        "upper_limit", "lower_limit", "spec_upper", "spec_lower",
        "observable_safety_slope", "profile_id", "split",
        "latent_defect_label", "defect_state", "high_but_safe", "measurement_only_anomaly",
        "precursor_strength", "precursor_effect", "failure_mode", "primary_failure_mode",
    ]
    present = [c for c in candidate if c in d.columns and d[c].notna().any()]
    if not present: return wide
    keys = [c for c in ["part_id", "lot_id", "parameter", "component_family", "component_type"] if c in d.columns and c in wide.columns]
    if not keys: return wide
    meta_keys = [k for k in ["part_id", "lot_id", "parameter", "component_family", "component_type"] if k in d.columns]
    meta = d[meta_keys + present].copy()
    
    if canonical is not None and "parameter" in meta.columns and "raw_parameter" in canonical.columns:
        mapping = canonical[["part_id", "lot_id", "raw_parameter", "parameter"]].drop_duplicates()
        mapping = mapping.rename(columns={"raw_parameter": "raw_parameter_key", "parameter": "canonical_parameter"})
        meta = meta.merge(
            mapping,
            left_on=[c for c in ["part_id", "lot_id", "parameter"] if c in meta.columns],
            right_on=["part_id", "lot_id", "raw_parameter_key"],
            how="left",
        )
        meta["parameter"] = meta["canonical_parameter"].fillna(meta["parameter"])
        meta = meta.drop(columns=[c for c in ["raw_parameter_key", "canonical_parameter"] if c in meta.columns])
        
    if "part_type" in d.columns and "component_type" not in d.columns:
        meta["component_type"] = d["part_type"].to_numpy()
        
    for c in present:
        if c not in meta.columns: continue
        if c not in {"split", "defect_state", "high_but_safe", "profile_id", "failure_mode", "primary_failure_mode"}:
            converted = pd.to_numeric(meta[c], errors="coerce")
            if converted.notna().any(): meta[c] = converted
    meta = meta.drop_duplicates(keys, keep="first")
    return wide.merge(meta, on=keys, how="left", suffixes=("", "_source"))


def _ensure_artifacts(
    artifacts: PipelineArtifacts,
    reference_anomaly: Path,
    reference_forecast: Path,
    reference_ood: Path,
    as_of_h: float,
    target_horizon: float = 168.0,
) -> None:
    artifacts.anomaly_model.parent.mkdir(parents=True, exist_ok=True)
    artifacts.forecast_model.parent.mkdir(parents=True, exist_ok=True)
    artifacts.ood_profile.parent.mkdir(parents=True, exist_ok=True)

    if not artifacts.anomaly_model.exists(): fit_anomaly(reference_anomaly, artifacts.anomaly_model, as_of_h=as_of_h)
    if not artifacts.forecast_model.exists(): fit_forecast(reference_forecast, artifacts.forecast_model, target_horizon_h=float(target_horizon))
    if not artifacts.ood_profile.exists(): fit_ood_profile(reference_ood, artifacts.ood_profile)


def screen_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    artifacts: PipelineArtifacts | None = None,
    reference_anomaly: str | Path | None = None,
    reference_forecast: str | Path | None = None,
    reference_ood: str | Path | None = None,
    as_of_h: float = 24.0,
    target_horizon: float = 168.0,
    auto_train_missing: bool = False,
    compute_features: bool = True,
    render_explanations: bool = True,
) -> ScreeningRun:
    config = load_yaml(PROJECT_ROOT / "configs" / "models.yaml")
    seed_everything(int(config.get("runtime", {}).get("random_seed", 20260831)))
    artifacts = artifacts or _default_paths()
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)

    raw = _read_csv(input_path)
    source_profile = profile_csv(input_path)
    source_file = str(input_path)

    with tempfile.TemporaryDirectory(prefix="screening_", dir=outdir) as td:
        td_path = Path(td)
        canonical_path = td_path / "canonical.csv"
        canonicalize_csv(input_path, canonical_path)
        canonical = pd.read_csv(canonical_path)
        valid, errors = validate_canonical(canonical)
        if not valid: raise PipelineError("Canonical validation failed: " + "; ".join(errors[:10]))

        wide = _model_input_from_canonical(canonical, as_of_h=as_of_h)
        wide = _overlay_source_metadata(raw, wide, canonical)

        if compute_features:
            engine = FeatureEngine.from_project_config()
            features = engine.build(canonical)
            feature_ok, feature_errors = validate_features(features)
            if not feature_ok: raise PipelineError("Feature validation failed: " + "; ".join(feature_errors[:10]))
        else: features = pd.DataFrame()
            
        canonical_out = outdir / "canonical.csv"; features_out = outdir / "features.csv"
        _write(canonical, canonical_out); _write(features, features_out)

        default_ref_a = Path(reference_anomaly or PROJECT_ROOT / "data" / "processed" / "module_A_dataset.csv")
        default_ref_b = Path(reference_forecast or PROJECT_ROOT / "data" / "processed" / "module_B_drift_full.csv")
        default_ref_o = Path(reference_ood or default_ref_a)
        if auto_train_missing: _ensure_artifacts(artifacts, default_ref_a, default_ref_b, default_ref_o, as_of_h, target_horizon)
            
        missing_artifacts = [str(p) for p in [artifacts.anomaly_model, artifacts.forecast_model, artifacts.ood_profile] if not Path(p).exists()]
        if missing_artifacts: raise PipelineError("Required model artifacts are missing: " + ", ".join(missing_artifacts) + ". Train compatible artifacts or pass auto_train_missing=True.")

        anomaly_model = load_anomaly_model(artifacts.anomaly_model)
        anomaly = score_anomaly(anomaly_model, wide, as_of_h=as_of_h, calibrate=True)
        _write(anomaly, outdir / "anomaly.csv")

        forecast_model = load_forecast_model(artifacts.forecast_model)
        forecast_input = wide.copy()
        for col in getattr(forecast_model, "feature_columns", []):
            if col not in forecast_input.columns:
                if col in getattr(forecast_model, "numeric_columns", []): forecast_input[col] = np.nan
                else: forecast_input[col] = "<UNKNOWN>"
        forecast = predict_forecast(forecast_model, forecast_input, target_horizon=float(target_horizon), forecast_origin_h=None)
        _write(forecast, outdir / "forecast.csv")

        ood_profile = load_ood_profile(artifacts.ood_profile)
        from src.safety import assess_data_ood
        ood_report = assess_data_ood(ood_profile, wide)

        screening = assess_screening(anomaly, forecast, wide, ood_profile)
        _write(screening, outdir / "screening.csv")

        explanations = render_report(screening, outdir / "explanations.csv", features=features if not features.empty else None) if render_explanations else pd.DataFrame()
        if not render_explanations: _write(explanations, outdir / "explanations.csv")

        manifest = {
            "source": source_file, "source_profile": source_profile, "as_of_h": float(as_of_h), "target_horizon_h": float(target_horizon),
            "forecast_origin_policy": "last observed readpoint <= as_of_h",
            "rows": {"raw": int(len(raw)), "canonical": int(len(canonical)), "features": int(len(features)), "anomaly": int(len(anomaly)), "forecast": int(len(forecast)), "screening": int(len(screening))},
            "artifacts": {k: str(v) for k, v in artifacts.__dict__.items()},
            "decisions": screening["decision"].value_counts(dropna=False).to_dict() if "decision" in screening.columns else {},
            "ood": {k: v for k, v in ood_report.items() if k != "parts"},
        }
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        return ScreeningRun(screening=screening, explanations=explanations, canonical=canonical, features=features, anomaly=anomaly, forecast=forecast, ood=ood_report, manifest=manifest)


def screen_dataframe(df: pd.DataFrame, output_dir: str | Path, **kwargs: Any) -> ScreeningRun:
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True); input_path = outdir / "input.csv"; df.to_csv(input_path, index=False)
    return screen_file(input_path, outdir, **kwargs)


def progressive_screen_dataframe(df: pd.DataFrame, output_dir: str | Path, *, origins: tuple[float, ...] = (12.0, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0), target_horizon: float = 168.0, artifacts: PipelineArtifacts | None = None, **kwargs: Any) -> dict[str, Any]:
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True); runs = {}; decision_rows = []
    for origin in origins:
        key = float(origin)
        od = outdir / f"origin_{int(origin) if float(origin).is_integer() else str(origin).replace('.', '_')}h"
        runs[key] = screen_dataframe(df, od, artifacts=artifacts, as_of_h=key, target_horizon=float(target_horizon), compute_features=False, render_explanations=False, **kwargs)
        r = runs[key].screening[[c for c in ["part_id", "decision", "risk_score"] if c in runs[key].screening.columns]].copy()
        r["origin_h"] = key
        decision_rows.append(r)
        
    decisions = pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame(columns=["part_id", "decision", "risk_score", "origin_h"])
    flags = decisions[decisions.decision.astype(str).isin(["REVIEW", "REJECT"])].sort_values(["part_id", "origin_h"])
    first = flags.drop_duplicates("part_id", keep="first")[["part_id", "origin_h", "decision", "risk_score"]].rename(columns={"origin_h": "first_flag_time_h"}) if not flags.empty else pd.DataFrame(columns=["part_id", "first_flag_time_h", "decision", "risk_score"])
    
    parts = pd.DataFrame({"part_id": df["part_id"].astype(str).unique()}) if "part_id" in df.columns else pd.DataFrame()
    lead = parts.merge(first, on="part_id", how="left") if not parts.empty else first.copy()
    lead["lead_time_h"] = float(target_horizon) - pd.to_numeric(lead["first_flag_time_h"], errors="coerce")
    
    decisions.to_csv(outdir / "progressive_decisions.csv", index=False); lead.to_csv(outdir / "progressive_lead_time.csv", index=False)
    return {"runs": runs, "decisions": decisions, "lead_time": lead}


def progressive_screen_file(input_path: str | Path, output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    return progressive_screen_dataframe(pd.read_csv(input_path, low_memory=False), output_dir, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("screen"); s.add_argument("input"); s.add_argument("output_dir"); s.add_argument("--as-of", type=float, default=24.0); s.add_argument("--target-horizon", type=float, default=168.0); s.add_argument("--auto-train-missing", action="store_true"); s.add_argument("--progressive", action="store_true")
    args = parser.parse_args()
    if args.progressive:
        r = progressive_screen_file(args.input, args.output_dir, target_horizon=args.target_horizon, auto_train_missing=args.auto_train_missing)
        print(r["lead_time"].to_json(orient="records", indent=2))
    else:
        run = screen_file(args.input, args.output_dir, as_of_h=args.as_of, target_horizon=args.target_horizon, auto_train_missing=args.auto_train_missing)
        print(run.screening[["part_id", "decision", "risk_score", "confidence", "ood_status"]].to_json(orient="records", indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
