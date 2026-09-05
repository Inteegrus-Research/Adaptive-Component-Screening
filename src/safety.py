"""Safety intelligence and policy engine.

Pruned clean of legacy duplicate function blocks. Implements calibrated OODProfileV4
with the 10% Physical, 20% Contextual, and 70% Population evidence schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.utils import (
    PROJECT_ROOT,
    load_yaml,
    seed_everything,
    stable_config_hash as _stable_hash,
    write_artifact_manifest,
    validate_artifact_compatibility,
)

DECISIONS = ("SAFE", "REVIEW", "REJECT", "UNKNOWN")


@dataclass
class OODProfileV4:
    profile_version: str
    config_hash: str
    expected_columns: list[str]
    known_families: list[str]
    known_parameters: list[str]
    known_semantic_types: list[str]
    known_test_methods: list[str]
    known_stress_modes: list[str]
    reference_rows: int
    reference_parts: int
    max_reference_rows: int
    conditional_groups: dict[str, dict[str, dict[str, float]]]
    trajectory_groups: dict[str, dict[str, dict[str, float]]]
    moderate_threshold: float = 0.40
    severe_threshold: float = 0.75


# Backward-compatible aliases
OODProfile = OODProfileV4
OODProfileV3 = OODProfileV4


@dataclass
class ScreeningAssessment:
    part_id: str
    decision: str
    risk_score: float
    confidence: str
    failure_risk: float
    anomaly_risk: float
    ood_score: float
    ood_status: str
    uncertainty_score: float
    data_quality_score: float
    evidence_state: str
    hard_limit_violation: bool
    near_limit: bool
    supporting_evidence_count: int
    high_but_safe: bool = False
    measurement_only_anomaly: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


def _read_any(path: str | Path, table: str | None = None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(p) as con:
            if table is not None:
                return pd.read_sql_query(f"SELECT * FROM {table}", con)
            for candidate in ("module_A_dataset", "module_B_train", "component_parameter_state", "measurements"):
                try:
                    return pd.read_sql_query(f"SELECT * FROM {candidate}", con)
                except Exception:
                    continue
        raise ValueError("No supported table found in database")
    raise ValueError(f"Unsupported input format: {p.suffix}")


def _config_hash(cfg: Mapping[str, Any]) -> str:
    return _stable_hash(cfg)


def _canonical_category(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="object")
    return df[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _aggregate_unique(values: pd.Series) -> set[str]:
    return {str(v) for v in values.dropna().astype(str).unique()}


def _canonical_parameter_series(s: pd.Series) -> pd.Series:
    try:
        from src.ingest import ParameterOntology
        ontology = ParameterOntology(load_yaml(PROJECT_ROOT / "configs" / "parameters.yaml"))
        def one(x):
            if pd.isna(x):
                return x
            spec, _, _ = ontology.resolve(x)
            return spec.name if spec is not None else str(x).strip()
        return s.map(one)
    except Exception:
        return s.map(lambda x: x if pd.isna(x) else str(x).strip())


def _quality_from_input(input_df: pd.DataFrame, policy: Mapping[str, Any]) -> dict[str, Any]:
    n = len(input_df)
    if n == 0:
        return {"status": "FAIL", "score": 0.0, "reasons": ["No observations supplied."], "missing_fraction": 1.0}
    parts = int(input_df["part_id"].nunique()) if "part_id" in input_df.columns else 0
    missing_cells = float(input_df.isna().mean().mean())
    duplicate_fraction = 0.0
    if "observation_id" in input_df.columns:
        duplicate_fraction = float(input_df["observation_id"].duplicated().mean()) if n else 0.0
    req = policy.get("data_quality", {})
    reasons: list[str] = []
    pass_flag = True
    if req.get("require_part_identifier", True) and "part_id" not in input_df.columns:
        pass_flag = False
        reasons.append("Part identifier is missing.")
    if req.get("require_parameter_identifier", True) and "parameter" not in input_df.columns:
        pass_flag = False
        reasons.append("Parameter identifier is missing.")
    if missing_cells > float(req.get("maximum_missing_fraction", 0.40)):
        pass_flag = False
        reasons.append(f"Overall missing-cell fraction {missing_cells:.3f} exceeds policy limit.")
    if duplicate_fraction > float(req.get("maximum_duplicate_fraction", 0.01)):
        pass_flag = False
        reasons.append(f"Duplicate observation fraction {duplicate_fraction:.3f} exceeds policy limit.")
    if parts < int(req.get("minimum_parts", 1)):
        pass_flag = False
        reasons.append("Insufficient unique parts.")
    score = max(0.0, 1.0 - missing_cells / max(float(req.get("maximum_missing_fraction", 0.40)), 1e-9))
    if duplicate_fraction > 0:
        score *= max(0.0, 1.0 - duplicate_fraction / max(float(req.get("maximum_duplicate_fraction", 0.01)), 1e-9))
    return {
        "status": "PASS" if pass_flag else "FAIL",
        "score": float(np.clip(score, 0, 1)),
        "reasons": reasons,
        "rows": n,
        "parts": parts,
        "missing_fraction": missing_cells,
        "duplicate_fraction": duplicate_fraction,
    }


def _part_quality(input_df: pd.DataFrame, part_id: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    if "part_id" not in input_df.columns:
        return {"status": "FAIL", "score": 0.0, "missing_fraction": 1.0, "reasons": ["part_id unavailable"]}
    d = input_df[input_df["part_id"].astype(str).eq(str(part_id))].copy()
    if d.empty:
        return {"status": "FAIL", "score": 0.0, "missing_fraction": 1.0, "reasons": ["No observations for part"]}
    q = _quality_from_input(d, policy)
    if "quality_flag" in d.columns:
        bad = d["quality_flag"].astype(str).str.upper().isin({"ERROR", "INVALID", "BAD"})
        q["bad_quality_fraction"] = float(bad.mean())
        if bad.any():
            q["status"] = "FAIL"
            q["score"] *= float(1.0 - bad.mean())
            q["reasons"].append(f"{int(bad.sum())} observation(s) carry bad quality flags.")
    if "measurement_status" in d.columns:
        invalid = d["measurement_status"].astype(str).str.upper().isin({"ERROR", "INVALID"})
        if invalid.any():
            q["status"] = "FAIL"
            q["score"] *= float(1.0 - invalid.mean())
            q["reasons"].append(f"{int(invalid.sum())} invalid measurement-status row(s).")
    return q


def _v3_key(family: Any, parameter: Any) -> str:
    f = "<MISSING>" if family is None or (isinstance(family, float) and pd.isna(family)) else str(family)
    p = "<MISSING>" if parameter is None or (isinstance(parameter, float) and pd.isna(parameter)) else str(parameter)
    return f"{f}||{p}"


def _v3_robust_stats(s: pd.Series) -> dict[str, float] | None:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if x.size < 8:
        return None
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    q01, q99 = np.quantile(x, [0.01, 0.99])
    return {
        "median": med,
        "scale": max(1.4826 * mad, float((q99 - q01) / 5.15), 1e-12),
        "q01": float(q01), "q99": float(q99), "n": float(x.size),
    }


def _v3_from_wide(d: pd.DataFrame) -> pd.DataFrame:
    required = {"part_id", "parameter"}
    if not required.issubset(d.columns):
        return d.copy()
    time_cols = []
    import re
    for c in d.columns:
        m = re.search(r"(?:value|val)[_@]?(\d+(?:\.\d+)?)h$", str(c), flags=re.I)
        if m:
            time_cols.append((c, float(m.group(1))))
    if not time_cols:
        return d.copy()
    id_cols = [c for c in d.columns if c not in {c for c, _ in time_cols}]
    rows = []
    for c, t in time_cols:
        tmp = d[id_cols].copy()
        tmp["time_h"] = t
        tmp["value_asof"] = pd.to_numeric(d[c], errors="coerce")
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def _v3_prepare(d: pd.DataFrame, as_of_h: float | None = None) -> pd.DataFrame:
    x = _v3_from_wide(d)
    if "part_id" not in x.columns:
        x["part_id"] = "<UNKNOWN>"
    if "parameter" not in x.columns:
        x["parameter"] = "<UNKNOWN>"
    x["part_id"] = x["part_id"].astype(str)
    x["parameter"] = _canonical_parameter_series(x["parameter"])
    if "component_family" not in x.columns:
        x["component_family"] = "<MISSING>"
    if "time_h" not in x.columns:
        x["time_h"] = np.nan
    x["time_h"] = pd.to_numeric(x["time_h"], errors="coerce")
    x["value_num"] = pd.to_numeric(x.get("value_asof", x.get("value", np.nan)), errors="coerce")
    if as_of_h is not None and x["time_h"].notna().any():
        x = x[x["time_h"] <= float(as_of_h)].copy()
    return x


def _ood_v4_status(score: float, profile: OODProfileV4) -> str:
    if score >= profile.severe_threshold:
        return "SEVERE"
    if score >= profile.moderate_threshold:
        return "MODERATE"
    return "LOW"


def _ood_v4_score_part(profile: OODProfileV4, part: pd.DataFrame, global_schema: set[str]) -> dict[str, Any]:
    reasons = []
    c = {}
    family = str(part["component_family"].dropna().iloc[0]) if "component_family" in part.columns and part["component_family"].notna().any() else "<MISSING>"
    params = sorted({str(v) for v in part["parameter"].dropna().unique()}) if "parameter" in part.columns else []
    fs = "MISSING" if family == "<MISSING>" else ("KNOWN" if family in profile.known_families else "NOVEL")
    c["family_status"] = fs
    c["family_novelty"] = float(fs == "NOVEL")
    if fs == "NOVEL":
        reasons.append(f"Component family '{family}' is outside the reference domain.")
    unknown = [p for p in params if p not in profile.known_parameters]
    c["parameter_status"] = "MISSING" if not params else ("NOVEL" if unknown else "KNOWN")
    c["parameter_novelty"] = float(bool(unknown))
    if unknown:
        reasons.append("Parameter semantics are unseen: " + ", ".join(unknown[:5]))
    for col, known, key in [("test_method", profile.known_test_methods, "test_method"), ("stress_mode", profile.known_stress_modes, "stress")]:
        if col not in part.columns or not part[col].notna().any():
            c[f"{key}_status"] = "MISSING"
            c[f"{key}_novelty"] = 0.0
            continue
        vals = {str(v) for v in part[col].dropna().unique()}
        if not known:
            c[f"{key}_status"] = "UNOBSERVED_IN_REFERENCE"
            c[f"{key}_novelty"] = 0.0
        else:
            nov = vals - set(known)
            c[f"{key}_status"] = "NOVEL" if nov else "KNOWN"
            c[f"{key}_novelty"] = float(bool(nov))
            if nov:
                reasons.append(f"{col} values are outside the reference domain: {', '.join(sorted(nov)[:5])}")
    missing = sorted(set(profile.expected_columns) - global_schema)
    c["schema_missing_fraction"] = len(missing) / max(len(profile.expected_columns), 1)

    x = _v3_prepare(part)
    vals = []
    slopes = []
    for (fam, param), g in x.groupby(["component_family", "parameter"], dropna=False, sort=True):
        key = _v3_key(fam, param)
        pkey = _v3_key("*", param)
        ref = profile.conditional_groups.get(key) or profile.conditional_groups.get(pkey)
        if ref and ref.get("value"):
            st = ref["value"]
            arr = pd.to_numeric(g["value_num"], errors="coerce").dropna().to_numpy(float)
            if arr.size:
                z = np.abs((arr - st["median"]) / max(st["scale"], 1e-12))
                tail = float(np.mean((arr < st["q01"]) | (arr > st["q99"])))
                # Tight z-score scale factor (/2.0)
                vals.append(float(np.clip(np.median(z) / 2.0 + 0.4 * tail, 0, 1)))
                if vals[-1] >= 0.5:
                    reasons.append(f"{param}: measurement distribution shifted")
        tr = profile.trajectory_groups.get(key) or profile.trajectory_groups.get(pkey)
        if tr and tr.get("slope"):
            for _, pg in g.groupby("part_id", sort=False):
                pg = pg.dropna(subset=["time_h", "value_num"]).sort_values("time_h")
                if len(pg) >= 2:
                    dt = float(pg["time_h"].iloc[-1] - pg["time_h"].iloc[0])
                    if dt > 0:
                        st = tr["slope"]
                        sl = float((pg["value_num"].iloc[-1] - pg["value_num"].iloc[0]) / dt)
                        slopes.append(float(np.clip(abs(sl - st["median"]) / max(2.0 * st["scale"], 1e-12), 0, 1)))
    c["measurement_shift"] = float(np.mean(vals)) if vals else 0.0
    c["trajectory_shift"] = float(np.mean(slopes)) if slopes else 0.0
    
    physical = max(c["family_novelty"], c["parameter_novelty"])
    context = max(c.get("test_method_novelty", 0.0), c.get("stress_novelty", 0.0))
    population = max(c["measurement_shift"], c["trajectory_shift"])
    
    # Strictly calibrated weighting: 10% Physical, 20% Contextual, 70% Population
    score = float(np.clip(0.10 * physical + 0.20 * context + 0.70 * population, 0, 1))
    status = "SEVERE" if physical >= 1 else _ood_v4_status(score, profile)
    return {
        "part_id": str(part["part_id"].iloc[0]),
        "score": score,
        "status": status,
        "reasons": list(dict.fromkeys(reasons)),
        "components": c,
        "schema_missing_fields": missing,
        "semantic_status": fs,
        "parameter_status": c["parameter_status"]
    }


def fit_ood_profile(
    reference: pd.DataFrame | str | Path,
    output_path: str | Path | None = None,
    config_path: str | Path | None = None
) -> tuple[OODProfileV4, dict[str, Any]]:
    cfg = load_yaml(config_path or (PROJECT_ROOT / "configs" / "models.yaml"))
    d = _read_any(reference) if isinstance(reference, (str, Path)) else reference.copy()
    original_rows = len(d)
    max_rows = int(cfg.get("ood", {}).get("max_reference_rows", 50000))
    if len(d) > max_rows:
        d = d.sample(n=max_rows, random_state=int(cfg.get("runtime", {}).get("random_seed", 20260831)))
    expected = sorted(map(str, d.columns))
    x = _v3_prepare(d)
    fam = sorted(_aggregate_unique(_canonical_category(d, "component_family")))
    params = sorted(_aggregate_unique(_canonical_category(x, "parameter")))
    sem = sorted(_aggregate_unique(_canonical_category(d, "semantic_type")))
    methods = sorted(_aggregate_unique(_canonical_category(d, "test_method")))
    stress = sorted(_aggregate_unique(_canonical_category(d, "stress_mode")))
    cond = {}
    traj = {}
    for (family, param), g in x.groupby(["component_family", "parameter"], dropna=False, sort=True):
        key = _v3_key(family, param)
        st = _v3_robust_stats(g["value_num"])
        if st:
            cond[key] = {"value": st}
        ss = []
        for _, pg in g.groupby("part_id", sort=False):
            pg = pg.dropna(subset=["time_h", "value_num"]).sort_values("time_h")
            if len(pg) >= 2:
                dt = float(pg["time_h"].iloc[-1] - pg["time_h"].iloc[0])
                if dt > 0:
                    ss.append(float((pg["value_num"].iloc[-1] - pg["value_num"].iloc[0]) / dt))
        rst = _v3_robust_stats(pd.Series(ss))
        if rst:
            traj[key] = {"slope": rst}
    for param, g in x.groupby("parameter", dropna=False, sort=True):
        key = _v3_key("*", param)
        if key not in cond:
            st = _v3_robust_stats(g["value_num"])
            if st:
                cond[key] = {"value": st}
        if key not in traj:
            ss = []
            for _, pg in g.groupby("part_id", sort=False):
                pg = pg.dropna(subset=["time_h", "value_num"]).sort_values("time_h")
                if len(pg) >= 2:
                    dt = float(pg["time_h"].iloc[-1] - pg["time_h"].iloc[0])
                    if dt > 0:
                        ss.append(float((pg["value_num"].iloc[-1] - pg["value_num"].iloc[0]) / dt))
            rst = _v3_robust_stats(pd.Series(ss))
            if rst:
                traj[key] = {"slope": rst}
    provisional = OODProfileV4(
        "ood_profile_v4", _config_hash(cfg), expected, fam, params, sem, methods, stress,
        int(len(d)), int(d["part_id"].nunique()) if "part_id" in d else 0, max_rows, cond, traj, 0.40, 0.75
    )
    wide = _v3_prepare(d)
    scores = []
    if "part_id" in wide.columns:
        for _, pg in wide.groupby(wide["part_id"].astype(str), sort=False):
            scores.append(_ood_v4_score_part(provisional, pg, set(expected))["score"])
    arr = np.asarray(scores, dtype=float)
    moderate = float(np.quantile(arr, 0.95)) if arr.size else 0.40
    severe = float(np.quantile(arr, 0.99)) if arr.size else 0.75
    if severe <= moderate:
        severe = min(1.0, moderate + 0.05)
    profile = OODProfileV4(
        "ood_profile_v4", _config_hash(cfg), expected, fam, params, sem, methods, stress,
        int(len(d)), int(d["part_id"].nunique()) if "part_id" in d else 0, max_rows, cond, traj, moderate, severe
    )
    metrics = {
        "profile_version": profile.profile_version,
        "reference_rows": profile.reference_rows,
        "original_rows": original_rows,
        "reference_parts": profile.reference_parts,
        "moderate_threshold": moderate,
        "severe_threshold": severe
    }
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format": "ood_profile_v4", "profile": asdict(profile)}, p, compress=3)
        write_artifact_manifest(
            p, artifact_kind="ood_profile", config_hash=profile.config_hash,
            training_data=reference if isinstance(reference, (str, Path)) else None,
            extra={"format": "ood_profile_v4", "moderate_threshold": moderate, "severe_threshold": severe}
        )
    return profile, metrics


def load_ood_profile(path: str | Path, *, require_compatibility: bool = True) -> OODProfileV4:
    p = Path(path)
    if require_compatibility:
        validate_artifact_compatibility(p, artifact_kind="ood_profile", require_manifest=True)
    payload = joblib.load(p)
    if isinstance(payload, OODProfileV4):
        return payload
    if isinstance(payload, dict) and payload.get("format") == "ood_profile_v4":
        return OODProfileV4(**payload["profile"])
    if isinstance(payload, dict) and payload.get("format") in {"ood_profile_v2", "ood_profile_v3", "ood_profile_v3_1"}:
        prof = payload.get("profile", payload)
        prof = dict(prof)
        prof.setdefault("moderate_threshold", 0.40)
        prof.setdefault("severe_threshold", 0.75)
        return OODProfileV4(**prof)
    raise ValueError("Unsupported OOD profile artifact; rebuild with ood_profile_v4.")


def assess_data_ood(profile: OODProfileV4, input_df: pd.DataFrame, as_of_h: float | None = None) -> dict[str, Any]:
    d = _v3_prepare(input_df, as_of_h=as_of_h)
    schema = set(map(str, input_df.columns))
    parts = []
    if "part_id" not in d.columns:
        return {"score": 1.0, "status": "SEVERE", "parts": [], "profile_version": profile.profile_version, "schema": {"status": "DEGRADED"}}
    for _, pg in d.groupby(d["part_id"].astype(str), sort=False):
        parts.append(_ood_v4_score_part(profile, pg, schema))
    score = max([p["score"] for p in parts], default=0.0)
    status = "SEVERE" if any(p["status"] == "SEVERE" for p in parts) else ("MODERATE" if any(p["status"] == "MODERATE" for p in parts) else "LOW")
    missing = sorted(set(profile.expected_columns) - schema)
    return {
        "score": score,
        "status": status,
        "parts": parts,
        "profile_version": profile.profile_version,
        "schema_expected": profile.expected_columns,
        "schema": {"status": "COMPLETE" if not missing else "DEGRADED", "missing_fields": missing}
    }


def _extract_forecast_evidence(forecast: pd.DataFrame, part_id: str) -> dict[str, Any]:
    d = forecast[forecast.get("part_id", pd.Series(index=forecast.index, dtype="object")).astype(str).eq(str(part_id))].copy()
    if d.empty:
        return {"failure_risk": 0.0, "uncertainty": 1.0, "near_limit": False, "limit_cross": False, "target_available": False, "evidence_count": 0, "rows": 0}
    pred = pd.to_numeric(d.get("prediction_168h"), errors="coerce") if "prediction_168h" in d else pd.Series(np.nan, index=d.index)
    lo = pd.to_numeric(d.get("prediction_lower"), errors="coerce") if "prediction_lower" in d else pd.Series(np.nan, index=d.index)
    hi = pd.to_numeric(d.get("prediction_upper"), errors="coerce") if "prediction_upper" in d else pd.Series(np.nan, index=d.index)
    p_ex = pd.to_numeric(d.get("limit_exceedance_probability_proxy"), errors="coerce") if "limit_exceedance_probability_proxy" in d else pd.Series(np.nan, index=d.index)
    upper_limit = None
    for c in ("absolute_limit_upper", "engineering_limit_upper", "upper_limit", "spec_upper"):
        if c in d.columns:
            upper_limit = pd.to_numeric(d[c], errors="coerce")
            break
    valid_pred = pred.notna()
    failure_risk = float(p_ex[valid_pred].max()) if valid_pred.any() and p_ex[valid_pred].notna().any() else 0.0
    limit_cross = bool(((hi > upper_limit) & upper_limit.notna()).any()) if upper_limit is not None else False
    near_limit = False
    if upper_limit is not None:
        ratio = (pred / upper_limit.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        near_limit = bool((ratio >= 0.95).fillna(False).any())
    widths = (hi - lo).abs() if len(lo) else pd.Series(dtype=float)
    rel_width = widths / pred.abs().replace(0, np.nan) if len(pred) else pd.Series(dtype=float)
    uncertainty = float(np.clip(rel_width.median(skipna=True) if rel_width.notna().any() else 1.0, 0, 1))
    support = int(limit_cross) + int(near_limit) + int(failure_risk >= 0.60) + int("safety_slope_flag" in d.columns and pd.to_numeric(d["safety_slope_flag"], errors="coerce").fillna(0).max() > 0)
    return {
        "failure_risk": failure_risk,
        "uncertainty": uncertainty,
        "near_limit": near_limit,
        "limit_cross": limit_cross,
        "target_available": valid_pred.any(),
        "evidence_count": support,
        "rows": len(d),
        "max_prediction": float(pred.max()) if valid_pred.any() else None,
        "max_upper": float(hi.max()) if hi.notna().any() else None
    }


def _extract_anomaly_evidence(anomaly: pd.DataFrame, part_id: str) -> dict[str, Any]:
    d = anomaly[anomaly.get("part_id", pd.Series(index=anomaly.index, dtype="object")).astype(str).eq(str(part_id))].copy()
    if d.empty:
        return {"anomaly_risk": 0.0, "absolute_violation": False, "support": 0, "population": 0.0, "temporal": 0.0, "multivariate": 0.0}
    raw = pd.to_numeric(d.get("anomaly_score"), errors="coerce") if "anomaly_score" in d else pd.Series(0.0, index=d.index)
    cal = pd.to_numeric(d.get("calibrated_risk_score"), errors="coerce") if "calibrated_risk_score" in d else pd.Series(np.nan, index=d.index)
    risk = cal.max(skipna=True) if cal.notna().any() else raw.max(skipna=True)
    abs_v = bool(pd.to_numeric(d.get("absolute_violation"), errors="coerce").fillna(0).max() >= 1) if "absolute_violation" in d else False
    support = int(pd.to_numeric(d.get("anomaly_support_count"), errors="coerce").fillna(0).max()) if "anomaly_support_count" in d else 0
    return {
        "anomaly_risk": float(np.clip(risk if pd.notna(risk) else 0, 0, 1)),
        "raw_anomaly": float(np.clip(raw.max(skipna=True) if raw.notna().any() else 0, 0, 1)),
        "absolute_violation": abs_v,
        "support": support,
        "population": float(pd.to_numeric(d.get("population_evidence"), errors="coerce").fillna(0).max()) if "population_evidence" in d else 0.0,
        "temporal": float(pd.to_numeric(d.get("temporal_evidence"), errors="coerce").fillna(0).max()) if "temporal_evidence" in d else 0.0,
        "multivariate": float(pd.to_numeric(d.get("multivariate_component_score"), errors="coerce").fillna(0).max()) if "multivariate_component_score" in d else 0.0,
        "high_but_safe": bool(pd.to_numeric(d.get("high_but_safe", 0), errors="coerce").fillna(0).max() >= 1) if "high_but_safe" in d else False,
        "measurement_only_anomaly": bool(pd.to_numeric(d.get("measurement_only_anomaly", 0), errors="coerce").fillna(0).max() >= 1) if "measurement_only_anomaly" in d else False
    }


def _combine_risk(a: dict[str, Any], f: dict[str, Any], ood: dict[str, Any], quality: dict[str, Any]) -> tuple[float, float, float, int]:
    anomaly = float(a["anomaly_risk"])
    failure = float(f["failure_risk"])
    combined = max(anomaly, failure)
    if anomaly >= 0.60 and failure >= 0.60:
        combined = min(1.0, combined + 0.12)
    if a["absolute_violation"]:
        combined = 1.0
    uncertainty = float(np.clip(max(f.get("uncertainty", 1.0 if not f.get("target_available") else 0.0), ood.get("score", 0.0), 1.0 - quality.get("score", 0.0)), 0, 1))
    support = int(a.get("support", 0) >= 2) + int(f.get("evidence_count", 0) >= 1) + int(a.get("population", 0) >= 0.65) + int(a.get("temporal", 0) >= 0.65) + int(a.get("multivariate", 0) >= 0.65)
    return float(np.clip(combined, 0, 1)), float(failure), uncertainty, support


def _human_confidence(risk: float, uncertainty: float, ood_score: float, quality: float) -> str:
    if quality < 0.60 or uncertainty >= 0.70 or ood_score >= 0.75:
        return "LOW"
    if uncertainty >= 0.40 or ood_score >= 0.40:
        return "MODERATE"
    if risk >= 0.80 or risk <= 0.20:
        return "HIGH"
    return "MODERATE"


def _decision_for_case(part_id: str, anomaly_e: dict[str, Any], forecast_e: dict[str, Any], ood_e: dict[str, Any], quality_e: dict[str, Any], policy: Mapping[str, Any]) -> ScreeningAssessment:
    combined, failure_risk, uncertainty, support = _combine_risk(anomaly_e, forecast_e, ood_e, quality_e)
    rp = policy["risk_score"]["thresholds"]
    safe_max = float(rp["safe_max"])
    reject_min = float(rp["reject_min"])
    hard_limit = bool(anomaly_e["absolute_violation"])
    near = bool(forecast_e["near_limit"])
    reasons: list[str] = []
    warnings: list[str] = list(quality_e.get("reasons", []))

    if hard_limit:
        decision = "REJECT"
        reasons.append("Authoritative absolute-limit violation detected; safety policy mandates REJECT.")
    elif ood_e.get("status") == "SEVERE":
        decision = "REVIEW"
        reasons.extend(ood_e.get("reasons", [])[:3])
    elif quality_e.get("status") != "PASS":
        decision = "UNKNOWN"
        reasons.extend(quality_e.get("reasons", [])[:3])
    elif not anomaly_e.get("support") and not forecast_e.get("target_available"):
        decision = "UNKNOWN"
        reasons.append("Insufficient anomaly/forecast evidence for a defensible safety decision.")
    elif uncertainty >= float(policy["uncertainty"].get("high_uncertainty_threshold", 0.70)):
        decision = "REVIEW"
        reasons.append("Prediction/anomaly evidence is too uncertain for automatic disposition.")
    elif failure_risk >= reject_min and support >= 2:
        decision = "REJECT"
        reasons.append(f"Forecasted/observed failure risk is high ({failure_risk:.2f}) with independent supporting evidence.")
    elif combined >= reject_min and support >= 2:
        decision = "REJECT"
        reasons.append(f"Combined risk is high ({combined:.2f}) with multiple supporting evidence channels.")
    elif near:
        decision = "REVIEW"
        reasons.append("Upper prediction evidence approaches/crosses the engineering limit; human review required.")
    elif ood_e.get("status") == "MODERATE":
        decision = "REVIEW"
        reasons.append("Input lies outside part of the reference operating distribution.")
    elif combined > safe_max:
        decision = "REVIEW"
        reasons.append(f"Risk ({combined:.2f}) is above the automatic-safe region.")
    else:
        decision = "SAFE"
        reasons.append("No hard-limit violation and no material supported anomaly/future-risk evidence was found.")

    if anomaly_e.get("population", 0) >= 0.65:
        reasons.append(f"Lot-relative population evidence is elevated ({anomaly_e['population']:.2f}).")
    if anomaly_e.get("temporal", 0) >= 0.65:
        reasons.append(f"Early temporal-deviation evidence is elevated ({anomaly_e['temporal']:.2f}).")
    if forecast_e.get("limit_cross"):
        reasons.append("Forecast upper bound crosses the available engineering limit.")
    elif forecast_e.get("target_available"):
        reasons.append(f"168 h point prediction is {forecast_e.get('max_prediction')!s}; upper bound {forecast_e.get('max_upper')!s}.")

    high_safe = bool(anomaly_e.get("high_but_safe", False))
    measurement_only = bool(anomaly_e.get("measurement_only_anomaly", False))
    confidence = _human_confidence(combined, uncertainty, ood_e.get("score", 0), quality_e.get("score", 0))
    evidence_state = "SUPPORTED" if support >= 2 else ("LIMITED" if support == 1 else "INSUFFICIENT")
    trace = {
        "module_a": anomaly_e,
        "module_b": forecast_e,
        "ood": {k: v for k, v in ood_e.items() if k != "parts"},
        "data_quality": quality_e,
        "fusion": {"combined_risk": combined, "supporting_evidence_count": support, "uncertainty": uncertainty},
        "policy": {"safe_max": safe_max, "reject_min": reject_min, "hard_limit_override": hard_limit},
    }
    return ScreeningAssessment(
        part_id=str(part_id), decision=decision, risk_score=combined, confidence=confidence,
        failure_risk=failure_risk, anomaly_risk=anomaly_e["anomaly_risk"], ood_score=float(ood_e.get("score", 0)),
        ood_status=str(ood_e.get("status", "UNKNOWN")), uncertainty_score=uncertainty,
        data_quality_score=float(quality_e.get("score", 0)), evidence_state=evidence_state,
        hard_limit_violation=hard_limit, near_limit=near, supporting_evidence_count=support,
        high_but_safe=high_safe, measurement_only_anomaly=measurement_only,
        reasons=list(dict.fromkeys(reasons)), warnings=warnings,
        trace=trace,
    )


def assess_screening(
    anomaly: pd.DataFrame,
    forecast: pd.DataFrame | None,
    input_df: pd.DataFrame,
    ood_profile: OODProfileV4 | None = None,
    policy_path: str | Path | None = None
) -> pd.DataFrame:
    policy = load_yaml(policy_path or (PROJECT_ROOT / "configs" / "policy.yaml"))
    if "part_id" not in input_df.columns:
        raise ValueError("Safety assessment requires part_id in input data")
    if ood_profile is None:
        ood_rows = {str(p): {"score": 0.0, "status": "UNKNOWN", "reasons": []} for p in input_df.part_id.astype(str).unique()}
    else:
        ood_global = assess_data_ood(ood_profile, input_df)
        ood_rows = {r["part_id"]: r for r in ood_global["parts"]}
    rows = []
    for pid in input_df.part_id.astype(str).unique():
        a = _extract_anomaly_evidence(anomaly, pid)
        f = _extract_forecast_evidence(forecast, pid) if forecast is not None else {"failure_risk": 0.0, "uncertainty": 1.0, "near_limit": False, "limit_cross": False, "target_available": False, "evidence_count": 0}
        o = ood_rows.get(pid, {"score": 0.5, "status": "UNKNOWN", "reasons": ["OOD profile did not cover this part."]})
        q = _part_quality(input_df, pid, policy)
        assessment = _decision_for_case(pid, a, f, o, q, policy)
        rows.append(asdict(assessment))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["reasons_json"] = out["reasons"].apply(json.dumps)
        out["warnings_json"] = out["warnings"].apply(json.dumps)
        out["trace_json"] = out["trace"].apply(lambda x: json.dumps(x, default=str))
        out = out.drop(columns=["reasons", "warnings", "trace"])
    return out


def screen(
    anomaly_path: str | Path,
    forecast_path: str | Path | None,
    input_path: str | Path,
    ood_artifact: str | Path | None = None,
    output_path: str | Path | None = None,
    policy_path: str | Path | None = None
) -> pd.DataFrame:
    anomaly = _read_any(anomaly_path)
    forecast = _read_any(forecast_path) if forecast_path else None
    input_df = _read_any(input_path)
    profile = load_ood_profile(ood_artifact) if ood_artifact else None
    out = assess_screening(anomaly, forecast, input_df, profile, policy_path)
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Safety/OOD decision engine")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit-ood")
    f.add_argument("reference")
    f.add_argument("artifact")
    s = sub.add_parser("screen")
    s.add_argument("anomaly")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--forecast", default=None)
    s.add_argument("--ood-artifact", default=None)
    args = p.parse_args()
    if args.cmd == "fit-ood":
        _, m = fit_ood_profile(args.reference, args.artifact)
        print(json.dumps(m, indent=2))
    else:
        d = screen(args.anomaly, args.forecast, args.input, args.ood_artifact, args.output)
        print(d[["part_id", "decision", "risk_score", "confidence", "ood_status"]].to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
