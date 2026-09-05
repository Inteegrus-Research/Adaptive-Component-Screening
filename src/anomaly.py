"""Module A — Component Anomaly and Latent Failure Screening.

Implements multi-channel unsupervised evidence generation (Absolute limits,
Robust PAT population statistics, Temporal degradation, and Multivariate novelty)
coupled with validation-calibrated fusion and an aerospace sparse-positive safeguard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.utils import (
    PROJECT_ROOT,
    load_yaml,
    seed_everything,
    stable_config_hash as _stable_hash,
    validate_artifact_compatibility,
    write_artifact_manifest,
)

SAFE_FUTURE_COLUMNS = {
    "value_96h", "value_168h", "target_168h", "latent_96h", "latent_168h",
    "future_defective_48h", "future_defective_96h", "future_defective_168h",
    "future_defective_240h", "future_defective_300h",
}
EVIDENCE_COLUMNS = ["absolute_limits", "robust_population", "temporal_deviation", "multivariate"]
DEFAULT_PARAMETER_FEATURES = [
    "population_robust_z", "population_z", "population_percentile",
    "population_relative_deviation", "relative_delta_0_24", "normalized_slope_0_24",
]


def stable_config_hash(config: Mapping[str, object]) -> str:
    return _stable_hash(config)


def _canonicalize_parameter_series(series: pd.Series) -> pd.Series:
    """Ensure parameters match the canonical ontology across train and score pipelines."""
    try:
        from src.ingest import ParameterOntology
        ontology = ParameterOntology(load_yaml(PROJECT_ROOT / "configs" / "parameters.yaml"))
        def resolve(val: Any) -> Any:
            if pd.isna(val):
                return val
            spec, _, _ = ontology.resolve(val)
            return spec.name if spec is not None else str(val).strip()
        return series.map(resolve)
    except Exception:
        return series.astype(str).str.strip()


@dataclass
class FamilyDetector:
    feature_columns: list[str]
    scaler: StandardScaler
    isolation_forest: IsolationForest | None = None
    robust_covariance: MinCovDet | None = None
    lof: LocalOutlierFactor | None = None


@dataclass
class AnomalyModel:
    parameter_detectors: dict[str, IsolationForest]
    family_detectors: dict[str, FamilyDetector]
    feature_columns: list[str]
    train_lots: list[str]
    validation_lots: list[str]
    test_lots: list[str]
    weights: dict[str, float]
    fusion_model: LogisticRegression | None
    calibration_model: IsotonicRegression | None
    calibrator: IsotonicRegression | None
    calibration_method: str
    config_hash: str
    as_of_h: float
    reference_version: str = "phase4_fusion_safeguard"
    validation_positive_count: int = 0
    validation_positive_prevalence: float = 0.0


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False)
    if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(p) as con:
            d = pd.read_sql_query("SELECT * FROM component_parameter_state", con)
            try:
                lim = pd.read_sql_query(
                    "SELECT profile_id, parameter, "
                    "MAX(absolute_limit_lower) AS absolute_limit_lower, "
                    "MAX(absolute_limit_upper) AS absolute_limit_upper "
                    "FROM measurements GROUP BY profile_id, parameter", con)
                if {"profile_id", "parameter"}.issubset(d.columns):
                    d = d.merge(lim, on=["profile_id", "parameter"], how="left", suffixes=("", "_lim"))
                    for c in ["absolute_limit_lower", "absolute_limit_upper"]:
                        if f"{c}_lim" in d.columns:
                            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(
                                pd.to_numeric(d[f"{c}_lim"], errors="coerce")
                            )
                            d.drop(columns=[f"{c}_lim"], inplace=True)
            except Exception:
                pass
            return d
    raise ValueError(f"Unsupported input format: {p.suffix}")


def _read_dense_table(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    candidates: list[Path] = []
    if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        try:
            with sqlite3.connect(p) as con:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                if "dense_measurements" in tables:
                    return pd.read_sql_query("SELECT * FROM dense_measurements", con)
        except Exception:
            pass
        candidates.append(p.parent / "csv" / "dense_measurements.csv")
        candidates.append(p.parent / "dense_measurements.csv")
    else:
        candidates.append(p.parent / "dense_measurements.csv")
    for c in candidates:
        if c.exists():
            try:
                return pd.read_csv(c, low_memory=False)
            except Exception:
                continue
    return None


def _read_labels(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    try:
        if p.suffix.lower() == ".csv":
            lab = pd.read_csv(p, low_memory=False)
            return lab if "part_id" in lab.columns else None
        if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            with sqlite3.connect(p) as con:
                return pd.read_sql_query("SELECT * FROM labels", con)
    except Exception:
        return None
    return None


def _wide_state(df: pd.DataFrame, as_of_h: float) -> pd.DataFrame:
    d = df.copy()
    if "parameter" in d.columns:
        d["raw_parameter"] = d["parameter"].copy()
        d["parameter"] = _canonicalize_parameter_series(d["parameter"])

    required = {"part_id", "lot_id", "parameter", "value_0h"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Module-A input missing: {missing}")
    if as_of_h >= 24 and "value_24h" not in d.columns:
        raise ValueError("24h anomaly mode requires value_24h")

    keep = [c for c in [
        "part_id", "lot_id", "profile_id", "component_family", "part_type", "component_type",
        "parameter", "raw_parameter", "semantic_type", "physical_quantity", "unit",
        "value_0h", "value_24h", "robust_z_0h", "robust_z_24h",
        "slope_z_0_24", "observable_slope_0_24", "observable_safety_slope",
        "absolute_limit_lower", "absolute_limit_upper", "split", "defect_state",
        "latent_defect_label", "high_but_safe", "measurement_only_anomaly",
        "precursor_strength", "precursor_effect", "failure_mode", "primary_failure_mode",
    ] if c in d.columns]
    x = d[keep].copy()
    numeric_cols = [
        "value_0h", "value_24h", "robust_z_0h", "robust_z_24h", "slope_z_0_24",
        "observable_slope_0_24", "observable_safety_slope", "absolute_limit_lower",
        "absolute_limit_upper", "precursor_strength", "precursor_effect",
    ]
    for c in numeric_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x["value_asof"] = x["value_24h"] if as_of_h >= 24 else x["value_0h"]
    if "value_24h" in x.columns:
        x["delta_0_24"] = x["value_24h"] - x["value_0h"]
        x["relative_delta_0_24"] = x["delta_0_24"] / x["value_0h"].abs().replace(0, np.nan)
        x["slope_0_24"] = x["delta_0_24"] / 24.0
        x["normalized_slope_0_24"] = x["relative_delta_0_24"] / 24.0
    else:
        for c in ["delta_0_24", "relative_delta_0_24", "slope_0_24", "normalized_slope_0_24"]:
            x[c] = np.nan
    return x


def _lot_robust(df: pd.DataFrame, value_col: str = "value_asof") -> pd.DataFrame:
    x = df.copy()
    keys = [c for c in ["lot_id", "component_family", "parameter"] if c in x.columns]
    v = pd.to_numeric(x[value_col], errors="coerce")
    g = v.groupby([x[k] for k in keys], dropna=False, sort=False)
    x["ref_n"] = g.transform("count")
    x["ref_mean"] = g.transform("mean")
    x["ref_std"] = g.transform("std")
    x["ref_median"] = g.transform("median")
    x["ref_q25"] = g.transform(lambda s: s.quantile(0.25))
    x["ref_q75"] = g.transform(lambda s: s.quantile(0.75))
    x["ref_mad"] = g.transform(
        lambda s: np.median(np.abs(s.dropna() - np.median(s.dropna()))) if s.dropna().size else np.nan
    )
    mad_scale = 1.4826 * x["ref_mad"]
    iqr_scale = (x["ref_q75"] - x["ref_q25"]) / 1.349
    scale = pd.concat([mad_scale, iqr_scale], axis=1).max(axis=1).replace(0, np.nan)
    x["population_deviation"] = v - x["ref_median"]
    x["population_robust_z"] = x["population_deviation"] / scale
    x["population_z"] = (v - x["ref_mean"]) / x["ref_std"].replace(0, np.nan)
    x["population_iqr"] = x["ref_q75"] - x["ref_q25"]
    x["population_percentile"] = v.groupby([x[k] for k in keys], dropna=False, sort=False).rank(pct=True)
    x["population_relative_deviation"] = x["population_deviation"] / x["ref_median"].abs().replace(0, np.nan)
    return x


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _robust_score_to_probability(z) -> np.ndarray:
    a = np.abs(pd.to_numeric(z, errors="coerce").to_numpy(dtype=float))
    a = np.nan_to_num(a, nan=0.0)
    return _logistic((a - 3.0) * 1.0)


def _robust_probability(z) -> np.ndarray:
    return _robust_score_to_probability(z)


def _temporal_probability(df: pd.DataFrame) -> np.ndarray:
    z = np.abs(pd.to_numeric(df.get("population_robust_z"), errors="coerce").fillna(0.0).to_numpy(dtype=float))
    rel = np.abs(pd.to_numeric(df.get("relative_delta_0_24"), errors="coerce").fillna(0.0).to_numpy(dtype=float))
    ref = float(np.nanpercentile(rel, 75)) if np.isfinite(rel).any() else 0.0
    drift = _logistic((rel - ref) / max(ref * 0.75, 1e-9))
    return np.clip(0.6 * _logistic((z - 2.5) * 0.9) + 0.4 * drift, 0, 1)


def _dense_temporal_map(dense: pd.DataFrame | None, as_of_h: float) -> pd.DataFrame:
    if dense is None or dense.empty:
        return pd.DataFrame(columns=["part_id", "parameter", "dense_last2_slope", "dense_curvature",
                                     "dense_smoothed_deviation", "dense_temporal_score"])
    required = {"part_id", "parameter", "time_h", "value"}
    if not required.issubset(dense.columns):
        return pd.DataFrame(columns=["part_id", "parameter", "dense_last2_slope", "dense_curvature",
                                     "dense_smoothed_deviation", "dense_temporal_score"])
    d = dense.copy()
    d["parameter"] = _canonicalize_parameter_series(d["parameter"])
    d["time_h"] = pd.to_numeric(d["time_h"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d[d["time_h"].le(float(as_of_h)) & d["time_h"].notna() & d["value"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["part_id", "parameter", "dense_last2_slope", "dense_curvature",
                                     "dense_smoothed_deviation", "dense_temporal_score"])
    rows = []
    for (pid, param), g in d.groupby([d["part_id"].astype(str), d["parameter"].astype(str)], sort=False):
        g = g.sort_values("time_h")
        if len(g) < 2:
            continue
        t = g["time_h"].to_numpy(float)
        y = g["value"].to_numpy(float)
        dt = np.diff(t)
        dy = np.diff(y)
        valid = dt > 0
        slopes = dy[valid] / dt[valid]
        last_slope = float(slopes[-1]) if len(slopes) else 0.0
        prev_slope = float(slopes[-2]) if len(slopes) >= 2 else last_slope
        curvature = last_slope - prev_slope
        win = min(5, len(y))
        smoothed = float(np.mean(y[-win:]))
        baseline_window = min(max(3, len(y) // 2), len(y))
        baseline = float(np.median(y[:baseline_window]))
        deviation = (smoothed - baseline) / max(abs(baseline), 1e-12)
        scale = float(np.median(np.abs(y - np.median(y))))
        slope_score = float(np.clip(abs(last_slope) / max(scale, 1e-12), 0, 1))
        curve_score = float(np.clip(abs(curvature) / max(abs(last_slope) + 1e-12, 1e-12), 0, 1))
        dev_score = float(_logistic(np.array([(abs(deviation) - 0.02) / 0.02]))[0])
        score = float(np.clip(0.45 * slope_score + 0.25 * curve_score + 0.30 * dev_score, 0, 1))
        rows.append({
            "part_id": str(pid), "parameter": str(param), "dense_last2_slope": last_slope,
            "dense_curvature": curvature, "dense_smoothed_deviation": deviation,
            "dense_temporal_score": score,
        })
    return pd.DataFrame(rows)


def _split_lots(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    if "split" in df.columns and df["split"].notna().any():
        train = sorted(df.loc[df["split"].astype(str).eq("train"), "lot_id"].astype(str).unique())
        val = sorted(df.loc[df["split"].astype(str).eq("val"), "lot_id"].astype(str).unique())
        test = sorted(df.loc[df["split"].astype(str).eq("test"), "lot_id"].astype(str).unique())
        if train and val:
            return train, val, test
    lots = sorted(df["lot_id"].astype(str).unique())
    n = len(lots)
    nt = max(1, int(round(0.70 * n)))
    nv = max(1, int(round(0.15 * n))) if n >= 7 else 0
    return lots[:nt], lots[nt:nt + nv], lots[nt + nv:]


def _build_component_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    base_cols = [c for c in [
        "part_id", "lot_id", "profile_id", "component_family", "part_type", "component_type", "split",
        "latent_defect_label", "defect_state", "high_but_safe", "measurement_only_anomaly",
        "failure_mode", "primary_failure_mode",
    ] if c in d.columns]
    d["pkey"] = d["semantic_type"].fillna(d["parameter"]).astype(str) if "semantic_type" in d.columns else d["parameter"].astype(str)
    cols_needed = base_cols + ["pkey", "value_asof", "population_robust_z", "normalized_slope_0_24"]
    if "absolute_limit_upper" in d.columns:
        work = d[base_cols + ["pkey", "value_asof", "population_robust_z", "normalized_slope_0_24", "absolute_limit_upper"]].copy()
        work["_hard_upper"] = (
            (pd.to_numeric(work["value_asof"], errors="coerce") > pd.to_numeric(work["absolute_limit_upper"], errors="coerce"))
            & work["absolute_limit_upper"].notna()
        ).astype(float)
    else:
        work = d[cols_needed].copy()
        work["_hard_upper"] = 0.0
    x = work.groupby(base_cols + ["pkey"], dropna=False, sort=False, as_index=False).agg(
        value_asof=("value_asof", "median"),
        population_robust_z=("population_robust_z", "median"),
        normalized_slope_0_24=("normalized_slope_0_24", "median"),
        _hard_upper=("_hard_upper", "max"),
    )
    keys = [c for c in base_cols if c in x.columns]
    z = x.pivot_table(index=keys, columns="pkey", values="population_robust_z", aggfunc="first")
    dr = x.pivot_table(index=keys, columns="pkey", values="normalized_slope_0_24", aggfunc="first")
    present = x.assign(_present=x["value_asof"].notna().astype(float)).pivot_table(index=keys, columns="pkey", values="_present", aggfunc="max")
    hard = x.pivot_table(index=keys, columns="pkey", values="_hard_upper", aggfunc="max")
    z.columns = [f"z__{c}" for c in z.columns]
    dr.columns = [f"drift__{c}" for c in dr.columns]
    present.columns = [f"present__{c}" for c in present.columns]
    hard.columns = [f"hard_upper__{c}" for c in hard.columns]
    out = z.join(dr, how="outer").join(present, how="outer").join(hard, how="outer").reset_index()
    feature_cols = [c for c in out.columns if c.startswith(("z__", "drift__", "present__", "hard_upper__"))]
    return out, feature_cols


def _fit_family_detector(X: np.ndarray, config: Mapping[str, object], seed: int) -> FamilyDetector:
    scaler = StandardScaler().fit(np.nan_to_num(X, nan=0.0))
    Xs = scaler.transform(np.nan_to_num(X, nan=0.0))
    max_n = min(len(Xs), 10000)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), max_n, replace=False) if max_n < len(Xs) else np.arange(len(Xs))
    train = Xs[idx]
    iso_cfg = config.get("anomaly", {}).get("baselines", {}).get("isolation_forest", {})
    iso = IsolationForest(
        n_estimators=min(int(iso_cfg.get("n_estimators", 400)), 250),
        contamination="auto",
        max_samples="auto",
        random_state=seed,
        n_jobs=int(config.get("runtime", {}).get("n_jobs", -1)),
    ).fit(train)
    cov = None
    cov_cfg = config.get("anomaly", {}).get("baselines", {}).get("robust_covariance", {})
    if cov_cfg.get("enabled") is True and train.shape[1] <= 8 and len(train) >= 300:
        try:
            cov = MinCovDet(random_state=seed, support_fraction=0.75).fit(train)
        except Exception:
            cov = None
    lof = None
    lof_cfg = config.get("anomaly", {}).get("baselines", {}).get("local_outlier_factor", {})
    if lof_cfg.get("enabled") is True and len(train) <= 5000 and len(train) >= 100:
        try:
            lof = LocalOutlierFactor(n_neighbors=min(int(lof_cfg.get("n_neighbors", 35)), len(train) - 1), novelty=True).fit(train)
        except Exception:
            lof = None
    return FamilyDetector(feature_columns=[], scaler=scaler, isolation_forest=iso, robust_covariance=cov, lof=lof)


def _score_unsupervised(model: IsolationForest | LocalOutlierFactor | None, X: np.ndarray) -> np.ndarray:
    if model is None or len(X) == 0:
        return np.zeros(len(X))
    raw = -model.score_samples(X)
    if not np.isfinite(raw).any():
        return np.zeros(len(X))
    lo, hi = np.nanpercentile(raw, [5, 95])
    return np.clip((raw - lo) / max(hi - lo, 1e-9), 0, 1)


def _score_cov(model: MinCovDet | None, X: np.ndarray) -> np.ndarray:
    if model is None or len(X) == 0:
        return np.zeros(len(X))
    try:
        d = model.mahalanobis(X)
        lo, hi = np.nanpercentile(d, [50, 99])
        return np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    except Exception:
        return np.zeros(len(X))


def _extract_high_safe_flag(df: pd.DataFrame) -> pd.Series:
    c = "high_but_safe" if "high_but_safe" in df.columns else None
    return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int) if c else pd.Series(0, index=df.index)


def _extract_measurement_flag(df: pd.DataFrame) -> pd.Series:
    c = "measurement_only_anomaly" if "measurement_only_anomaly" in df.columns else None
    return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int) if c else pd.Series(0, index=df.index)


def _part_flags(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["part_id", "lot_id", "component_family", "part_type", "component_type", "failure_mode", "primary_failure_mode"] if c in df.columns]
    if "part_id" not in keys:
        keys = ["part_id"]
    out = df.groupby("part_id", dropna=False).agg(
        high_but_safe=("high_but_safe", "max") if "high_but_safe" in df.columns else ("part_id", lambda s: 0),
        measurement_only_anomaly=("measurement_only_anomaly", "max") if "measurement_only_anomaly" in df.columns else ("part_id", lambda s: 0),
    ).reset_index()
    for c in ["lot_id", "component_family", "part_type", "component_type", "failure_mode", "primary_failure_mode"]:
        if c in df.columns:
            tmp = df[["part_id", c]].drop_duplicates("part_id")
            out = out.merge(tmp, on="part_id", how="left")
    return out


def _high_safe_likelihood(part: pd.DataFrame) -> np.ndarray:
    v0 = pd.to_numeric(part.get("value_0h"), errors="coerce")
    v24 = pd.to_numeric(part.get("value_24h"), errors="coerce")
    if v0 is None:
        return np.zeros(len(part))
    baseline = pd.to_numeric(part.get("population_percentile"), errors="coerce")
    if baseline is None or baseline.isna().all():
        baseline = pd.Series(0.5, index=part.index)
    baseline = baseline.fillna(0.5).clip(0, 1)
    drift = ((v24 - v0).abs() / v0.abs().replace(0, np.nan)).fillna(0.0).clip(0, 1)
    high = np.clip((baseline.to_numpy(float) - 0.80) / 0.20, 0, 1)
    low_drift = np.clip(1.0 - drift.to_numpy(float) / 0.05, 0, 1)
    explicit = _extract_high_safe_flag(part).to_numpy(float)
    return np.clip(0.75 * (high * low_drift) + 0.25 * explicit, 0, 1)


def _measurement_anomaly_likelihood(part: pd.DataFrame) -> np.ndarray:
    def _series(names: list[str], default=np.nan):
        for name in names:
            if name in part.columns:
                return pd.to_numeric(part[name], errors="coerce")
        return pd.Series(default, index=part.index, dtype=float)

    repeat_cv = _series(["repeat_cv_percent", "repeatability_cv_percent", "cv_percent"])
    repeat_std = _series(["repeat_std", "repeatability_std", "measurement_noise", "noise_std", "noise_level"])
    value = _series(["value_asof", "value_24h", "value_0h"], default=1.0)
    cv_score = np.clip(repeat_cv.abs().fillna(0.0).to_numpy(float) / 10.0, 0, 1)
    noise_ratio = np.clip(
        (repeat_std.abs() / value.abs().replace(0, np.nan)).fillna(0.0).to_numpy(float) / 0.05,
        0, 1,
    )
    v0 = _series(["value_0h"], default=np.nan)
    v24 = _series(["value_24h"], default=np.nan)
    jump = ((v24 - v0).abs() / v0.abs().replace(0, np.nan)).fillna(0.0).to_numpy(float)
    jump_score = np.clip(jump / 0.20, 0, 1)
    explicit = _extract_measurement_flag(part).to_numpy(float)
    return np.clip(0.45 * cv_score + 0.25 * noise_ratio + 0.10 * jump_score + 0.20 * explicit, 0, 1)


def _evidence_features_for_part(d: pd.DataFrame, model: AnomalyModel, *, dense: pd.DataFrame | None, as_of_h: float) -> pd.DataFrame:
    work = d.copy()
    work["_param_absolute"] = 0.0
    work["_param_population"] = 0.0
    work["_param_temporal"] = 0.0
    work["_param_multi"] = 0.0

    for param, sub in work.groupby("parameter", dropna=False):
        idx = sub.index
        robust = _robust_probability(sub["population_robust_z"])
        temporal = _temporal_probability(sub)
        cols = [c for c in DEFAULT_PARAMETER_FEATURES if c in sub.columns]
        X = sub[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        
        # Double lookup: check canonical key first, raw string fallback second
        det = model.parameter_detectors.get(str(param))
        if det is None and "raw_parameter" in sub.columns:
            raw_p = str(sub["raw_parameter"].iloc[0])
            det = model.parameter_detectors.get(raw_p)
            
        multi = _score_unsupervised(det, X)
        absolute = np.zeros(len(sub))
        if {"absolute_limit_lower", "absolute_limit_upper"}.issubset(sub.columns):
            v = pd.to_numeric(sub["value_asof"], errors="coerce")
            lo = pd.to_numeric(sub["absolute_limit_lower"], errors="coerce")
            hi = pd.to_numeric(sub["absolute_limit_upper"], errors="coerce")
            absolute = ((v < lo) | (v > hi)).fillna(False).astype(float).to_numpy()
        work.loc[idx, "_param_absolute"] = absolute
        work.loc[idx, "_param_population"] = robust
        work.loc[idx, "_param_temporal"] = temporal
        work.loc[idx, "_param_multi"] = multi

    comp, _ = _build_component_matrix(work)
    family_rows = []
    for family, sub in comp.groupby("component_family", dropna=False):
        fd = model.family_detectors.get(str(family))
        score = np.zeros(len(sub))
        if fd:
            X = sub.reindex(columns=fd.feature_columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
            Xs = fd.scaler.transform(X)
            iso = _score_unsupervised(fd.isolation_forest, Xs)
            cov = _score_cov(fd.robust_covariance, Xs)
            lof = _score_unsupervised(fd.lof, Xs)
            score = np.maximum(iso, np.maximum(cov, lof))
        family_rows.append(pd.DataFrame({"part_id": sub.part_id.astype(str).values, "multivariate": score}))
    fam = pd.concat(family_rows, ignore_index=True) if family_rows else pd.DataFrame(columns=["part_id", "multivariate"])

    agg_keys = [c for c in ["part_id", "lot_id", "component_family", "part_type", "component_type"] if c in work.columns]
    param_agg = work.groupby(agg_keys, dropna=False).agg(
        absolute_limits=("_param_absolute", "max"),
        robust_population=("_param_population", "max"),
        temporal_deviation=("_param_temporal", "max"),
        parameter_multivariate=("_param_multi", "max"),
    ).reset_index()
    part = param_agg.merge(fam, on="part_id", how="left")
    part["multivariate"] = np.maximum(
        pd.to_numeric(part["multivariate"], errors="coerce").fillna(0).to_numpy(float),
        pd.to_numeric(part["parameter_multivariate"], errors="coerce").fillna(0).to_numpy(float),
    )

    dense_scores = _dense_temporal_map(dense, as_of_h)
    if not dense_scores.empty:
        dt = dense_scores.groupby("part_id", as_index=False).agg(
            dense_temporal_score=("dense_temporal_score", "max"),
            dense_last2_slope=("dense_last2_slope", "median"),
            dense_curvature=("dense_curvature", "median"),
            dense_smoothed_deviation=("dense_smoothed_deviation", "median"),
        )
        part = part.merge(dt, on="part_id", how="left")
        part["temporal_deviation"] = np.maximum(
            pd.to_numeric(part["temporal_deviation"], errors="coerce").fillna(0).to_numpy(float),
            pd.to_numeric(part["dense_temporal_score"], errors="coerce").fillna(0).to_numpy(float),
        )
    else:
        part["dense_temporal_score"] = 0.0
        part["dense_last2_slope"] = np.nan
        part["dense_curvature"] = np.nan
        part["dense_smoothed_deviation"] = np.nan

    flags = _part_flags(work)
    part = part.merge(flags, on="part_id", how="left", suffixes=("", "_flag"))

    aux = work.groupby("part_id", dropna=False).agg(
        value_0h=("value_0h", "median"),
        value_24h=("value_24h", "median") if "value_24h" in work.columns else ("value_0h", "median"),
        value_asof=("value_asof", "median"),
        population_percentile=("population_percentile", "max") if "population_percentile" in work.columns else ("value_asof", lambda s: 0.5),
        repeat_cv_percent=("repeat_cv_percent", "median") if "repeat_cv_percent" in work.columns else ("value_asof", lambda s: np.nan),
        repeat_std=("repeat_std", "median") if "repeat_std" in work.columns else ("value_asof", lambda s: np.nan),
    ).reset_index()
    part = part.merge(aux, on="part_id", how="left", suffixes=("", "_aux"))
    part["high_safe_likelihood"] = _high_safe_likelihood(part)
    part["measurement_anomaly_likelihood"] = _measurement_anomaly_likelihood(part)

    for c in EVIDENCE_COLUMNS:
        part[c] = pd.to_numeric(part[c], errors="coerce").fillna(0).clip(0, 1)
    return part


def _fit_fusion_models(model: AnomalyModel, validation_evidence: pd.DataFrame, y: np.ndarray | pd.Series) -> tuple[LogisticRegression | None, IsotonicRegression | None, str]:
    X = validation_evidence[EVIDENCE_COLUMNS].to_numpy(float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != len(X):
        raise ValueError(f"Validation label/evidence length mismatch: {len(y_arr)} labels vs {len(X)} evidence rows")
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y_arr)
    X = np.nan_to_num(X[finite], nan=0.0)
    y = y_arr[finite].astype(int)
    if len(y) < 2 or np.unique(y).size < 2:
        return None, None, "none_no_distinct_validation_labels"
    fusion = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=0)
    fusion.fit(X, y)
    p = fusion.predict_proba(X)[:, 1]
    calibration = None
    method = "logistic_validation"
    if len(y) >= 20 and np.unique(p).size >= 4:
        try:
            calibration = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibration.fit(p, y)
            method = "isotonic_validation"
        except Exception:
            calibration = None
    return fusion, calibration, method


def save_anomaly_model(model: AnomalyModel, path: str | Path, training_data: str | Path | None = None) -> None:
    families = {}
    for name, fd in model.family_detectors.items():
        families[name] = {
            "feature_columns": fd.feature_columns,
            "scaler": fd.scaler,
            "isolation_forest": fd.isolation_forest,
            "robust_covariance": fd.robust_covariance,
            "lof": fd.lof,
        }
    payload = {
        "format": "anomaly_model_v3",
        "parameter_detectors": model.parameter_detectors,
        "family_detectors": families,
        "feature_columns": model.feature_columns,
        "train_lots": model.train_lots,
        "validation_lots": model.validation_lots,
        "test_lots": model.test_lots,
        "weights": model.weights,
        "fusion_model": model.fusion_model,
        "calibration_model": model.calibration_model,
        "calibrator": model.calibrator,
        "calibration_method": model.calibration_method,
        "config_hash": model.config_hash,
        "as_of_h": model.as_of_h,
        "reference_version": model.reference_version,
        "validation_positive_count": model.validation_positive_count,
        "validation_positive_prevalence": model.validation_positive_prevalence,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, p, compress=3)
    write_artifact_manifest(
        p,
        artifact_kind="anomaly_model",
        config_hash=model.config_hash,
        training_data=training_data,
        extra={"format": "anomaly_model_v3", "as_of_h": model.as_of_h, "reference_version": model.reference_version},
    )


def load_anomaly_model(path: str | Path, *, require_compatibility: bool = True) -> AnomalyModel:
    p = Path(path)
    if require_compatibility:
        validate_artifact_compatibility(p, artifact_kind="anomaly_model", require_manifest=True)
    payload = joblib.load(p)
    if not isinstance(payload, dict) or payload.get("format") not in {"anomaly_model_v1", "anomaly_model_v2", "anomaly_model_v3"}:
        raise ValueError("Unsupported anomaly artifact format")
    families = {}
    for name, item in payload["family_detectors"].items():
        families[name] = item if isinstance(item, FamilyDetector) else FamilyDetector(**item)
    fusion = payload.get("fusion_model")
    calibration = payload.get("calibration_model")
    calibrator = payload.get("calibrator")
    if calibration is None and calibrator is not None:
        calibration = calibrator
    return AnomalyModel(
        parameter_detectors=payload["parameter_detectors"],
        family_detectors=families,
        feature_columns=payload["feature_columns"],
        train_lots=payload["train_lots"],
        validation_lots=payload["validation_lots"],
        test_lots=payload["test_lots"],
        weights=payload.get("weights", {}),
        fusion_model=fusion,
        calibration_model=calibration,
        calibrator=calibrator,
        calibration_method=payload.get("calibration_method", "none_legacy"),
        config_hash=payload["config_hash"],
        as_of_h=payload["as_of_h"],
        reference_version=payload.get("reference_version", "phase4_fusion_safeguard"),
        validation_positive_count=int(payload.get("validation_positive_count", 0)),
        validation_positive_prevalence=float(payload.get("validation_positive_prevalence", 0.0)),
    )


def _label_column(labels: pd.DataFrame, horizon_h: float = 168.0) -> str | None:
    def hkey(h: float) -> str:
        return str(int(h)) if float(h).is_integer() else str(h).replace(".", "_")
    candidates = [f"future_defective_{hkey(horizon_h)}h", "future_defective_168h", "latent_defect_label"]
    for c in candidates:
        if c in labels.columns:
            return c
    return None


def fit_anomaly(
    input_path: str | Path,
    artifact_path: str | Path | None = None,
    as_of_h: float = 24.0,
    config_path: str | Path | None = None,
    target_horizon_h: float = 168.0,
) -> tuple[AnomalyModel, dict]:
    config = load_yaml(config_path or (PROJECT_ROOT / "configs" / "models.yaml"))
    seed = int(config.get("runtime", {}).get("random_seed", 20260831))
    seed_everything(seed)

    raw = _read_table(input_path)
    dense = _read_dense_table(input_path)
    d = _lot_robust(_wide_state(raw, as_of_h))
    train_lots, val_lots, test_lots = _split_lots(d)
    train = d[d.lot_id.astype(str).isin(train_lots)].copy()
    if "defect_state" in train.columns:
        train = train[train.defect_state.astype(str).str.upper().eq("HEALTHY")]

    comp, comp_features = _build_component_matrix(d)
    train_parts = comp[comp.lot_id.astype(str).isin(train_lots)].copy()
    if "defect_state" in train_parts.columns:
        train_parts = train_parts[train_parts.defect_state.astype(str).str.upper().eq("HEALTHY")]

    family_detectors: dict[str, FamilyDetector] = {}
    for family, sub in train_parts.groupby("component_family", dropna=False):
        X = sub[comp_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        if len(X) >= 50:
            family_seed = int.from_bytes(hashlib.blake2b(str(family).encode("utf-8"), digest_size=4).digest(), "big")
            fd = _fit_family_detector(X, config, seed + family_seed % 100000)
            fd.feature_columns = comp_features
            family_detectors[str(family)] = fd

    parameter_detectors: dict[str, IsolationForest] = {}
    for param, sub in train.groupby("parameter", dropna=False):
        cols = [c for c in DEFAULT_PARAMETER_FEATURES if c in sub.columns]
        X = sub[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        if len(X) >= 50:
            parameter_detectors[str(param)] = IsolationForest(
                n_estimators=175,
                random_state=seed,
                n_jobs=int(config["runtime"].get("n_jobs", -1)),
                contamination="auto",
            ).fit(X)

    model = AnomalyModel(
        parameter_detectors=parameter_detectors,
        family_detectors=family_detectors,
        feature_columns=comp_features,
        train_lots=train_lots,
        validation_lots=val_lots,
        test_lots=test_lots,
        weights=dict(config.get("anomaly", {}).get("ensemble", {}).get("fusion", {}).get("weights", {})),
        fusion_model=None,
        calibration_model=None,
        calibrator=None,
        calibration_method="max_evidence_sparse_validation",
        config_hash=stable_config_hash(config),
        as_of_h=as_of_h,
    )

    labels = _read_labels(input_path)
    if labels is None and "part_id" in raw.columns:
        wanted = [c for c in ["part_id", "future_defective_168h", "latent_defect_label"] if c in raw.columns]
        if len(wanted) >= 2:
            labels = raw[wanted].drop_duplicates("part_id")

    val_rows = d[d.lot_id.astype(str).isin(val_lots)].copy()
    val_evidence = _evidence_features_for_part(val_rows, model, dense=dense, as_of_h=as_of_h)
    label_col = _label_column(labels, target_horizon_h) if labels is not None else None
    
    if labels is not None and label_col is not None:
        lab = labels[["part_id", label_col]].drop_duplicates("part_id").copy()
        lab["part_id"] = lab["part_id"].astype(str)
        val_evidence = val_evidence.merge(lab, on="part_id", how="left")
        y = pd.to_numeric(val_evidence[label_col], errors="coerce")
        y_clean = y.dropna().astype(int)
        pos_count = int((y_clean == 1).sum())
        prevalence = float(pos_count / max(len(y_clean), 1))
        model.validation_positive_count = pos_count
        model.validation_positive_prevalence = prevalence
        
        # Sparse-positive safeguard: under label scarcity, don't trust learned fusion
        sparse_validation = pos_count < 30 or prevalence < 0.02
        if sparse_validation:
            model.fusion_model = None
            model.calibration_model = None
            model.calibrator = None
            model.calibration_method = "max_evidence_sparse_validation"
        elif y.notna().sum() >= 2 and y.dropna().nunique() >= 2:
            fusion, calibration, method = _fit_fusion_models(model, val_evidence, y.to_numpy())
            model.fusion_model = fusion
            model.calibration_model = calibration
            model.calibrator = calibration
            model.calibration_method = method

    metrics = {
        "train_lots": len(train_lots),
        "validation_lots": len(val_lots),
        "test_lots": len(test_lots),
        "train_rows": len(train),
        "multivariate_train_parts": len(train_parts),
        "multivariate_feature_count": len(comp_features),
        "parameter_detector_count": len(parameter_detectors),
        "family_detector_count": len(family_detectors),
        "calibration": model.calibration_method,
        "fusion_model": type(model.fusion_model).__name__ if model.fusion_model is not None else "ConservativeMaxEvidence",
        "validation_label_column": label_col,
        "validation_positive_count": int(model.validation_positive_count),
        "validation_positive_prevalence": float(model.validation_positive_prevalence),
    }
    if artifact_path:
        save_anomaly_model(model, artifact_path, training_data=input_path)
    return model, metrics


def score_anomaly(
    model: AnomalyModel,
    input_data: str | Path | pd.DataFrame,
    as_of_h: float = 24.0,
    output_path: str | Path | None = None,
    calibrate: bool = True,
) -> pd.DataFrame:
    raw = _read_table(input_data) if not isinstance(input_data, pd.DataFrame) else input_data.copy()
    dense = None if isinstance(input_data, pd.DataFrame) else _read_dense_table(input_data)
    d = _lot_robust(_wide_state(raw, as_of_h))
    part = _evidence_features_for_part(d, model, dense=dense, as_of_h=as_of_h)

    X = part[EVIDENCE_COLUMNS].to_numpy(float)
    # Robust max-evidence across unsupervised population, temporal, and multivariate channels
    max_evidence = np.max(X[:, [1, 2, 3]], axis=1)
    
    if model.calibration_method == "max_evidence_sparse_validation" or model.fusion_model is None:
        raw_ensemble = np.clip(max_evidence, 0.0, 1.0)
        fused = raw_ensemble.copy()
    else:
        try:
            fused = model.fusion_model.predict_proba(X)[:, 1]
            raw_ensemble = np.maximum(fused, max_evidence)
        except Exception:
            raw_ensemble = np.clip(max_evidence, 0.0, 1.0)
            fused = raw_ensemble.copy()
            
    part["base_anomaly_score"] = raw_ensemble

    calibrated = fused.copy()
    calibration_active = bool(calibrate and model.calibration_model is not None)
    if calibration_active:
        try:
            calibrated = np.asarray(model.calibration_model.predict(fused), dtype=float)
        except Exception:
            calibrated = fused.copy()

    part["anomaly_score"] = raw_ensemble
    part["fusion_risk_score"] = fused
    part["calibrated_risk_score"] = calibrated if calibration_active else raw_ensemble
    part["anomaly_risk"] = part["calibrated_risk_score"]
    part["anomaly_support_count"] = (
        (part["absolute_limits"] >= 0.65).astype(int)
        + (part["robust_population"] >= 0.65).astype(int)
        + (part["temporal_deviation"] >= 0.65).astype(int)
        + (part["multivariate"] >= 0.65).astype(int)
    )
    part["absolute_violation"] = part["absolute_limits"] >= 1.0
    part["population_evidence"] = part["robust_population"]
    part["temporal_evidence"] = part["temporal_deviation"]
    part["multivariate_component_score"] = part["multivariate"]
    part["multivariate_parameter_evidence"] = part.get("parameter_multivariate", 0.0)
    part["parameter_isolation_evidence"] = part.get("parameter_multivariate", 0.0)
    part["anomaly_risk_level"] = np.select(
        [part.anomaly_score >= 0.80, part.anomaly_score >= 0.60, part.anomaly_score >= 0.35],
        ["HIGH", "MODERATE", "LOW"], default="NORMAL"
    )
    part["calibration_enabled"] = calibration_active or model.calibration_method == "logistic_validation"
    part["calibration_method"] = model.calibration_method
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(p, index=False)
    return part


def aggregate_part_anomaly(scored: pd.DataFrame) -> pd.DataFrame:
    if "part_id" not in scored or "anomaly_score" not in scored:
        raise ValueError("Scored data must contain part_id and anomaly_score.")
    keys = [c for c in ["part_id", "lot_id", "component_family", "part_type", "component_type"] if c in scored.columns]
    agg = {"anomaly_score": ("anomaly_score", "max"), "mean_anomaly_score": ("anomaly_score", "mean")}
    for src, dst in [
        ("high_safe_likelihood", "high_safe_likelihood"),
        ("measurement_anomaly_likelihood", "measurement_anomaly_likelihood"),
    ]:
        if src in scored.columns:
            agg[dst] = (src, "max")
    return scored.groupby(keys, dropna=False).agg(**agg).reset_index()


def benchmark_anomaly(scored: pd.DataFrame, labels: pd.DataFrame | str | Path, split: str | None = "test") -> dict:
    d = scored.copy()
    p = Path(labels) if isinstance(labels, (str, Path)) else None
    if p and p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(p) as con:
            lab = pd.read_sql_query("SELECT * FROM labels", con)
    elif p:
        lab = pd.read_csv(p, low_memory=False)
    else:
        lab = labels.copy()
    lab = lab.drop_duplicates("part_id")
    cols = [c for c in [
        "part_id", "latent_defect_label", "future_defective_168h", "defect_state",
        "high_but_safe", "measurement_only_anomaly", "split",
    ] if c in lab.columns]
    add_cols = [c for c in cols if c not in d.columns and c != "part_id"]
    if add_cols:
        d = d.merge(lab[["part_id"] + add_cols], on="part_id", how="left")
    if split and "split" in d.columns:
        d = d[d.split.astype(str).eq(split)].copy()
    label_col = _label_column(d, 168.0)
    if label_col is None:
        raise ValueError("No future-defect label available for Module-A benchmark.")
    y = pd.to_numeric(d[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    outputs = {}
    channels = {
        "absolute_limits": "absolute_limits",
        "robust_population": "robust_population",
        "temporal": "temporal_deviation",
        "multivariate": "multivariate",
        "ensemble_raw": "base_anomaly_score",
        "fusion": "fusion_risk_score",
        "calibrated": "calibrated_risk_score",
    }
    for name, col in channels.items():
        if col not in d.columns:
            continue
        score = pd.to_numeric(d[col], errors="coerce").fillna(0).to_numpy(float)
        pred = (score >= 0.60).astype(int)
        outputs[name] = {
            "pr_auc": float(average_precision_score(y, score)) if y.sum() and y.sum() < len(y) else None,
            "roc_auc": float(roc_auc_score(y, score)) if y.sum() and y.sum() < len(y) else None,
            "latent_defect_recall": float(recall_score(y, pred, zero_division=0)),
            "false_negative_rate": float(((y == 1) & (pred == 0)).sum() / max(y.sum(), 1)),
            "precision": float(precision_score(y, pred, zero_division=0)),
        }
    return {"split": split, "rows": int(len(d)), "positive_count": int(y.sum()), "models": outputs}


def evaluate_anomaly(scored: pd.DataFrame, labels: pd.DataFrame | str | Path | None = None, threshold: float = 0.60, split: str | None = None) -> dict:
    d = scored.copy()
    if labels is not None:
        p = Path(labels) if isinstance(labels, (str, Path)) else None
        if p and p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            with sqlite3.connect(p) as con:
                lab = pd.read_sql_query("SELECT * FROM labels", con)
        elif p:
            lab = pd.read_csv(p, low_memory=False)
        else:
            lab = labels.copy()
        attach = [c for c in [
            "part_id", "latent_defect_label", "future_defective_168h", "defect_state",
            "high_but_safe", "measurement_only_anomaly", "split",
        ] if c in lab.columns]
        missing = [c for c in attach if c not in d.columns]
        if missing:
            d = d.merge(lab[["part_id"] + [c for c in missing if c != "part_id"]].drop_duplicates("part_id"), on="part_id", how="left")
    if split and "split" in d.columns:
        d = d[d["split"].astype(str).eq(split)].copy()
    label_col = _label_column(d, 168.0)
    if label_col is None:
        raise ValueError("future_defective_168h or latent_defect_label required")
    y = pd.to_numeric(d[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    s = pd.to_numeric(d["anomaly_score"], errors="coerce").fillna(0).to_numpy()
    pred = (s >= threshold).astype(int)
    return {
        "rows": int(len(d)), "positive_count": int(y.sum()),
        "false_negative_rate": float(((y == 1) & (pred == 0)).sum() / max(y.sum(), 1)),
        "latent_defect_recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, s)) if y.sum() and y.sum() < len(y) else None,
        "roc_auc": float(roc_auc_score(y, s)) if y.sum() and y.sum() < len(y) else None,
        "high_safe_false_reject_rate": float(((d.get("high_but_safe", pd.Series(0, index=d.index)).astype(int).eq(1)) & (pred == 1)).sum() /
                                           max((d.get("high_but_safe", pd.Series(0, index=d.index)).astype(int).eq(1)).sum(), 1)),
        "measurement_artifact_false_flag_rate": float(((d.get("measurement_only_anomaly", pd.Series(0, index=d.index)).astype(int).eq(1)) & (pred == 1)).sum() /
                                                        max((d.get("measurement_only_anomaly", pd.Series(0, index=d.index)).astype(int).eq(1)).sum(), 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Module A anomaly screening")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("input")
    t.add_argument("artifact")
    t.add_argument("--as-of", type=float, default=24.0)
    t.add_argument("--target-horizon", type=float, default=168.0)
    s = sub.add_parser("score")
    s.add_argument("artifact")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--as-of", type=float, default=24.0)
    s.add_argument("--no-calibration", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("scored")
    e.add_argument("labels")
    e.add_argument("output")
    e.add_argument("--threshold", type=float, default=0.60)
    e.add_argument("--split", default=None)
    b = sub.add_parser("benchmark")
    b.add_argument("scored")
    b.add_argument("labels")
    b.add_argument("output")
    b.add_argument("--split", default="test")
    a = p.parse_args()
    if a.cmd == "train":
        _, m = fit_anomaly(a.input, a.artifact, a.as_of, target_horizon_h=a.target_horizon)
        print(json.dumps(m, indent=2))
    elif a.cmd == "score":
        m = load_anomaly_model(a.artifact)
        d = score_anomaly(m, a.input, a.as_of, a.output, calibrate=not a.no_calibration)
        print(json.dumps({"rows": len(d), "output": a.output, "calibration": m.calibration_method}, indent=2))
    elif a.cmd == "evaluate":
        s = pd.read_csv(a.scored, low_memory=False)
        m = evaluate_anomaly(s, a.labels, a.threshold, a.split)
        Path(a.output).write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(json.dumps(m, indent=2))
    else:
        s = pd.read_csv(a.scored, low_memory=False)
        m = benchmark_anomaly(s, a.labels, a.split)
        Path(a.output).write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()



# Backward-compatible public test helpers retained for repository consumers.
def _canonicalize_state(df: pd.DataFrame, as_of_h: float = 24.0) -> pd.DataFrame:
    return _wide_state(df, as_of_h)


def _robust_lot_features(df: pd.DataFrame) -> pd.DataFrame:
    return _lot_robust(df, "value_asof")
