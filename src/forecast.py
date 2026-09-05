"""Module B — Universal, Leakage-Safe Degradation Forecasting.

Predicts continuous parametric drift specifically for the 168h target horizon, 
incorporating initial drift velocity and strict baseline tournament gating.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    PROJECT_ROOT,
    load_yaml,
    seed_everything,
    stable_config_hash,
    validate_artifact_compatibility,
    write_artifact_manifest,
)

FORMAT = "forecast_model_v3_universal"
LEGACY_FORMATS = {"forecast_model_v2", "forecast_model_v1"}
DEFAULT_TARGET_HORIZONS = (48.0, 96.0, 168.0, 240.0, 300.0)

CATEGORICAL = [
    "component_family", "part_type", "component_type", "parameter", "semantic_type",
    "physical_quantity", "unit", "profile_id",
]
NUMERIC_CONTEXT = [
    "temperature_C", "burnin_temperature_C", "voltage_V", "current_A",
]


def _horizon_token(h: float) -> str:
    x = float(h)
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:g}".replace(".", "_")


def horizon_column(prefix: str, h: float) -> str:
    return f"{prefix}_{_horizon_token(h)}h"


def _value_columns(df: pd.DataFrame) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for c in df.columns:
        m = re.fullmatch(r"value_(\d+(?:\.\d+)?)h", str(c))
        if m: out.append((float(m.group(1)), str(c)))
    return sorted(out)


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv": return pd.read_csv(p, low_memory=False)
    if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(p) as con:
            for table in ("module_B_drift_full", "module_B_train", "component_parameter_state"):
                try: return pd.read_sql_query(f"SELECT * FROM {table}", con)
                except Exception: pass
    raise ValueError(f"Unsupported input: {p}")


def _split_by_lot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns:
        labels = set(df["split"].dropna().astype(str))
        if {"train", "val", "test"}.issubset(labels):
            return tuple(df[df["split"].astype(str).eq(s)].copy() for s in ("train", "val", "test"))
    lots = sorted(df["lot_id"].astype(str).unique())
    n = len(lots)
    nt = max(1, int(round(0.70 * n))); nv = max(1, int(round(0.15 * n))) if n >= 7 else 0
    sets = (lots[:nt], lots[nt:nt + nv], lots[nt + nv:])
    return tuple(df[df["lot_id"].astype(str).isin(set(x))].copy() for x in sets)


def _coerce_target_horizons(value: Sequence[float] | str | None, df: pd.DataFrame) -> list[float]:
    if value is not None:
        if isinstance(value, str): vals = [float(x.strip()) for x in value.split(",") if x.strip()]
        else: vals = [float(x) for x in value]
        return sorted(set(x for x in vals if x >= 0))
    detected: list[float] = []
    for c in df.columns:
        m = re.fullmatch(r"(?:target|value)_(\d+(?:\.\d+)?)h", str(c))
        if m: detected.append(float(m.group(1)))
    vals = sorted(set(detected) & set(DEFAULT_TARGET_HORIZONS))
    return vals or list(DEFAULT_TARGET_HORIZONS)


def _available_points(row: pd.Series, as_of_h: float) -> tuple[np.ndarray, np.ndarray]:
    pts: list[tuple[float, float]] = []
    for t, c in _value_columns(row.to_frame().T):
        if t > float(as_of_h): continue
        v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
        if pd.notna(v): pts.append((float(t), float(v)))
    pts.sort()
    if not pts and "forecast_origin_h" in row.index and pd.notna(row.get("forecast_origin_h")):
        origin = float(row["forecast_origin_h"])
        if "value_asof" in row.index and pd.notna(row.get("value_asof")): pts.append((origin, float(row["value_asof"])))
    if not pts and "value" in row.index and pd.notna(row.get("value")):
        pts.append((float(row.get("time_h", as_of_h)), float(row["value"])))
    return np.asarray([p[0] for p in pts], dtype=float), np.asarray([p[1] for p in pts], dtype=float)


def _safe_relative(delta: float, base: float) -> float:
    denom = abs(base)
    return float(delta / denom) if np.isfinite(base) and denom > 1e-12 else 0.0


def _robust_lot_z_from_stats(row: pd.Series, value: float, time_h: float, stats: Mapping[tuple[Any, ...], tuple[float, float, float]] | None) -> float:
    if not stats or "lot_id" not in row.index or "parameter" not in row.index: return float("nan")
    keys = [str(row.get("lot_id")), str(row.get("parameter"))]
    if "component_family" in row.index: keys.append(str(row.get("component_family")))
    key = tuple(keys + [float(time_h)])
    if key not in stats:
        key2 = (str(row.get("lot_id")), str(row.get("parameter")), float(time_h))
        if key2 not in stats: return float("nan")
        key = key2
    median, scale, _ = stats[key]
    return float((value - median) / max(scale, 1e-12))


def _lot_reference_stats(df: pd.DataFrame, as_of_h: float) -> dict[tuple[Any, ...], tuple[float, float, float]]:
    if df.empty: return {}
    records: list[dict[str, Any]] = []
    value_columns: list[tuple[float, str]] = []
    for col in df.columns:
        match = re.fullmatch(r"value_([0-9]+(?:\.[0-9]+)?)h", str(col))
        if match:
            time_h = float(match.group(1))
            if time_h <= float(as_of_h): value_columns.append((time_h, col))
    if not value_columns: return {}
    value_columns.sort(key=lambda x: x[0])

    for _, row in df.iterrows():
        chosen_time: float | None = None
        chosen_value: float | None = None
        for time_h, col in reversed(value_columns):
            value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.notna(value):
                chosen_time = float(time_h); chosen_value = float(value); break
        if chosen_time is None or chosen_value is None: continue
        records.append({"lot_id": row.get("lot_id"), "component_family": row.get("component_family"), "parameter": row.get("parameter"), "_time": chosen_time, "_value": chosen_value})
    if not records: return {}

    work = pd.DataFrame(records)
    stats: dict[tuple[Any, ...], tuple[float, float, float]] = {}
    group_columns = ["lot_id", "component_family", "parameter", "_time"]
    for key, group in work.groupby(group_columns, dropna=False):
        values = pd.to_numeric(group["_value"], errors="coerce").dropna()
        if len(values) == 0: continue
        median = float(values.median()); mad = float(np.median(np.abs(values.to_numpy(dtype=float) - median)))
        robust_scale = max(1.4826 * mad, 1e-9); stats[key] = (median, mad, robust_scale)
    return stats


def _series_features(row: pd.Series, as_of_h: float) -> dict[str, float]:
    t, v = _available_points(row, as_of_h)
    out = {
        "last_value": np.nan, "previous_value": np.nan, "last_time_h": np.nan, "previous_time_h": np.nan,
        "last_delta": np.nan, "last_slope": np.nan, "last_normalized_drift": np.nan,
        "all_slope": np.nan, "all_normalized_drift": np.nan, "all_acceleration": np.nan, "all_curvature": np.nan,
        "last_acceleration": np.nan, "last_curvature": np.nan, "trajectory_span_h": np.nan,
        "smoothed_baseline_deviation": np.nan, "initial_drift_velocity": np.nan
    }
    if len(v) == 0: return out
    out["last_value"] = float(v[-1]); out["last_time_h"] = float(t[-1])
    if len(v) >= 2:
        dt = float(t[-1] - t[-2]); delta = float(v[-1] - v[-2]); slope = delta / dt if abs(dt) > 1e-12 else 0.0
        out["previous_value"] = float(v[-2]); out["previous_time_h"] = float(t[-2]); out["last_delta"] = delta
        out["last_slope"] = slope; out["last_normalized_drift"] = _safe_relative(delta, float(v[-2]))
    if len(v) >= 3:
        dt1 = float(t[-2] - t[-3]); dt2 = float(t[-1] - t[-2])
        s1 = float((v[-2] - v[-3]) / dt1) if abs(dt1) > 1e-12 else 0.0
        s2 = float((v[-1] - v[-2]) / dt2) if abs(dt2) > 1e-12 else 0.0
        out["last_acceleration"] = (s2 - s1) / max((dt1 + dt2) / 2.0, 1e-12); out["last_curvature"] = s2 - s1
    if len(v) >= 2:
        try: coef = np.polyfit(t, v, 1); out["all_slope"] = float(coef[0])
        except Exception: out["all_slope"] = 0.0
        out["all_normalized_drift"] = _safe_relative(float(v[-1] - v[0]), float(v[0])); out["trajectory_span_h"] = float(t[-1] - t[0])
    
    # NEW FEATURE: Explicit early drift velocity
    v0 = next((float(v[i]) for i, tt in enumerate(t) if abs(tt) < 1e-9), np.nan)
    v24 = next((float(v[i]) for i, tt in enumerate(t) if abs(tt - 24.0) < 1e-9), np.nan)
    if np.isfinite(v0) and np.isfinite(v24):
        out["initial_drift_velocity"] = (v24 - v0) / 24.0
    else:
        out["initial_drift_velocity"] = out["last_slope"]
        
    return out


def _feature_row(df: pd.DataFrame, row: pd.Series, *, as_of_h: float, target_horizon_h: float, lot_stats: Mapping[tuple[Any, ...], tuple[float, float, float]] | None = None) -> dict[str, Any]:
    feat = _series_features(row, as_of_h)
    feat["forecast_origin_h"] = float(as_of_h); feat["target_horizon_h"] = float(target_horizon_h); feat["relative_horizon_h"] = float(target_horizon_h - (feat["last_time_h"] if np.isfinite(feat["last_time_h"]) else as_of_h))
    feat["baseline_value_0h"] = pd.to_numeric(pd.Series([row.get("value_0h", np.nan)]), errors="coerce").iloc[0]
    feat["precursor_strength"] = pd.to_numeric(pd.Series([row.get("precursor_strength", np.nan)]), errors="coerce").iloc[0]
    feat["precursor_effect"] = pd.to_numeric(pd.Series([row.get("precursor_effect", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(feat["last_value"]) and np.isfinite(feat["last_time_h"]): feat["last_lot_robust_z"] = _robust_lot_z_from_stats(row, feat["last_value"], feat["last_time_h"], lot_stats)
    else: feat["last_lot_robust_z"] = np.nan
    for c in CATEGORICAL: feat[c] = row.get(c, np.nan)
    for c in NUMERIC_CONTEXT: feat[c] = row.get(c, np.nan)
    return feat


UNIVERSAL_NUMERIC = [
    "last_value", "previous_value", "last_time_h", "previous_time_h",
    "last_delta", "last_slope", "last_normalized_drift",
    "all_slope", "all_normalized_drift", "last_acceleration", "last_curvature",
    "all_acceleration", "all_curvature", "trajectory_span_h",
    "smoothed_baseline_deviation", "last_lot_robust_z",
    "forecast_origin_h", "target_horizon_h", "relative_horizon_h",
    "baseline_value_0h", "precursor_strength", "precursor_effect", "initial_drift_velocity"
] + NUMERIC_CONTEXT


@dataclass
class ForecastModel:
    primary: Pipeline
    ridge: Pipeline
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    target_horizon_h: float
    residual_scale: float
    conformal_quantile: float
    config_hash: str
    selected_model: str
    validation_metrics: dict[str, float]
    train_lots: list[str]
    validation_lots: list[str]
    test_lots: list[str]
    horizon_scaling_factor: float = 1.0
    conformal_quantiles: dict[str, float] | None = None
    training_horizons: list[float] | None = None
    as_of_h: float = 24.0
    reference_version: str = "phase-universal-168h-exclusive"


def _pre(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop")


def _ensure_frame(df: pd.DataFrame, as_of_h: float = 24.0) -> pd.DataFrame:
    d = df.copy()
    required = {"part_id", "lot_id"}
    missing = sorted(required - set(d.columns))
    if missing: raise ValueError(f"Forecast input missing: {missing}")
    vals = [t for t, _ in _value_columns(d) if t <= as_of_h]
    if not vals and "value" not in d.columns: raise ValueError(f"No observed value_<time>h columns at or before as_of={as_of_h}h")
    return d


def _build_examples(df: pd.DataFrame, horizons: Sequence[float], as_of_h: float) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []; targets: list[float] = []; meta: list[dict[str, Any]] = []
    lot_stats = _lot_reference_stats(df, as_of_h)
    target_candidates = [h for h in horizons if horizon_column("target", h) in df.columns]
    if not target_candidates: target_candidates = [h for h in horizons if horizon_column("value", h) in df.columns]
    if not target_candidates: raise ValueError("Forecast training requires observed target_<h>h or value_<h>h columns.")
    for idx, row in df.iterrows():
        baseline = pd.to_numeric(pd.Series([row.get("value_0h", np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(baseline) or abs(float(baseline)) <= 1e-12: continue
        for h in target_candidates:
            target_col = horizon_column("target", h) if horizon_column("target", h) in df.columns else horizon_column("value", h)
            y_abs = pd.to_numeric(pd.Series([row.get(target_col)]), errors="coerce").iloc[0]
            if pd.isna(y_abs): continue
            feat = _feature_row(df, row, as_of_h=as_of_h, target_horizon_h=h, lot_stats=lot_stats)
            rows.append(feat); targets.append(float(y_abs) / float(baseline))
            meta.append({"source_index": idx, "target_horizon_h": float(h), "baseline_value_0h": float(baseline), "target_abs": float(y_abs)})
    if not rows: raise ValueError("No valid supervised forecast examples after target/baseline validation.")
    return pd.DataFrame(rows), np.asarray(targets, dtype=float), pd.DataFrame(meta)


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    ok = np.isfinite(y) & np.isfinite(p)
    return {
        "n": int(ok.sum()), "mae": float(mean_absolute_error(y[ok], p[ok])) if ok.any() else None,
        "rmse": float(np.sqrt(mean_squared_error(y[ok], p[ok]))) if ok.any() else None,
    }


def _conformal(y: np.ndarray, p: np.ndarray, coverage: float = 0.90) -> float:
    r = np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float)); r = r[np.isfinite(r)]
    if len(r) == 0: return 0.0
    q = min(1.0, float(np.ceil((len(r) + 1) * coverage) / len(r)))
    return float(np.quantile(r, q))


def _predict_core(model: ForecastModel, d: pd.DataFrame, target_horizon_h: float) -> tuple[np.ndarray, np.ndarray]:
    lot_stats = _lot_reference_stats(d, model.as_of_h)
    feat_df = pd.DataFrame([
        _feature_row(d, row, as_of_h=float(row.get("forecast_origin_h", model.as_of_h)), target_horizon_h=target_horizon_h, lot_stats=lot_stats)
        for _, row in d.iterrows()
    ])
    for c in model.feature_columns:
        if c not in feat_df.columns: feat_df[c] = np.nan
    X = feat_df[model.feature_columns]
    primary = model.primary.predict(X); ridge = model.ridge.predict(X)
    return np.asarray(primary, dtype=float), np.asarray(ridge, dtype=float)


def save_forecast_model(model: ForecastModel, path: str | Path, training_data: str | Path | None = None) -> None:
    payload = {
        "format": FORMAT, "primary": model.primary, "ridge": model.ridge, "feature_columns": model.feature_columns,
        "numeric_columns": model.numeric_columns, "categorical_columns": model.categorical_columns, "target_horizon_h": model.target_horizon_h,
        "residual_scale": model.residual_scale, "conformal_quantile": model.conformal_quantile, "config_hash": model.config_hash,
        "selected_model": model.selected_model, "validation_metrics": model.validation_metrics, "train_lots": model.train_lots,
        "validation_lots": model.validation_lots, "test_lots": model.test_lots, "horizon_scaling_factor": model.horizon_scaling_factor,
        "conformal_quantiles": model.conformal_quantiles or {}, "training_horizons": model.training_horizons or [], "as_of_h": model.as_of_h, "reference_version": model.reference_version,
    }
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); joblib.dump(payload, p, compress=3)
    write_artifact_manifest(p, artifact_kind="forecast_model", config_hash=model.config_hash, training_data=training_data, extra={"format": FORMAT, "selected_model": model.selected_model, "training_horizons": model.training_horizons or []})


def load_forecast_model(path: str | Path, *, require_compatibility: bool = True) -> ForecastModel:
    p = Path(path)
    if require_compatibility: validate_artifact_compatibility(p, artifact_kind="forecast_model", require_manifest=True)
    payload = joblib.load(p)
    if not isinstance(payload, dict) or payload.get("format") not in {FORMAT, *LEGACY_FORMATS}: raise ValueError("Unsupported forecast artifact format")
    return ForecastModel(
        primary=payload["primary"], ridge=payload["ridge"], feature_columns=payload.get("feature_columns", []),
        numeric_columns=payload.get("numeric_columns", []), categorical_columns=payload.get("categorical_columns", []),
        target_horizon_h=float(payload.get("target_horizon_h", 168.0)), residual_scale=float(payload.get("residual_scale", 0.0)),
        conformal_quantile=float(payload.get("conformal_quantile", 0.0)), config_hash=payload.get("config_hash", ""),
        selected_model=payload.get("selected_model", "gradient_boosting"), validation_metrics=payload.get("validation_metrics", {}),
        train_lots=payload.get("train_lots", []), validation_lots=payload.get("validation_lots", []), test_lots=payload.get("test_lots", []),
        horizon_scaling_factor=float(payload.get("horizon_scaling_factor", 1.0)), conformal_quantiles=payload.get("conformal_quantiles", {}),
        training_horizons=payload.get("training_horizons", []), as_of_h=float(payload.get("as_of_h", 24.0)), reference_version=payload.get("reference_version", "phase-universal-tournament"),
    )


def fit_forecast(
    input_path: str | Path,
    artifact_path: str | Path | None = None,
    target_horizon_h: float = 168.0,
    config_path: str | Path | None = None,
    *,
    as_of_h: float = 24.0,
    target_horizons: Sequence[float] | str | None = None,
) -> tuple[ForecastModel, dict, pd.DataFrame]:
    cfg = load_yaml(config_path or PROJECT_ROOT / "configs" / "models.yaml")
    seed = int(cfg.get("runtime", {}).get("random_seed", 20260831))
    seed_everything(seed)
    raw = _ensure_frame(_read(input_path), as_of_h)
    
    # FIX: Strictly isolate training to the target horizon (168h) rather than pooling all horizons.
    horizons = [float(target_horizon_h)]
    
    train, val, test = _split_by_lot(raw)
    if len(test) == 0: raise ValueError("Forecast training data has no held-out test lots.")

    Xtr, ytr, meta_tr = _build_examples(train, horizons, as_of_h)
    Xv, yv, meta_v = _build_examples(val, horizons, as_of_h) if len(val) else (pd.DataFrame(columns=Xtr.columns), np.array([]), pd.DataFrame())

    numeric = [c for c in UNIVERSAL_NUMERIC if c in Xtr.columns]
    categorical = [c for c in CATEGORICAL if c in Xtr.columns]
    feature_cols = numeric + categorical
    if len(ytr) < 10: raise ValueError("Not enough supervised forecast examples to fit a model.")

    ridge = Pipeline([("pre", _pre(numeric, categorical)), ("model", Ridge(alpha=1.0))])
    hgb = Pipeline([
        ("pre", _pre(numeric, categorical)),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, random_state=seed,
        )),
    ])
    ridge.fit(Xtr[feature_cols], ytr); hgb.fit(Xtr[feature_cols], ytr)

    metrics: dict[str, dict[str, float | None]] = {}
    if len(yv):
        base_v = pd.to_numeric(meta_v["baseline_value_0h"], errors="coerce").to_numpy(float)
        yv_abs = pd.to_numeric(meta_v["target_abs"], errors="coerce").to_numpy(float)
        rp_rel = ridge.predict(Xv[feature_cols]); metrics["ridge"] = _metric(yv_abs, rp_rel * base_v)
        hp_rel = hgb.predict(Xv[feature_cols]); metrics["gradient_boosting"] = _metric(yv_abs, hp_rel * base_v)
        p_persist = Xv["last_value"].to_numpy(float); metrics["persistence"] = _metric(yv_abs, p_persist)
        slope = Xv["last_slope"].fillna(0.0).to_numpy(float); dt = Xv["relative_horizon_h"].to_numpy(float)
        p_linear = p_persist + slope * dt; metrics["linear"] = _metric(yv_abs, p_linear)

        # Composite score (MAE + RMSE) to heavily penalize large catastrophic misses (where Persistence fails)
        composite_scores = {
            "gradient_boosting": 0.5 * metrics["gradient_boosting"]["mae"] + 0.5 * metrics["gradient_boosting"]["rmse"],
            "ridge": 0.5 * metrics["ridge"]["mae"] + 0.5 * metrics["ridge"]["rmse"]
        }
        selected = min(["gradient_boosting", "ridge"], key=lambda k: composite_scores[k] if metrics[k]["mae"] is not None else float("inf"))
        
        baseline_composite = 0.5 * metrics["persistence"]["mae"] + 0.5 * metrics["persistence"]["rmse"]
        if composite_scores[selected] > baseline_composite: selected = "persistence"
    else: selected = "gradient_boosting"

    if selected == "ridge": chosen_rel = ridge.predict(Xv[feature_cols])
    elif selected == "gradient_boosting": chosen_rel = hgb.predict(Xv[feature_cols])
    else: chosen_rel = (Xv["last_value"] / base_v).to_numpy(float) if len(yv) else np.array([])
        
    chosen_val = chosen_rel * base_v if len(yv) else np.array([])
    residuals = np.abs(yv_abs - chosen_val) if len(yv) else np.array([])
    per_h: dict[str, float] = {}
    if len(yv):
        for h in horizons:
            m = np.isfinite(yv_abs) & np.isclose(meta_v["target_horizon_h"].to_numpy(float), h)
            if m.any(): per_h[_horizon_token(h)] = _conformal(yv_abs[m], chosen_val[m], 0.90)
    base_q = float(np.median(list(per_h.values()))) if per_h else _conformal(yv, chosen_val, 0.90) if len(yv) else 0.0
    rs = float(np.median(residuals)) if len(residuals) else 0.0

    model = ForecastModel(
        primary=hgb, ridge=ridge, feature_columns=feature_cols, numeric_columns=numeric, categorical_columns=categorical,
        target_horizon_h=float(target_horizon_h), residual_scale=rs, conformal_quantile=base_q, config_hash=stable_config_hash(cfg),
        selected_model=selected, validation_metrics={k: float(v["mae"]) for k, v in metrics.items() if v.get("mae") is not None},
        train_lots=sorted(train.lot_id.astype(str).unique()), validation_lots=sorted(val.lot_id.astype(str).unique()), test_lots=sorted(test.lot_id.astype(str).unique()),
        horizon_scaling_factor=max(base_q, 1e-12), conformal_quantiles=per_h, training_horizons=horizons, as_of_h=float(as_of_h), reference_version="phase-universal-168h-exclusive",
    )

    val_pred = predict_forecast(model, val, target_horizon=float(target_horizon_h)) if len(val) else pd.DataFrame()
    test_pred = predict_forecast(model, test, target_horizon=float(target_horizon_h)) if len(test) else pd.DataFrame()
    test_eval = evaluate_forecast(test, test_pred, target_horizon=float(target_horizon_h)) if len(test) else {}
    result = {
        "train_rows": int(len(train)), "validation_rows": int(len(val)), "test_rows": int(len(test)),
        "train_examples": int(len(Xtr)), "validation_examples": int(len(Xv)),
        "train_lots": len(model.train_lots), "validation_lots": len(model.validation_lots), "test_lots": len(model.test_lots),
        "training_horizons": horizons, "selected_model": selected, "validation_metrics": metrics, "heldout_test": test_eval, "conformal_quantiles": per_h,
    }
    if artifact_path: save_forecast_model(model, artifact_path, input_path)
    return model, result, pd.concat([val_pred.assign(_split="val"), test_pred.assign(_split="test")], ignore_index=True)


def predict_forecast(
    model: ForecastModel,
    data: str | Path | pd.DataFrame,
    output_path: str | Path | None = None,
    *,
    target_horizon: float | None = None,
    forecast_origin_h: float | None = None,
) -> pd.DataFrame:
    raw = _ensure_frame(_read(data) if not isinstance(data, pd.DataFrame) else data.copy(), model.as_of_h)
    d = raw.copy()
    target = float(model.target_horizon_h if target_horizon is None else target_horizon)
    if target < 0: raise ValueError("target_horizon must be non-negative")
    if forecast_origin_h is not None: d["forecast_origin_h"] = float(forecast_origin_h)
    elif "forecast_origin_h" not in d.columns:
        origins = []
        for _, row in d.iterrows():
            tt, _ = _available_points(row, model.as_of_h)
            origins.append(float(tt[-1]) if len(tt) else float(model.as_of_h))
        d["forecast_origin_h"] = origins

    primary_rel, ridge_rel = _predict_core(model, d, target)
    baselines = pd.to_numeric(d.get("value_0h", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    primary = np.asarray(primary_rel, dtype=float) * baselines; ridge = np.asarray(ridge_rel, dtype=float) * baselines

    baseline_rows = []
    for _, row in d.iterrows():
        t, v = _available_points(row, float(row.get("forecast_origin_h", model.as_of_h)))
        if len(v) >= 1:
            origin = float(t[-1]); last = float(v[-1])
            if len(v) >= 2: dt = float(t[-1] - t[-2]); slope = float((v[-1] - v[-2]) / dt) if abs(dt) > 1e-12 else 0.0
            else: slope = 0.0
            persistence = last; linear = last + slope * (target - origin)
        else: persistence = np.nan; linear = np.nan
        baseline_rows.append((persistence, linear))
    baseline_arr = np.asarray(baseline_rows, dtype=float) if baseline_rows else np.empty((0, 2))

    if model.selected_model == "ridge": selected = ridge
    elif model.selected_model == "gradient_boosting": selected = primary
    elif model.selected_model == "persistence": selected = baseline_arr[:, 0] if len(baseline_arr) else primary
    else: selected = baseline_arr[:, 1] if len(baseline_arr) else primary

    q = None
    if model.conformal_quantiles: q = model.conformal_quantiles.get(_horizon_token(target))
    if q is None:
        base_h = 168.0
        if model.conformal_quantiles and "168" in model.conformal_quantiles: base = float(model.conformal_quantiles["168"])
        else: base = float(model.conformal_quantile)
        q = base * math.sqrt(max(target, 1.0) / base_h)
    q = float(q)

    out = d.copy()
    out["target_horizon_h"] = target; out["forecast_origin_h"] = d["forecast_origin_h"].astype(float); out["forecast_relative_horizon_h"] = target - out["forecast_origin_h"]
    out["selected_forecast_model"] = model.selected_model; out["prediction_persistence"] = baseline_arr[:, 0] if len(baseline_arr) else np.array([])
    out["prediction_linear"] = baseline_arr[:, 1] if len(baseline_arr) else np.array([])
    out["prediction_ridge"] = ridge; out["prediction_gradient_boosting"] = primary
    out["prediction_" + _horizon_token(target) + "h"] = selected; out["prediction_168h"] = selected if abs(target - 168.0) < 1e-9 else np.nan
    out["prediction_lower"] = selected - q; out["prediction_upper"] = selected + q
    out["prediction_interval_width"] = 2.0 * q; out["conformal_half_width"] = q

    limit = None
    for c in ["absolute_limit_upper", "spec_upper", "upper_limit", "engineering_limit_upper"]:
        if c in out.columns:
            limit = pd.to_numeric(out[c], errors="coerce"); break
    if limit is not None:
        z = (limit.to_numpy(float) - selected) / max(model.residual_scale, q, 1e-12)
        out["limit_exceedance_probability_proxy"] = 1.0 / (1.0 + np.exp(np.clip(z, -30, 30))); out["predicted_limit_exceedance"] = (out["prediction_upper"] > limit).astype(int)
    else: out["limit_exceedance_probability_proxy"] = np.nan; out["predicted_limit_exceedance"] = 0

    if "observable_safety_slope" in out.columns:
        ss = pd.to_numeric(out["observable_safety_slope"], errors="coerce")
        slope = pd.Series([_series_features(r, float(r.get("forecast_origin_h", model.as_of_h))).get("last_slope", np.nan) for _, r in out.iterrows()], index=out.index, dtype=float)
        out["safety_slope_excess"] = slope - ss; out["safety_slope_flag"] = (slope.abs() > ss.abs()).fillna(False).astype(int)
    else: out["safety_slope_excess"] = np.nan; out["safety_slope_flag"] = 0

    scale = np.maximum(np.abs(selected), 1e-9); out["uncertainty_score"] = np.clip(q / scale, 0.0, 1.0); out["failure_risk"] = out["limit_exceedance_probability_proxy"]
    if output_path: p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True); out.to_csv(p, index=False)
    return out


def _persist(df: pd.DataFrame, horizon_h: float | None = None) -> np.ndarray:
    preds = []
    for _, row in df.iterrows():
        _, v = _available_points(row, float(row.get("forecast_origin_h", 24.0))); preds.append(float(v[-1]) if len(v) else np.nan)
    return np.asarray(preds, dtype=float)


def _linear(df: pd.DataFrame, horizon_h: float = 168.0) -> np.ndarray:
    preds = []; h = float(horizon_h)
    for _, row in df.iterrows():
        t, v = _available_points(row, float(row.get("forecast_origin_h", 24.0)))
        if len(v) == 0: preds.append(np.nan); continue
        if len(v) < 2: preds.append(float(v[-1])); continue
        dt = float(t[-1] - t[-2]); slope = float((v[-1] - v[-2]) / dt) if abs(dt) > 1e-12 else 0.0
        preds.append(float(v[-1] + slope * (h - float(t[-1]))))
    return np.asarray(preds, dtype=float)


def evaluate_forecast(reference: pd.DataFrame, pred: pd.DataFrame, *, target_horizon: float = 168.0) -> dict[str, Any]:
    target_col = horizon_column("target", target_horizon)
    if target_col not in reference.columns: target_col = horizon_column("value", target_horizon)
    if target_col not in reference.columns: target_col = "target_168h"
    pred_col = "prediction_" + _horizon_token(target_horizon) + "h"
    if pred_col not in pred.columns: pred_col = "prediction_168h"
    y = pd.to_numeric(reference[target_col], errors="coerce").to_numpy(float); p = pd.to_numeric(pred[pred_col], errors="coerce").to_numpy(float)
    m = np.isfinite(y) & np.isfinite(p); out = _metric(y[m], p[m])
    if "prediction_lower" in pred.columns and "prediction_upper" in pred.columns:
        lo = pd.to_numeric(pred["prediction_lower"], errors="coerce").to_numpy(float); hi = pd.to_numeric(pred["prediction_upper"], errors="coerce").to_numpy(float)
        ok = m & np.isfinite(lo) & np.isfinite(hi)
        out["conformal_coverage"] = float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else None
        out["mean_interval_width"] = float(np.mean(hi[ok] - lo[ok])) if ok.any() else None
    out["target_horizon_h"] = float(target_horizon)
    return out


def benchmark_forecast(reference: pd.DataFrame, pred: pd.DataFrame, *, target_horizon: float = 168.0) -> pd.DataFrame:
    target_col = horizon_column("target", target_horizon) if horizon_column("target", target_horizon) in reference.columns else horizon_column("value", target_horizon)
    if target_col not in reference.columns: target_col = "target_168h"
    y = pd.to_numeric(reference[target_col], errors="coerce").to_numpy(float); m = np.isfinite(y)
    names = [("ridge", "prediction_ridge"), ("gradient_boosting", "prediction_gradient_boosting"), ("persistence", "prediction_persistence"), ("linear", "prediction_linear"), ("selected", "prediction_" + _horizon_token(target_horizon) + "h")]
    rows: list[dict[str, Any]] = []
    for name, col in names:
        if col not in pred.columns: continue
        p = pd.to_numeric(pred[col], errors="coerce").to_numpy(float); z = m & np.isfinite(p)
        if z.any(): rows.append({"model": name, "target_horizon_h": float(target_horizon), "n": int(z.sum()), "mae": float(mean_absolute_error(y[z], p[z])), "rmse": float(np.sqrt(mean_squared_error(y[z], p[z])))})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Universal horizon-conditioned Module-B forecast")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("input"); t.add_argument("artifact"); t.add_argument("--horizon", type=float, default=168.0); t.add_argument("--as-of", type=float, default=24.0); t.add_argument("--target-horizons", default=None)
    p = sub.add_parser("predict"); p.add_argument("artifact"); p.add_argument("input"); p.add_argument("output"); p.add_argument("--target-horizon", type=float, default=None); p.add_argument("--forecast-origin-h", type=float, default=None)
    e = sub.add_parser("evaluate"); e.add_argument("reference"); e.add_argument("predictions"); e.add_argument("output"); e.add_argument("--target-horizon", type=float, default=168.0)
    b = sub.add_parser("benchmark"); b.add_argument("reference"); b.add_argument("predictions"); b.add_argument("output"); b.add_argument("--target-horizon", type=float, default=168.0)
    a = ap.parse_args()
    if a.cmd == "train":
        _, metrics, _ = fit_forecast(a.input, a.artifact, a.horizon, as_of_h=a.as_of, target_horizons=a.target_horizons)
        print(json.dumps(metrics, indent=2, default=str)); return 0
    if a.cmd == "predict":
        m = load_forecast_model(a.artifact)
        d = predict_forecast(m, a.input, a.output, target_horizon=a.target_horizon, forecast_origin_h=a.forecast_origin_h)
        print(json.dumps({"rows": len(d), "output": a.output, "target_horizon_h": float(a.target_horizon if a.target_horizon is not None else m.target_horizon_h)}, indent=2)); return 0
    ref = _read(a.reference); pred = pd.read_csv(a.predictions)
    out = evaluate_forecast(ref, pred, target_horizon=a.target_horizon) if a.cmd == "evaluate" else benchmark_forecast(ref, pred, target_horizon=a.target_horizon)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(out.to_json(orient="records", indent=2), encoding="utf-8")
    print(out.to_json(orient="records", indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
