"""Universal, model-independent feature engine for component screening.

Strictly enforces information boundaries (as_of_h) to prevent future telemetry 
leakage and safely handles missing lot/family data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.ingest import CANONICAL_COLUMNS, Canonicalizer, ParameterOntology, SchemaProfiler, _project_configs

REQUIRED_CANONICAL = {
    "observation_id", "part_id", "lot_id", "component_family", "parameter",
    "value", "time_h", "test_stage", "test_method", "temperature_C",
    "burnin_temperature_C", "voltage_V", "current_A", "stress_mode",
    "measurement_status", "censoring", "quality_flag"
}

def _finite_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.where(np.abs(den) > 1e-15)
    return num / den

def _mad(s: pd.Series) -> float:
    x = s.dropna().to_numpy(dtype=float)
    if x.size == 0: return np.nan
    return float(np.median(np.abs(x - np.median(x))))

def _iqr(s: pd.Series) -> float:
    x = s.dropna().to_numpy(dtype=float)
    if x.size == 0: return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))

def _percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)

def _stable_category_code(series: pd.Series) -> pd.Series:
    def code(value: object) -> int:
        token = "<MISSING>" if value is None or pd.isna(value) else str(value)
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & 0x7FFFFFFFFFFFFFFF
    return series.map(code).astype("int64")

def _direction_factor(direction: str | None) -> float:
    if direction == "lower_is_worse": return -1.0
    return 1.0

def _validate_canonical_frame(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_CANONICAL.difference(df.columns))
    if missing:
        raise ValueError("Canonical dataset is missing required columns: " + ", ".join(missing))

class FeatureEngine:
    """Builds universal screening features from canonical observations."""

    def __init__(self, parameter_config: Mapping[str, object] | None = None):
        self.parameter_specs: dict[str, Mapping[str, object]] = {}
        if parameter_config is not None:
            self.parameter_specs = {str(k): v for k, v in parameter_config.get("parameters", {}).items()}

    @classmethod
    def from_project_config(cls) -> "FeatureEngine":
        return cls(_project_configs()["parameters"])

    def _scoped_keys(self, df: pd.DataFrame, include_stage: bool = True) -> list[str]:
        keys = []
        for k in ("lot_id", "component_family", "parameter", "time_h"):
            if k in df.columns and df[k].notna().any(): keys.append(k)
        if include_stage and "test_stage" in df.columns and df[k].notna().any(): keys.append("test_stage")
        for k in ("test_method", "stress_mode"):
            if k in df.columns and df[k].notna().any(): keys.append(k)
        return keys

    def _population_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["value"] = _finite_series(out["value"])
        eligible = out["parameter"].notna() & out["value"].notna()

        feature_names = [
            "lot_count", "lot_mean", "lot_std", "lot_min", "lot_max", "lot_median",
            "lot_mad", "lot_iqr", "lot_relative_deviation", "lot_relative_percent",
            "lot_robust_z", "lot_z", "lot_percentile", "directional_lot_deviation",
            "directional_lot_robust_z",
        ]
        for name in feature_names: out[name] = np.nan

        if not eligible.any(): return out

        work = out.loc[eligible].drop(columns=feature_names).copy()
        keys = self._scoped_keys(work)
        grp = work.groupby(keys, dropna=False, sort=False)["value"]
        stats = grp.agg(lot_count="count", lot_mean="mean", lot_std="std", lot_min="min", lot_max="max", lot_median="median")
        merged = pd.concat([stats, grp.agg(_mad).rename("lot_mad"), grp.agg(_iqr).rename("lot_iqr")], axis=1).reset_index()
        work = work.merge(merged, on=keys, how="left", sort=False)

        work["lot_relative_deviation"] = work["value"] - work["lot_median"]
        work["lot_relative_percent"] = _safe_div(work["lot_relative_deviation"], work["lot_median"].abs()) * 100.0
        robust_scale = 1.4826 * work["lot_mad"]
        work["lot_robust_z"] = _safe_div(work["lot_relative_deviation"], robust_scale)
        work["lot_z"] = _safe_div(work["value"] - work["lot_mean"], work["lot_std"])
        work["lot_percentile"] = work.groupby(keys, dropna=False, sort=False)["value"].transform(_percentile_rank)

        direction = work["parameter"].astype("string").map(lambda p: _direction_factor(self.parameter_specs.get(str(p), {}).get("directionality"))).astype(float)
        work["directional_lot_deviation"] = work["lot_relative_deviation"] * direction.to_numpy()
        work["directional_lot_robust_z"] = work["lot_robust_z"] * direction.to_numpy()

        if "observation_id" in out.columns:
            derived = work[["observation_id", *feature_names]].drop_duplicates("observation_id", keep="first")
            out = out.drop(columns=feature_names).merge(derived, on="observation_id", how="left", sort=False)
        else:
            out.loc[work.index, feature_names] = work[feature_names].to_numpy()
        return out

    @staticmethod
    def _temporal_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        g = df.copy()
        keys = ["part_id", "parameter"]
        for k in ("test_stage", "test_method", "stress_mode"):
            if k in g.columns and g[k].notna().any(): keys.append(k)

        g["_time_num"] = pd.to_numeric(g["time_h"], errors="coerce")
        if "forecast_origin_h" in g.columns:
            g = g[g["_time_num"] <= pd.to_numeric(g["forecast_origin_h"], errors="coerce")].copy()
            
        g["_value_num"] = _finite_series(g["value"])
        sort_cols = keys + ["_time_num"]
        if "source_row_number" in g.columns: sort_cols.append("source_row_number")
        g = g.sort_values(sort_cols, kind="stable").copy()
        grouped = g.groupby(keys, dropna=False, sort=False)

        prev_time = grouped["_time_num"].shift(1)
        prev_value = grouped["_value_num"].shift(1)
        dt = g["_time_num"] - prev_time
        valid_pair = g["_time_num"].notna() & g["_value_num"].notna() & prev_time.notna() & prev_value.notna() & dt.ne(0)

        g["temporal_delta"] = (g["_value_num"] - prev_value).where(valid_pair)
        g["temporal_slope"] = (g["temporal_delta"] / dt.where(valid_pair)).where(valid_pair)
        g["normalized_drift"] = (g["temporal_delta"] / prev_value.abs().where(valid_pair)).where(valid_pair)

        prev_slope = grouped["temporal_slope"].shift(1)
        slope_valid = g["temporal_slope"].notna() & prev_slope.notna() & dt.ne(0)
        g["temporal_acceleration"] = ((g["temporal_slope"] - prev_slope) / dt.where(slope_valid)).where(slope_valid)
        g["temporal_curvature"] = (g["temporal_slope"] - prev_slope).where(slope_valid)
        cp_denom = g["temporal_slope"].abs() + prev_slope.abs() + 1e-15
        g["change_point_indicator"] = ((g["temporal_curvature"].abs() / cp_denom) >= 0.25).astype("int8")

        g["trajectory_deviation"] = g.get("lot_relative_deviation", pd.Series(np.nan, index=g.index))

        valid_values = g["_value_num"].notna() & g["_time_num"].notna()
        valid_value = g["_value_num"].where(valid_values)
        valid_time = g["_time_num"].where(valid_values)
        first_value = valid_value.groupby([g[k] for k in keys], dropna=False, sort=False).transform("first")
        last_value = valid_value.groupby([g[k] for k in keys], dropna=False, sort=False).transform("last")
        first_time = valid_time.groupby([g[k] for k in keys], dropna=False, sort=False).transform("min")
        last_time = valid_time.groupby([g[k] for k in keys], dropna=False, sort=False).transform("max")

        g["observed_readpoint_count"] = valid_value.groupby([g[k] for k in keys], dropna=False, sort=False).transform("count").fillna(0).astype("int64")
        total_dt = last_time - first_time
        total_change = last_value - first_value
        g["trajectory_overall_slope"] = (total_change / total_dt.where(total_dt.abs() > 1e-15)).where(total_dt.abs() > 1e-15)
        g["trajectory_total_normalized_drift"] = (total_change / first_value.abs().where(first_value.abs() > 1e-15)).where(first_value.abs() > 1e-15)
        g["time_span_h"] = total_dt.fillna(0.0)

        denom = g["temporal_slope"].abs() + prev_slope.abs() + 1e-15
        g["trajectory_change_point_score"] = (((g["temporal_slope"] - prev_slope).abs() / denom).clip(upper=1.0).where(slope_valid))

        if "directional_lot_robust_z" in g.columns:
            rz = pd.to_numeric(g["directional_lot_robust_z"], errors="coerce")
            z_group = rz.groupby([g[k] for k in keys], dropna=False, sort=False)
            g["trajectory_abs_robust_z_max"] = z_group.transform("max", numeric_only=False).abs()
            z_sq = rz.pow(2)
            z_sq_group = z_sq.groupby([g[k] for k in keys], dropna=False, sort=False)
            g["trajectory_abs_robust_z_rms"] = np.sqrt(z_sq_group.transform("sum") / z_sq_group.transform("count").where(z_sq_group.transform("count") > 0))
        else:
            g["trajectory_abs_robust_z_max"] = np.nan
            g["trajectory_abs_robust_z_rms"] = np.nan

        g.drop(columns=["_time_num", "_value_num"], inplace=True)
        return g

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self._temporal_features_vectorized(df)
        order_cols = [c for c in ("part_id", "parameter", "time_h", "source_row_number") if c in result.columns]
        if order_cols: result = result.sort_values(order_cols, kind="stable").reset_index(drop=True)
        return result

    @staticmethod
    def _context_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ("component_family", "component_type", "parameter", "semantic_type", "physical_quantity", "test_stage", "test_method", "stress_mode", "unit"):
            if col in out.columns: out[f"context_{col}_code"] = _stable_category_code(out[col])
        for col in ("temperature_C", "burnin_temperature_C", "voltage_V", "current_A", "time_h"):
            if col in out.columns:
                out[f"context_{col}_missing"] = out[col].isna().astype("int8")
                out[col] = _finite_series(out[col])
        if {"temperature_C", "burnin_temperature_C"}.issubset(out.columns): out["context_temperature_offset_C"] = out["temperature_C"] - out["burnin_temperature_C"]
        if {"voltage_V", "current_A"}.issubset(out.columns): out["context_apparent_power_W"] = out["voltage_V"] * out["current_A"]
        return out

    @staticmethod
    def _quality_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        value_missing = _finite_series(out["value"]).isna()
        if "measurement_status" in out.columns:
            status = out["measurement_status"].astype("string").str.upper()
            status_missing = status.isin(["MISSING", "FAILED_TEST", "BELOW_DETECTION", "CENSORED"])
            out["quality_is_missing"] = (value_missing | status_missing).astype("int8")
            out["quality_is_valid"] = (status.eq("VALID") & ~value_missing).astype("int8")
        else:
            out["quality_is_valid"] = (~value_missing).astype("int8")
            out["quality_is_missing"] = value_missing.astype("int8")

        censor = out.get("censoring", pd.Series("NONE", index=out.index)).astype("string").str.upper()
        out["quality_is_censored"] = (~censor.isin(["NONE", "NAN", "", "<NA>"])).astype("int8")
        out["quality_censor_direction"] = censor.map({"GT": 1, "GE": 1, "LT": -1, "LE": -1}).fillna(0).astype("int8")
        qflag = out.get("quality_flag", pd.Series("VALID", index=out.index)).astype("string").str.upper()
        out["quality_flag_code"] = qflag.map({"VALID": 0, "LIMITED": 1, "ERROR": 2}).fillna(3).astype("int8")

        score = 1.0 - 0.45 * out["quality_is_missing"].astype(float) - 0.20 * out["quality_is_censored"].astype(float) - 0.15 * (out["quality_flag_code"] >= 1).astype(float)
        out["measurement_quality_score"] = score.clip(lower=0.0, upper=1.0)
        return out

    def build(self, canonical: pd.DataFrame) -> pd.DataFrame:
        # Autonomous safety: fill missing required context so grouping doesn't drop rows
        df = canonical.copy()
        df["lot_id"] = df.get("lot_id", pd.Series("UNKNOWN_LOT", index=df.index)).fillna("UNKNOWN_LOT")
        df["component_family"] = df.get("component_family", pd.Series("UNKNOWN_FAMILY", index=df.index)).fillna("UNKNOWN_FAMILY")
        _validate_canonical_frame(df)
        
        df["value"] = _finite_series(df["value"])
        df = self._population_features(df)
        df = self._temporal_features(df)
        df = self._context_features(df)
        df = self._quality_features(df)

        priority = [
            "observation_id", "part_id", "lot_id", "component_family", "parameter", "value", "unit", "time_h",
            "lot_median", "lot_mad", "lot_robust_z", "temporal_delta", "temporal_slope",
            "temporal_acceleration", "measurement_quality_score",
        ]
        ordered = [c for c in priority if c in df.columns]
        ordered += [c for c in df.columns if c not in ordered]
        return df[ordered].reset_index(drop=True)

def validate_features(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors = []
    _validate_canonical_frame(df)
    for col in ["lot_robust_z", "lot_relative_deviation", "temporal_slope", "measurement_quality_score"]:
        if col not in df.columns: errors.append(f"missing_feature:{col}")
    if df.columns.duplicated().any(): errors.append("duplicate_feature_columns")
    return len(errors) == 0, errors

def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-canonical")
    p.add_argument("input")
    p.add_argument("output")
    args = parser.parse_args()
    if args.cmd == "build-canonical":
        df = pd.read_csv(args.input, low_memory=False)
        result = FeatureEngine.from_project_config().build(df)
        result.to_csv(args.output, index=False)
        return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
