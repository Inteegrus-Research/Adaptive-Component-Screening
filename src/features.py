"""Family-aware, leakage-safe feature engine for the final ACS benchmark."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

EPS = 1e-9

def _safe_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    m = float(median_abs_deviation(x, scale="normal"))
    return max(m, float(np.nanstd(x) * 0.10), 1e-9)

def _latest(group: pd.DataFrame, origin: float) -> pd.Series | None:
    z = group[group.time_h <= origin + 1e-9].sort_values("time_h")
    if z.empty: return None
    return z.iloc[-1]

def _slope(t: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2: return np.nan
    tt, yy = t[ok], y[ok]
    dt = tt[-1] - tt[0]
    return float((yy[-1] - yy[0]) / dt) if dt > 0 else np.nan

def _recent_slope(t: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2: return np.nan
    tt, yy = t[ok][-3:], y[ok][-3:]
    return _slope(tt, yy)

def _accel(t: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 3: return np.nan
    tt, yy = t[ok][-4:], y[ok][-4:]
    slopes = np.diff(yy) / np.maximum(np.diff(tt), EPS)
    if slopes.size < 2: return np.nan
    return float(slopes[-1] - slopes[0]) / max(tt[-1] - tt[0], EPS)

def _cusum(z: np.ndarray) -> float:
    z = z[np.isfinite(z)]
    if z.size == 0: return np.nan
    pos = neg = 0.0
    drift = 0.25
    threshold = 2.5
    max_abs = 0.0
    for v in z:
        pos = max(0.0, pos + v - drift)
        neg = min(0.0, neg + v + drift)
        max_abs = max(max_abs, abs(pos), abs(neg))
    return float(max_abs / max(threshold, EPS))

def _change_point(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 3: return np.nan, 0.0
    tt, yy = t[ok], y[ok]
    scale = _safe_mad(yy)
    # At the 24 h headline origin there are normally only 0/12/24 h samples.
    # Use a slope-change statistic for exactly three observations so a genuine
    # early change-point is not silently discarded for lack of a fourth sample.
    if len(yy) == 3:
        s1 = (yy[1] - yy[0]) / max(tt[1] - tt[0], EPS)
        s2 = (yy[2] - yy[1]) / max(tt[2] - tt[1], EPS)
        smd = abs(float(s2 - s1)) * (tt[2] - tt[0]) / max(scale, EPS)
        return float(smd), float(smd > 2.5)
    k = max(2, len(yy)//2)
    a, b = yy[:k], yy[-k:]
    smd = abs(float(np.mean(b)-np.mean(a))) / max(scale, EPS)
    return float(smd), float(smd > 3.0)

def build_part_table(df: pd.DataFrame, origin_h: float, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create one row per part with fixed family-relative features using observations <= origin."""
    d = df[df["time_h"] <= float(origin_h) + 1e-9].copy()
    if d.empty: return pd.DataFrame()
    part_meta = d.groupby("part_id", sort=False).agg(
        lot_id=("lot_id","first"), component_family=("component_family","first"), split=("split","first"),
        defect_state=("defect_state","first"), future_defective=("future_defective","first")
    ).reset_index()
    params_by_family = {f: sorted(g.parameter.astype(str).unique()) for f,g in d.groupby("component_family")}
    rows = []
    for pid, gpart in d.groupby("part_id", sort=False):
        family = str(gpart.component_family.iloc[0]); lot = str(gpart.lot_id.iloc[0])
        row: dict[str, Any] = {"part_id": str(pid), "lot_id": lot, "component_family": family,
                               "split": gpart.split.iloc[0] if "split" in gpart else "unknown",
                               "defect_state": gpart.defect_state.iloc[0] if "defect_state" in gpart else "unknown",
                               "future_defective": int(gpart.future_defective.iloc[0]) if "future_defective" in gpart else 0,
                               "origin_h": float(origin_h)}
        family_params = params_by_family.get(family, [])
        for p in family_params:
            gp = gpart[gpart.parameter.astype(str).eq(p)].sort_values("time_h")
            obs = gp[gp.time_h <= origin_h + 1e-9]
            pref = f"{p}__"
            if obs.empty:
                for name in ["current","value_0h","value_24h","slope","recent_slope","acceleration","slope_normalized","acceleration_normalized","ewma","cusum","change_point_score","change_point_flag","distance_lower","distance_upper","distance_nearest","rate_to_limit","warning_distance"]:
                    row[pref+name] = np.nan
                continue
            t = obs.time_h.to_numpy(float); y = obs.value.to_numpy(float)
            cur = float(y[np.isfinite(y)][-1]) if np.isfinite(y).any() else np.nan
            row[pref+"current"] = cur
            row[pref+"value_0h"] = float(obs.loc[obs.time_h.eq(0),"value"].dropna().iloc[-1]) if obs.loc[obs.time_h.eq(0),"value"].notna().any() else np.nan
            row[pref+"value_24h"] = float(obs.loc[obs.time_h.eq(24),"value"].dropna().iloc[-1]) if obs.loc[obs.time_h.eq(24),"value"].notna().any() else np.nan
            slope = _slope(t, y); rs = _recent_slope(t, y); acc = _accel(t, y)
            lo = float(gp.absolute_limit_lower.dropna().iloc[-1]) if "absolute_limit_lower" in gp and gp.absolute_limit_lower.notna().any() else np.nan
            hi = float(gp.absolute_limit_upper.dropna().iloc[-1]) if "absolute_limit_upper" in gp and gp.absolute_limit_upper.notna().any() else np.nan
            scale = hi - lo if np.isfinite(hi) and np.isfinite(lo) and hi > lo else np.nan
            row[pref+"slope"] = slope; row[pref+"recent_slope"] = rs; row[pref+"acceleration"] = acc
            row[pref+"slope_normalized"] = abs(slope)/max(scale,EPS) if np.isfinite(slope) and np.isfinite(scale) else np.nan
            row[pref+"acceleration_normalized"] = abs(acc)/max(scale,EPS) if np.isfinite(acc) and np.isfinite(scale) else np.nan
            baseline = float(np.nanmedian(y)) if np.isfinite(y).any() else np.nan
            row[pref+"ewma"] = float(np.nanmean(y[-3:])) if np.isfinite(y).any() else np.nan
            row[pref+"cusum"] = _cusum((y-baseline)/max(_safe_mad(y),EPS))
            cps, cpf = _change_point(t,y); row[pref+"change_point_score"] = cps; row[pref+"change_point_flag"] = cpf
            if np.isfinite(cur) and np.isfinite(lo) and np.isfinite(hi):
                row[pref+"distance_lower"] = (cur-lo)/max(scale,EPS)
                row[pref+"distance_upper"] = (hi-cur)/max(scale,EPS)
                row[pref+"distance_nearest"] = min(abs(cur-lo), abs(hi-cur))/max(scale,EPS)
                # Positive = approaching the nearest boundary based on recent trend.
                row[pref+"rate_to_limit"] = max((rs if np.isfinite(rs) else 0.0)/(max(scale,EPS)), 0.0)
                warning_lo, warning_hi = lo + 0.2*scale, hi - 0.2*scale
                row[pref+"warning_distance"] = min(abs(cur-warning_lo), abs(cur-warning_hi))/max(scale,EPS)
            else:
                for name in ["distance_lower","distance_upper","distance_nearest","rate_to_limit","warning_distance"]: row[pref+name]=np.nan
        # Cross-parameter summary is intentionally family-local.
        cur_cols = [f"{p}__current" for p in family_params]
        vals = np.array([row.get(c,np.nan) for c in cur_cols],dtype=float)
        row["current_missing_fraction"] = float(np.mean(~np.isfinite(vals))) if len(vals) else 1.0
        def _maxfinite(vals):
            vv=[float(v) for v in vals if np.isfinite(v)]
            return max(vv) if vv else np.nan
        row["max_slope_normalized"] = _maxfinite([row.get(f"{p}__slope_normalized",np.nan) for p in family_params]) if family_params else np.nan
        row["max_acceleration_normalized"] = _maxfinite([row.get(f"{p}__acceleration_normalized",np.nan) for p in family_params]) if family_params else np.nan
        row["max_change_point_score"] = _maxfinite([row.get(f"{p}__change_point_score",np.nan) for p in family_params]) if family_params else 0.0
        if not np.isfinite(row["max_change_point_score"]): row["max_change_point_score"] = 0.0
        row["joint_exceedance_count"] = int(sum(1 for p in family_params if np.isfinite(row.get(f"{p}__distance_nearest",np.nan)) and row[f"{p}__distance_nearest"] < 0.05))
        rows.append(row)
    out = pd.DataFrame(rows)
    if reference is not None and not out.empty:
        out = add_reference_evidence(out, reference, origin_h)
    return out

def add_reference_evidence(parts: pd.DataFrame, reference: pd.DataFrame, origin_h: float) -> pd.DataFrame:
    ref = reference[reference.time_h <= origin_h + 1e-9].copy()
    stats: dict[tuple[str,str], tuple[float,float]] = {}
    for (fam,p), g in ref.groupby(["component_family","parameter"]):
        vals = g.groupby("part_id").last(numeric_only=False).value.to_numpy(float)
        med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
        mad = _safe_mad(vals)
        stats[(str(fam),str(p))] = (med,mad)
    for idx,r in parts.iterrows():
        fam = str(r.component_family)
        zs=[]
        for p in sorted([c[:-9] for c in parts.columns if c.endswith("__current")]):
            cur=r.get(f"{p}__current",np.nan); med,mad=stats.get((fam,p),(np.nan,np.nan))
            z=abs((cur-med)/max(mad,EPS)) if np.isfinite(cur) and np.isfinite(med) else np.nan
            parts.loc[idx,f"{p}__robust_z"] = z
            if np.isfinite(z): zs.append(z)
        parts.loc[idx,"max_robust_z"] = max(zs) if zs else np.nan
        parts.loc[idx,"mean_robust_z"] = float(np.mean(zs)) if zs else np.nan
    return parts

def feature_matrix(df: pd.DataFrame, extra_exclude: set[str] | None = None) -> tuple[np.ndarray, list[str]]:
    exclude = {"part_id","lot_id","component_family","split","defect_state","future_defective","origin_h"} | (extra_exclude or set())
    cols=[]
    for c in df.columns:
        if c in exclude: continue
        if pd.api.types.is_numeric_dtype(df[c]): cols.append(c)
    cols=sorted(cols)
    return df[cols].to_numpy(float), cols
