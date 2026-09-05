"""Evidence, benchmark, robustness, and decision-support utilities.

Strictly enforces operational constraints (max_reject_rate) during threshold tuning
and incorporates progressive lead-time mapping for latent escapes.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score, roc_auc_score)

from src.utils import PROJECT_ROOT, load_yaml, file_sha256
from src.pipeline import PipelineArtifacts, screen_dataframe, progressive_screen_dataframe


def _find_col(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for n in names:
        if not n: continue
        if n in df.columns: return n
        if str(n).lower() in lower: return lower[str(n).lower()]
    return None

def _part_series(df: pd.DataFrame) -> pd.Series:
    c = _find_col(df, ["part_id", "part", "component_id", "serial_id", "device_id"])
    if c is None: return pd.Series([f"ROW_{i}" for i in range(len(df))], index=df.index)
    return df[c].astype(str)

def _lot_series(df: pd.DataFrame) -> pd.Series:
    c = _find_col(df, ["lot_id", "lot", "batch_id", "batch"])
    if c is None: return pd.Series(["LOT_UNKNOWN"] * len(df), index=df.index)
    return df[c].astype(str)

class MetricResult(dict):
    __getattr__ = dict.get
    def __setattr__(self, key, value):
        self[key] = value

def _load(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False)
    raise ValueError(f"Unsupported benchmark input: {p}")

def _binary_truth(d: pd.DataFrame, label_col: str = "future_defective") -> pd.Series:
    if label_col in d.columns:
        return pd.to_numeric(d[label_col], errors="coerce").fillna(0).astype(int)
    if "latent_defect_label" in d.columns:
        return pd.to_numeric(d["latent_defect_label"], errors="coerce").fillna(0).astype(int)
    if "defect_state" in d.columns:
        return d["defect_state"].astype(str).str.lower().isin(
            {"latent", "hard", "defective", "failed", "obvious_failure"}
        ).astype(int)
    raise ValueError("No future-defect label found.")

def _metrics(y: Sequence[int], score: Sequence[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    pred = (s >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel() if len(y) else (0, 0, 0, 0)
    return {
        "rows": int(len(y)), "positives": int(y.sum()), "threshold": float(threshold),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "false_negative_rate": float(fn / max(tp + fn, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "reject_rate": float(pred.mean()) if len(pred) else 0.0,
        "pr_auc": float(average_precision_score(y, s)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def optimize_threshold(y: Sequence[int], score: Sequence[float], *, fn_cost: float = 100.0,
                       fp_cost: float = 1.0, review_cost: float = 0.25,
                       max_reject_rate: float = 0.25,
                       min_recall: float | None = None,
                       target_metric: str = "future_recall") -> dict[str, Any]:
    """Choose a threshold strictly constrained by maximum allowable rejection."""
    y = np.asarray(y, dtype=int)
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    if len(y) != len(s):
        raise ValueError(f"Threshold calibration length mismatch: y={len(y)} score={len(s)}")
    
    candidates = np.unique(np.r_[0.0, 1.0, s])
    best = None
    
    for t in candidates:
        m = _metrics(y, s, float(t))
        # STRICT CONSTRAINT ENFORCEMENT
        if m["reject_rate"] > float(max_reject_rate):
            continue
        if min_recall is not None and m["recall"] < float(min_recall):
            continue
            
        cost = fn_cost * m["fn"] + fp_cost * m["fp"] + review_cost * m["reject_rate"] * len(y)
        if target_metric == "escape_recall":
            key = (-m["recall"], cost, m["false_negative_rate"], m["false_positive_rate"], m["reject_rate"])
            objective = "maximize escape recall subject to rejection burden"
        else:
            key = (cost, m["false_negative_rate"], m["false_positive_rate"], m["reject_rate"])
            objective = "minimize fn-cost subject to rejection burden"
            
        if best is None or key < best["key"]:
            best = {"threshold": float(t), "cost": float(cost), "metrics": m, "key": key, "objective": objective}
            
    if best is None:
        # Fallback to extremely conservative threshold if constraints cannot be met
        t = float(np.max(s) + 1e-12) if len(s) else 1.0
        m = _metrics(y, s, t)
        best = {"threshold": t, "cost": float(fn_cost * int(y.sum())), "metrics": m,
                "key": (float("inf"),), "objective": "Constraints infeasible; conservative fallback"}
                
    return MetricResult({
        "threshold": best["threshold"], "cost": best["cost"], **best["metrics"],
        "objective": best["objective"], "target_metric": target_metric,
        "max_reject_rate": float(max_reject_rate), "selection_set": "validation",
    })

def calibrate_safety_policy(validation_screening: pd.DataFrame, *, output_path: str | Path | None = None,
                            fn_cost: float = 100.0, fp_cost: float = 1.0, max_reject_rate: float = .25) -> dict[str, Any]:
    d = validation_screening.copy()
    y = _binary_truth(d)
    score = pd.to_numeric(d.get("risk_score", d.get("anomaly_risk", 0.0)), errors="coerce").fillna(0).to_numpy(float)
    res = optimize_threshold(y, score, fn_cost=fn_cost, fp_cost=fp_cost, max_reject_rate=max_reject_rate)
    
    # FIX: Dynamically set safe_max based on the trained threshold, but allow it to reach 0.35
    safe_boundary = min(0.35, float(res["threshold"]) * 0.75)
    
    payload = {
        "format": "safety_policy_calibration_v2", "selection_split": "validation",
        "thresholds": {"safe_max": safe_boundary, "reject_min": res["threshold"]},
        "optimization": res,
    }
    if output_path:
        p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

def escape_matrix(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy(); y = _binary_truth(x)
    absfail = pd.to_numeric(x.get("absolute_fail_168h", 0), errors="coerce").fillna(0).astype(int)
    abs_pass = absfail.eq(0)
    x["future_defective"] = y
    x["absolute_pass"] = abs_pass
    x["latent_escape"] = abs_pass & y.eq(1)
    x["group"] = np.select(
        [abs_pass & y.eq(0), abs_pass & y.eq(1), (~abs_pass) & y.eq(0), (~abs_pass) & y.eq(1)],
        ["absolute_pass_future_safe", "latent_escape", "obvious_absolute_fail_future_safe", "obvious_failure"],
        default="unknown",
    )
    keep = ["part_id", "lot_id", "group", "absolute_pass", "future_defective", "latent_escape"]
    return x[[c for c in keep + ["defect_state", "failure_mode", "split", "high_but_safe"] if c in x.columns]].copy()

def _forecast_metrics(y, pred, lower=None, upper=None, actual_limit=None, defective=None):
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)
    out = {"n": int(m.sum()), "mae": float(mean_absolute_error(y[m], pred[m])) if m.any() else None,
           "rmse": float(np.sqrt(mean_squared_error(y[m], pred[m]))) if m.any() else None}
    if lower is not None and upper is not None:
        lo = np.asarray(lower, float); hi = np.asarray(upper, float)
        ok = m & np.isfinite(lo) & np.isfinite(hi)
        out["conformal_coverage"] = float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else None
        out["mean_interval_width"] = float(np.mean(hi[ok] - lo[ok])) if ok.any() else None
    return out

def robustness_suite(d: pd.DataFrame, seed: int = 20260831) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {"baseline": d.copy()}
    for frac in (.2, .4, .6):
        z = d.copy()
        cols = [c for c in z.columns if re.search(r"^value_(?:12|24|48|96|120|144|168)h$", str(c))]
        if cols:
            idx = rng.choice(len(z), size=max(1, int(len(z) * frac)), replace=False)
            z.loc[z.index[idx], cols] = np.nan
        out[f"missing_{int(frac * 100)}pct"] = z
    if "value_24h" in d:
        z = d.copy()
        base = pd.to_numeric(z["value_24h"], errors="coerce")
        z["value_24h"] = base + rng.normal(0, 0.05, len(z)) * base.abs().fillna(1).to_numpy()
        out["noise_5pct"] = z
    if "parameter" in d.columns:
        unknown = d.copy(); unknown["parameter"] = "unknown_physical_quantity"; out["unknown_parameter"] = unknown
    return out

def run_master_benchmark(input_path: str|Path, output_dir: str|Path="reports/benchmark", *,
                         seed: int=20260831, n_seeds: int=5, target_horizon: float=168.0,
                         target_metric: str="escape_recall", max_reject_rate: float=0.25) -> dict[str,Any]:
    
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(input_path, low_memory=False)
    
    if "split" not in full.columns: 
        raise ValueError("Benchmark requires a split column with train/val/test lots.")
        
    train = full[full.split.astype(str).eq("train")].copy()
    val = full[full.split.astype(str).eq("val")].copy()
    test = full[full.split.astype(str).eq("test")].copy()
    
    n_seeds = max(1, int(n_seeds)); target_horizon = float(target_horizon)
    all_seed_system = []
    
    meta_cols = [c for c in ["part_id", "defect_state", "latent_defect_label", "high_but_safe", "measurement_only_anomaly"] if c in test.columns]
    test_meta = test[meta_cols].drop_duplicates("part_id")
    
    for i in range(n_seeds):
        current_seed = int(seed) + i
        np.random.seed(current_seed)
        run_dir = outdir / f"seed_{current_seed}"
        
        test_run = screen_dataframe(test, run_dir / "test_screen", as_of_h=24.0, target_horizon=target_horizon, auto_train_missing=True)
        prog_run = progressive_screen_dataframe(test, run_dir / "progressive_screen", target_horizon=target_horizon, auto_train_missing=False)
        
        s = test_run.screening.merge(test_meta, on="part_id", how="left")
        y = _binary_truth(s).to_numpy(int)
        flag = s["decision"].astype(str).isin(["REVIEW", "REJECT"]).to_numpy()
        
        sysm = {
            "seed": current_seed,
            "system_recall": float((flag & (y == 1)).sum() / max((y == 1).sum(), 1)),
            "system_fnr": float((~flag & (y == 1)).sum() / max((y == 1).sum(), 1)),
            "high_but_safe_false_reject_rate": float(flag[pd.to_numeric(s.get("high_but_safe", 0), errors="coerce").fillna(0).astype(int).to_numpy() == 1].mean()) if "high_but_safe" in s else None,
            "measurement_artifact_false_flag_rate": float(flag[pd.to_numeric(s.get("measurement_only_anomaly", 0), errors="coerce").fillna(0).astype(int).to_numpy() == 1].mean()) if "measurement_only_anomaly" in s else None
        }
        
        lead_df = prog_run["lead_time"].merge(test_meta, on="part_id", how="left")
        lead_df["future_defective"] = _binary_truth(lead_df).to_numpy(int)
        valid_leads = pd.to_numeric(lead_df.loc[lead_df["future_defective"] == 1, "lead_time_h"], errors="coerce").dropna()
        sysm["median_lead_time_h"] = float(valid_leads.median()) if len(valid_leads) else np.nan
        sysm["lead_time_q1_h"] = float(valid_leads.quantile(0.25)) if len(valid_leads) else np.nan
        sysm["lead_time_q3_h"] = float(valid_leads.quantile(0.75)) if len(valid_leads) else np.nan
        
        all_seed_system.append(sysm)

    system_df = pd.DataFrame(all_seed_system)
    system_df.to_csv(outdir / "system_benchmark.csv", index=False)
    
    robust = robustness_suite(test.head(min(len(test), 5000)).copy(), seed=int(seed))
    rob_report = {k: {"rows": len(v), "missing_fraction": float(v.isna().mean().mean())} for k, v in robust.items()}
    
    return {
        "horizon_h": target_horizon, 
        "n_seeds": n_seeds, 
        "system": system_df.to_dict(orient="records"),
        "robustness": rob_report
    }

if __name__ == "__main__":
    pass
