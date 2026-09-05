"""Leakage-safe, winner-grade evaluation and calibration for SIH 26170.

This module deliberately separates:
  1) model evidence,
  2) validation-selected operating policy,
  3) blind-test measurement.

No benchmark target is hard-coded as a required outcome. FPR, burden, confusion
counts, latent escapes, mechanism performance, robustness and forecast baselines
are all reported from observed predictions.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.forecast import _linear, _persist
from src.pipeline import PipelineArtifacts, progressive_screen_dataframe, screen_dataframe
from src.safety import fit_ood_profile, redecide_screening
from src.utils import PROJECT_ROOT, load_yaml

SEED = 20260831

class MetricResult(dict):
    __getattr__ = dict.get
    def __setattr__(self, key, value):
        self[key] = value

DEFAULT_ORIGINS = (12.0, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0)


def _find_col(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for n in names:
        if n in df.columns:
            return str(n)
        if str(n).lower() in lower:
            return lower[str(n).lower()]
    return None


def _part_series(df: pd.DataFrame) -> pd.Series:
    c = _find_col(df, ["part_id", "part", "component_id", "serial_id", "device_id"])
    if c is None:
        return pd.Series([f"ROW_{i}" for i in range(len(df))], index=df.index)
    return df[c].astype(str)


def _lot_series(df: pd.DataFrame) -> pd.Series:
    c = _find_col(df, ["lot_id", "lot", "batch_id", "batch"])
    if c is None:
        return pd.Series(["LOT_UNKNOWN"] * len(df), index=df.index)
    return df[c].astype(str)


def _binary_truth(df: pd.DataFrame, preferred: str = "future_defective_168h") -> pd.Series:
    for c in (preferred, "future_defective", "latent_defect_label"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "defect_state" in df.columns:
        return df["defect_state"].astype(str).str.lower().isin(
            {"latent", "hard", "defective", "failed", "obvious_failure"}
        ).astype(int)
    raise ValueError(f"No future-defect label found; expected {preferred} or a supported fallback.")


def _mechanism_name(df: pd.DataFrame) -> pd.Series:
    for c in ("primary_failure_mode", "failure_mode"):
        if c in df.columns:
            return df[c].astype(str)
    return pd.Series(["UNKNOWN"] * len(df), index=df.index)


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    x = df[col]
    if x.dtype == bool:
        return x.fillna(False)
    return pd.to_numeric(x, errors="coerce").fillna(0).astype(int).eq(1)


def classification_metrics(y: Sequence[int], flag: Sequence[bool], *, threshold: float | None = None) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(flag, dtype=bool)
    if len(y) != len(pred):
        raise ValueError(f"Length mismatch y={len(y)} flag={len(pred)}")
    tn, fp, fn, tp = confusion_matrix(y, pred.astype(int), labels=[0, 1]).ravel() if len(y) else (0, 0, 0, 0)
    positives = int(tp + fn)
    negatives = int(tn + fp)
    out = {
        "n": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "recall": float(tp / max(positives, 1)),
        "false_negative_rate": float(fn / max(positives, 1)),
        "precision": float(precision_score(y, pred.astype(int), zero_division=0)) if len(y) else None,
        "false_positive_rate": float(fp / max(negatives, 1)),
        "specificity": float(tn / max(negatives, 1)),
        "negative_predictive_value": float(tn / max(tn + fn, 1)),
        "flag_rate": float(pred.mean()) if len(pred) else 0.0,
    }
    if threshold is not None:
        out["threshold"] = float(threshold)
    return out


def score_metrics(y: Sequence[int], score: Sequence[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    pred = s >= float(threshold)
    out = classification_metrics(y, pred, threshold=threshold)
    out["pr_auc"] = float(average_precision_score(y, s)) if y.sum() and y.sum() < len(y) else None
    out["roc_auc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None
    return out


def optimize_threshold(
    y: Sequence[int],
    score: Sequence[float],
    *,
    fn_cost: float = 100.0,
    fp_cost: float = 1.0,
    review_cost: float = 0.25,
    max_reject_rate: float = 1.0,
    min_recall: float | None = None,
    target_metric: str = "future_recall",
) -> dict[str, Any]:
    """Find a threshold on validation data only.

    The constraint is on *flag burden*, not merely recall. This prevents the
    degenerate all-flag solution from winning by construction.
    """
    y = np.asarray(y, dtype=int)
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    if len(y) != len(s):
        raise ValueError("Threshold calibration length mismatch")
    candidates = np.unique(np.r_[0.0, 1.0, np.quantile(s, np.linspace(0, 1, 101))])
    best: dict[str, Any] | None = None
    for t in candidates:
        m = score_metrics(y, s, float(t))
        if m["flag_rate"] > float(max_reject_rate):
            continue
        if min_recall is not None and m["recall"] < float(min_recall):
            continue
        cost = (
            fn_cost * m["fn"]
            + fp_cost * m["fp"]
            + review_cost * m["flag_rate"] * len(y)
        )
        if target_metric == "escape_recall":
            key = (-m["recall"], m["false_positive_rate"], m["flag_rate"], cost)
        else:
            key = (cost, m["false_negative_rate"], m["false_positive_rate"], m["flag_rate"])
        if best is None or key < best["key"]:
            best = {"threshold": float(t), "cost": float(cost), "metrics": m, "key": key}
    if best is None:
        # This means the operational constraint is itself infeasible for the supplied scores.
        # We do not silently relax it; we expose the failure.
        return MetricResult({
            "feasible": False,
            "threshold": float(np.max(s) + 1e-12) if len(s) else 1.0,
            "objective": "infeasible_under_operational_burden_constraint",
            "selection_set": "validation",
            "max_reject_rate": float(max_reject_rate),
            "metrics": score_metrics(y, s, float(np.max(s) + 1e-12) if len(s) else 1.0),
        })
    return MetricResult({
        "feasible": True,
        "threshold": best["threshold"],
        "cost": best["cost"],
        **best["metrics"],
        "objective": "maximize_recall_under_burden" if target_metric == "escape_recall" else "minimize_fn_cost_under_burden",
        "target_metric": target_metric,
        "max_reject_rate": float(max_reject_rate),
        "selection_set": "validation",
    })


def calibrate_safety_policy(
    validation_screening: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    fn_cost: float = 100.0,
    fp_cost: float = 1.0,
    max_reject_rate: float = 0.25,
    target_metric: str = "future_recall",
) -> dict[str, Any]:
    """Calibrate the scalar risk threshold on validation only and emit a frozen policy artifact."""
    d = validation_screening.copy()
    y = _binary_truth(d, "future_defective") if "future_defective" in d.columns else _binary_truth(d, "future_defective_168h")
    if target_metric == "escape_recall":
        if "latent_escape_target" not in d.columns:
            if "absolute_fail_168h" in d.columns:
                absfail = _bool_series(d, "absolute_fail_168h")
                d["latent_escape_target"] = ((~absfail) & y.eq(1)).astype(int)
            else:
                d["latent_escape_target"] = y.astype(int)
        calibration_y = d["latent_escape_target"].astype(int).to_numpy()
    else:
        calibration_y = y.to_numpy()
    score = pd.to_numeric(d.get("risk_score", d.get("anomaly_risk", 0.0)), errors="coerce").fillna(0).to_numpy(float)
    opt = optimize_threshold(calibration_y, score, fn_cost=fn_cost, fp_cost=fp_cost, review_cost=0.25, max_reject_rate=max_reject_rate, target_metric=target_metric)
    if not opt.get("feasible", False):
        raise RuntimeError(f"Unable to select a validation operating point under max_reject_rate={max_reject_rate}: {opt}")
    reject_min = float(opt["threshold"])
    negative_scores = score[calibration_y == 0]
    healthy_p95 = float(np.quantile(negative_scores, 0.95)) if negative_scores.size else max(0.0, reject_min - 0.05)
    safe_max = float(max(0.0, min(reject_min - 1e-6, healthy_p95)))
    payload = {
        "format": "safety_policy_calibration_v3",
        "selection_split": "validation",
        "objective": "minimize catastrophic false negatives subject to bounded escalation burden",
        "thresholds": {"safe_max": safe_max, "review_max": reject_min, "reject_min": reject_min},
        "optimization": opt,
        "validation_rows": int(len(d)),
        "validation_positive_count": int(calibration_y.sum()),
        "calibration_target": target_metric,
        "max_reject_rate": float(max_reject_rate),
        "note": "Thresholds are selected from validation evidence only; final test labels are never used for calibration.",
    }
    if output_path:
        p=Path(output_path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _attach_labels(screening: pd.DataFrame, raw: pd.DataFrame, target_horizon: float = 168.0) -> pd.DataFrame:
    cols = [
        "part_id", "lot_id", "split", "defect_state", "latent_defect_label",
        "failure_mode", "primary_failure_mode", "high_but_safe", "measurement_only_anomaly",
        "precursor_strength", "precursor_effect",
        "future_defective_168h", "absolute_fail_168h", "drift_failure_168h",
    ]
    present = [c for c in cols if c in raw.columns]
    labels = raw[present].drop_duplicates("part_id") if present else pd.DataFrame({"part_id": raw["part_id"].astype(str).unique()})
    labels["part_id"] = labels["part_id"].astype(str)
    out = screening.copy()
    out["part_id"] = out["part_id"].astype(str)
    merged = out.merge(labels, on="part_id", how="left", suffixes=("", "_truth"))
    return merged


def system_disposition_metrics(screening: pd.DataFrame, raw: pd.DataFrame, target_horizon: float = 168.0) -> dict[str, Any]:
    d = _attach_labels(screening, raw, target_horizon)
    y = _binary_truth(d, f"future_defective_{int(target_horizon)}h")
    decisions = d["decision"].astype(str) if "decision" in d else pd.Series(["UNKNOWN"] * len(d))
    escalation = decisions.isin({"REVIEW", "REJECT"})
    reject = decisions.eq("REJECT")
    safe = decisions.eq("SAFE")
    unknown = decisions.eq("UNKNOWN")
    review = decisions.eq("REVIEW")
    policy_cfg = load_yaml(PROJECT_ROOT / "configs" / "policy.yaml")
    costs = policy_cfg.get("costs", {}) if isinstance(policy_cfg, dict) else {}
    fn_cost = float(costs.get("false_negative", 100.0))
    fp_cost = float(costs.get("false_positive", 1.0))
    review_cost = float(costs.get("uncertain_decision", 0.25))
    unknown_cost = float(costs.get("data_quality_failure", 10.0))
    cm = classification_metrics(y.to_numpy(), escalation.to_numpy())
    cm.update({
        "safe_count": int(safe.sum()),
        "review_count": int(review.sum()),
        "reject_count": int(reject.sum()),
        "unknown_count": int(unknown.sum()),
        "safe_rate": float(safe.mean()) if len(d) else 0.0,
        "review_rate": float(review.mean()) if len(d) else 0.0,
        "reject_rate": float(reject.mean()) if len(d) else 0.0,
        "unknown_rate": float(unknown.mean()) if len(d) else 0.0,
        "escalation_rate": float(escalation.mean()) if len(d) else 0.0,
        "automatic_reject_burden": float(reject.mean()) if len(d) else 0.0,
        "cost_weighted_loss": float(
            fn_cost * (~escalation & y.eq(1)).sum()
            + fp_cost * (escalation & y.eq(0)).sum()
            + review_cost * review.sum()
            + unknown_cost * unknown.sum()
        ),
        "ood_low_count": int(d.get("ood_status", pd.Series("UNKNOWN", index=d.index)).astype(str).eq("LOW").sum()),
        "ood_moderate_count": int(d.get("ood_status", pd.Series("UNKNOWN", index=d.index)).astype(str).eq("MODERATE").sum()),
        "ood_severe_count": int(d.get("ood_status", pd.Series("UNKNOWN", index=d.index)).astype(str).eq("SEVERE").sum()),
        "false_reject_rate": float((reject & y.eq(0)).sum() / max(int(y.eq(0).sum()), 1)),
        "false_review_rate": float((review & y.eq(0)).sum() / max(int(y.eq(0).sum()), 1)),
    })
    hs = _bool_series(d, "high_but_safe")
    ma = _bool_series(d, "measurement_only_anomaly")
    if hs.any():
        cm["high_safe_total"] = int(hs.sum())
        cm["high_safe_escalated"] = int((hs & escalation).sum())
        cm["high_safe_review"] = int((hs & review).sum())
        cm["high_safe_reject"] = int((hs & reject).sum())
        cm["high_safe_false_disposition_rate"] = float((hs & escalation).sum() / hs.sum())
    else:
        cm.update({"high_safe_total": 0, "high_safe_escalated": 0, "high_safe_review": 0, "high_safe_reject": 0, "high_safe_false_disposition_rate": None})
    if ma.any():
        cm["measurement_artifact_total"] = int(ma.sum())
        cm["measurement_artifact_escalated"] = int((ma & escalation).sum())
        cm["measurement_artifact_review"] = int((ma & review).sum())
        cm["measurement_artifact_reject"] = int((ma & reject).sum())
        cm["measurement_artifact_false_flag_rate"] = float((ma & escalation).sum() / ma.sum())
    else:
        cm.update({"measurement_artifact_total": 0, "measurement_artifact_escalated": 0, "measurement_artifact_review": 0, "measurement_artifact_reject": 0, "measurement_artifact_false_flag_rate": None})
    normal_aging = _mechanism_name(d).str.upper().eq("NORMAL_AGING")
    cm["normal_aging_total"] = int(normal_aging.sum())
    cm["normal_aging_false_alarm_rate"] = float((normal_aging & escalation).sum() / normal_aging.sum()) if normal_aging.any() else None
    return cm


def latent_escape_metrics(screening: pd.DataFrame, raw: pd.DataFrame, target_horizon: float = 168.0) -> dict[str, Any]:
    d = _attach_labels(screening, raw, target_horizon)
    y = _binary_truth(d, f"future_defective_{int(target_horizon)}h")
    absfail = _bool_series(d, f"absolute_fail_{int(target_horizon)}h")
    escape = (~absfail) & y.eq(1)
    escalation = d["decision"].astype(str).isin({"REVIEW", "REJECT"}) if "decision" in d else pd.Series(False, index=d.index)
    m = classification_metrics(escape.astype(int).to_numpy(), escalation.to_numpy())
    m["latent_escape_count"] = int(escape.sum())
    return {
        "latent_escape": m,
        "escape_precision_denominator": int(escalation.sum()),
    }


def mechanism_table(screening: pd.DataFrame, raw: pd.DataFrame, target_horizon: float = 168.0) -> pd.DataFrame:
    d = _attach_labels(screening, raw, target_horizon)
    y = _binary_truth(d, f"future_defective_{int(target_horizon)}h")
    absfail = _bool_series(d, f"absolute_fail_{int(target_horizon)}h")
    escalation = d["decision"].astype(str).isin({"REVIEW", "REJECT"})
    mech = _mechanism_name(d).str.upper()
    order = ["LATENT_CHANGE_POINT", "LATENT_ACCELERATING", "LATENT_ABRUPT", "HARD_EARLY_FAILURE", "NORMAL_AGING"]
    rows = []
    seen = []
    for name in order + sorted(set(mech.unique()) - set(order)):
        sub = mech.eq(name)
        if not sub.any():
            continue
        esc = classification_metrics(y[sub].to_numpy(), escalation[sub].to_numpy())
        escape = sub & (~absfail) & y.eq(1)
        escape_recall = float((escape & escalation).sum() / max(int(escape.sum()), 1))
        rows.append({
            "mechanism": name,
            "parts": int(sub.sum()),
            "future_defective": int(y[sub].sum()),
            "recall": esc["recall"],
            "false_negative_rate": esc["false_negative_rate"],
            "false_positive_rate": esc["false_positive_rate"],
            "escalation_rate": esc["flag_rate"],
            "escape_count": int(escape.sum()),
            "escape_recall": escape_recall,
        })
    return pd.DataFrame(rows)


def progressive_metrics(progressive: dict[str, Any], raw: pd.DataFrame, target_horizon: float = 168.0) -> pd.DataFrame:
    decisions = progressive["decisions"].copy()
    meta = _attach_labels(pd.DataFrame({"part_id": raw["part_id"].astype(str).unique()}), raw, target_horizon)
    meta = meta[[c for c in ["part_id", "future_defective_168h", "future_defective", "absolute_fail_168h", "failure_mode", "primary_failure_mode"] if c in meta.columns]]
    decisions["part_id"] = decisions["part_id"].astype(str)
    d = decisions.merge(meta, on="part_id", how="left")
    rows = []
    for origin, g in d.groupby("origin_h", sort=True):
        y = _binary_truth(g, f"future_defective_{int(target_horizon)}h")
        esc = g["decision"].astype(str).isin({"REVIEW", "REJECT"})
        absfail = _bool_series(g, f"absolute_fail_{int(target_horizon)}h")
        e = (~absfail) & y.eq(1)
        m = classification_metrics(y.to_numpy(), esc.to_numpy())
        rows.append({"origin_h": float(origin), **m, "escape_count": int(e.sum()), "escape_recall": float((e & esc).sum() / max(int(e.sum()), 1))})
    return pd.DataFrame(rows)


def lead_time_table(progressive: dict[str, Any], raw: pd.DataFrame, target_horizon: float = 168.0) -> pd.DataFrame:
    lead = progressive["lead_time"].copy()
    lead["part_id"] = lead["part_id"].astype(str)
    labels = raw[[c for c in ["part_id", f"future_defective_{int(target_horizon)}h", "defect_state", "failure_mode", "primary_failure_mode"] if c in raw.columns]].drop_duplicates("part_id")
    labels["part_id"] = labels["part_id"].astype(str)
    return lead.merge(labels, on="part_id", how="left")


def summarize_lead_time(lead: pd.DataFrame, target_horizon: float = 168.0) -> dict[str, Any]:
    y = _binary_truth(lead, f"future_defective_{int(target_horizon)}h")
    detected = pd.to_numeric(lead["lead_time_h"], errors="coerce").notna()
    vals = pd.to_numeric(lead.loc[y.eq(1) & detected, "lead_time_h"], errors="coerce").dropna()
    total_def = int(y.eq(1).sum())
    detected_def = int((y.eq(1) & detected).sum())
    out = {
        "future_defective_total": total_def,
        "detected_future_defective": detected_def,
        "lead_detection_rate": float(detected_def / max(total_def, 1)),
        "median_lead_time_h": float(vals.median()) if len(vals) else None,
        "lead_time_q1_h": float(vals.quantile(0.25)) if len(vals) else None,
        "lead_time_q3_h": float(vals.quantile(0.75)) if len(vals) else None,
        "lead_time_min_h": float(vals.min()) if len(vals) else None,
        "lead_time_max_h": float(vals.max()) if len(vals) else None,
    }
    if len(vals):
        for p in (0.10, 0.25, 0.50, 0.75, 0.90):
            out[f"lead_time_p{int(p*100)}_h"] = float(vals.quantile(p))
    return out


def _forecast_target_map(raw: pd.DataFrame, target_horizon: float) -> pd.Series:
    col = f"target_{int(target_horizon) if float(target_horizon).is_integer() else str(target_horizon).replace('.', '_')}h"
    if col not in raw.columns:
        raise ValueError(f"Forecast evaluation requires observed {col}; value_<h> or latent_<h> are not accepted as a hidden substitute.")
    keys = [c for c in ["part_id", "parameter"] if c in raw.columns]
    if "part_id" not in keys:
        raise ValueError("Forecast truth alignment requires part_id")
    return raw[keys + [col]].drop_duplicates(keys).set_index(keys)[col]


def forecast_comparison(raw_test: pd.DataFrame, pred: pd.DataFrame, target_horizon: float = 168.0) -> pd.DataFrame:
    target = _forecast_target_map(raw_test, target_horizon)
    keys = [c for c in ["part_id", "parameter"] if c in pred.columns]
    if keys != list(target.index.names):
        if "parameter" in target.index.names and "parameter" not in pred.columns:
            raise ValueError("Prediction output lacks parameter required for target alignment")
    idx = pd.MultiIndex.from_frame(pred[keys].astype(str))
    y = pd.to_numeric(target.reindex(idx), errors="coerce").to_numpy(float)
    pred_col = f"prediction_{int(target_horizon) if float(target_horizon).is_integer() else str(target_horizon).replace('.', '_')}h"
    if pred_col not in pred.columns:
        pred_col = "prediction_168h"
    lower = pd.to_numeric(pred.get("prediction_lower", np.nan), errors="coerce").to_numpy(float)
    upper = pd.to_numeric(pred.get("prediction_upper", np.nan), errors="coerce").to_numpy(float)
    limit = pd.to_numeric(pred.get("absolute_limit_upper", np.nan), errors="coerce").to_numpy(float) if "absolute_limit_upper" in pred else None
    labels = raw_test[[c for c in ["part_id", "defect_state", "failure_mode", "primary_failure_mode"] if c in raw_test.columns]].drop_duplicates("part_id")
    label_map = labels.set_index("part_id") if not labels.empty else None
    rows = []
    for name, col in [
        ("persistence", "prediction_persistence"),
        ("linear", "prediction_linear"),
        ("ridge", "prediction_ridge"),
        ("gradient_boosting", "prediction_gradient_boosting"),
        ("selected", pred_col),
    ]:
        if col not in pred.columns:
            continue
        p = pd.to_numeric(pred[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        row = {
            "model": name,
            "n": int(ok.sum()),
            "mae": float(mean_absolute_error(y[ok], p[ok])) if ok.any() else None,
            "rmse": float(np.sqrt(mean_squared_error(y[ok], p[ok]))) if ok.any() else None,
        }
        if name == "selected":
            iv = ok & np.isfinite(lower) & np.isfinite(upper)
            row["conformal_coverage"] = float(np.mean((y[iv] >= lower[iv]) & (y[iv] <= upper[iv]))) if iv.any() else None
            row["mean_interval_width"] = float(np.mean(upper[iv] - lower[iv])) if iv.any() else None
            if limit is not None:
                lc = ok & np.isfinite(limit) & (limit > 0)
                actual_cross = y > limit
                pred_cross = upper > limit
                row["limit_crossing_recall"] = float((actual_cross & pred_cross & lc).sum() / max(int((actual_cross & lc).sum()), 1)) if lc.any() else None
        else:
            row["conformal_coverage"] = None
            row["mean_interval_width"] = None
            if limit is not None:
                lc = ok & np.isfinite(limit) & (limit > 0)
                actual_cross = y > limit
                row["limit_crossing_recall"] = float((actual_cross & (p > limit) & lc).sum() / max(int((actual_cross & lc).sum()), 1)) if lc.any() else None
        if label_map is not None and "part_id" in pred.columns:
            parts = pred["part_id"].astype(str).map(label_map.get("defect_state", pd.Series(dtype=object))).astype(str).str.lower()
            dangerous = parts.isin({"latent", "hard", "defective", "failed"}).to_numpy()
            row["dangerous_case_mae"] = float(mean_absolute_error(y[ok & dangerous], p[ok & dangerous])) if np.any(ok & dangerous) else None
        else:
            row["dangerous_case_mae"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_scenarios(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {"baseline": df.copy()}
    value_cols = [c for c in df.columns if re.fullmatch(r"value_(?:12|24|48|72|96|120|144|168)h", str(c))]
    for frac in (0.20, 0.40, 0.60):
        z = df.copy()
        if value_cols and len(z):
            n = max(1, int(len(z) * frac))
            ix = rng.choice(len(z), n, replace=False)
            z.loc[z.index[ix], value_cols] = np.nan
        out[f"missing_{int(frac*100)}pct"] = z
    if "value_24h" in df.columns and len(df):
        z = df.copy()
        base = pd.to_numeric(z["value_24h"], errors="coerce")
        z["value_24h"] = base + rng.normal(0, 0.05, len(z)) * base.abs().fillna(1).to_numpy()
        out["noise_5pct"] = z
    if "parameter" in df.columns:
        z = df.copy(); z["parameter"] = "unknown_physical_quantity"; out["unknown_parameter"] = z
    if "component_family" in df.columns:
        z = df.copy(); z["component_family"] = "UNSEEN_FAMILY"; out["unknown_family"] = z
    if "stress_mode" in df.columns:
        z = df.copy(); z["stress_mode"] = "UNSEEN_STRESS"; out["unknown_stress"] = z
    # Irregular-readpoint scenario: rename one known future readpoint out of the input.
    if "value_48h" in df.columns:
        z = df.copy(); z["value_48h"] = np.nan; out["missing_mid_readpoint"] = z
    return out


def _method_scores(anomaly: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "absolute_limits": pd.to_numeric(anomaly.get("absolute_violation", 0), errors="coerce").fillna(0).to_numpy(float),
        "robust_PAT": pd.to_numeric(anomaly.get("population_evidence", 0), errors="coerce").fillna(0).to_numpy(float),
        "temporal": pd.to_numeric(anomaly.get("temporal_evidence", 0), errors="coerce").fillna(0).to_numpy(float),
        "isolation_forest": pd.to_numeric(anomaly.get("parameter_isolation_evidence", 0), errors="coerce").fillna(0).to_numpy(float),
        "multivariate": pd.to_numeric(anomaly.get("multivariate_component_score", 0), errors="coerce").fillna(0).to_numpy(float),
        "full_ensemble": pd.to_numeric(anomaly.get("anomaly_score", 0), errors="coerce").fillna(0).to_numpy(float),
    }


def _fit_method_operating_points(val_score: pd.DataFrame, raw_val: pd.DataFrame, max_burden: float) -> dict[str, Any]:
    y = _binary_truth(val_score, "future_defective_168h").to_numpy(int)
    absfail = _bool_series(val_score, "absolute_fail_168h")
    escape = (~absfail) & (y == 1)
    out: dict[str, Any] = {}
    for name, s in _method_scores(val_score).items():
        target = escape.astype(int) if escape.sum() else y
        out[name] = optimize_threshold(target, s, max_reject_rate=max_burden, target_metric="escape_recall" if escape.sum() else "future_recall")
    return out


def _ablation_table(test_anomaly: pd.DataFrame, raw_test: pd.DataFrame, operating_points: Mapping[str, Any]) -> pd.DataFrame:
    y = _binary_truth(test_anomaly, "future_defective_168h").to_numpy(int)
    absfail = _bool_series(test_anomaly, "absolute_fail_168h").to_numpy(bool)
    rows = []
    scores = _method_scores(test_anomaly)
    for name, s in scores.items():
        t = operating_points[name]["threshold"]
        pred = s >= t
        m = classification_metrics(y, pred, threshold=t)
        esc = (~absfail) & (y == 1)
        m["method"] = name
        m["latent_escape_recall"] = float((pred & esc).sum() / max(int(esc.sum()), 1))
        rows.append(m)
    return pd.DataFrame(rows)


def run_master_benchmark(
    input_path: str | Path,
    output_dir: str | Path = "reports/benchmark",
    *,
    seed: int = SEED,
    n_seeds: int = 5,
    target_horizon: float = 168.0,
    target_metric: str = "escape_recall",
    max_reject_rate: float = 0.25,
) -> dict[str, Any]:
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(input_path, low_memory=False)
    if "split" not in full.columns:
        raise ValueError("Benchmark requires train/val/test split labels by lot.")
    for split in ("train", "val", "test"):
        if not (full["split"].astype(str) == split).any():
            raise ValueError(f"Missing {split} partition")
    train = full[full.split.astype(str).eq("train")].copy()
    val = full[full.split.astype(str).eq("val")].copy()
    test = full[full.split.astype(str).eq("test")].copy()

    seeds = [int(seed) + i for i in range(max(1, int(n_seeds)))]
    seed_rows = []
    all_mechanisms = []
    all_progressive = []
    all_forecast = []
    all_robustness = []
    all_ablation = []

    for current_seed in seeds:
        run_dir = outdir / f"seed_{current_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(current_seed)

        # Train-only OOD profile for a strict blind-test protocol.
        ood_artifact = run_dir / "ood_train_only.joblib"
        fit_ood_profile(train, ood_artifact)
        artifacts = PipelineArtifacts(
            PROJECT_ROOT / "models" / "anomaly" / "model.joblib",
            PROJECT_ROOT / "models" / "forecast" / "model.joblib",
            ood_artifact,
        )

        val_run = screen_dataframe(val, run_dir / "validation_screen", artifacts=artifacts, as_of_h=24.0, target_horizon=target_horizon, auto_train_missing=False)
        # Compute test evidence once; final decision is re-applied later using the validation-selected thresholds.
        test_run = screen_dataframe(test, run_dir / "test_screen", artifacts=artifacts, as_of_h=24.0, target_horizon=target_horizon, auto_train_missing=False)

        val_screen = _attach_labels(val_run.screening, val, target_horizon)
        test_screen = _attach_labels(test_run.screening, test, target_horizon)

        # Select risk operating point from validation. This cannot touch blind test labels.
        policy_selection = val_screen.copy()
        absfail = _bool_series(policy_selection, f"absolute_fail_{int(target_horizon)}h")
        y_full = _binary_truth(policy_selection, f"future_defective_{int(target_horizon)}h")
        policy_selection["future_defective"] = y_full
        policy_selection["latent_escape_target"] = ((~absfail) & y_full.eq(1)).astype(int)
        calibration_target = "escape_recall" if int(policy_selection["latent_escape_target"].sum()) > 0 else target_metric
        calibration = calibrate_safety_policy(
            policy_selection,
            fn_cost=float(load_yaml(PROJECT_ROOT / "configs" / "policy.yaml").get("costs", {}).get("false_negative", 100.0)),
            fp_cost=float(load_yaml(PROJECT_ROOT / "configs" / "policy.yaml").get("costs", {}).get("false_positive", 1.0)),
            max_reject_rate=max_reject_rate,
            target_metric=calibration_target,
        )
        reject_threshold = float(calibration["thresholds"]["reject_min"])
        safe_threshold = float(calibration["thresholds"]["safe_max"])

        # Re-apply the selected policy to both validation and test without re-fitting evidence.
        val_final = redecide_screening(val_run.screening, safe_max=safe_threshold, reject_min=reject_threshold)
        test_final = redecide_screening(test_run.screening, safe_max=safe_threshold, reject_min=reject_threshold)
        val_final = _attach_labels(val_final, val, target_horizon)
        test_final = _attach_labels(test_final, test, target_horizon)
        val_metrics = system_disposition_metrics(val_final, val, target_horizon)
        test_metrics = system_disposition_metrics(test_final, test, target_horizon)
        esc = latent_escape_metrics(test_final, test, target_horizon)["latent_escape"]

        # Progressive earliest-warning analysis.
        progressive = progressive_screen_dataframe(
            test,
            run_dir / "progressive_screen",
            origins=DEFAULT_ORIGINS,
            target_horizon=target_horizon,
            artifacts=artifacts,
            auto_train_missing=False,
        )
        # Freeze the validation-selected operating point across every early-warning origin.
        tuned_runs = {}
        decision_frames = []
        for origin, rr in progressive["runs"].items():
            tuned = redecide_screening(rr.screening, safe_max=safe_threshold, reject_min=reject_threshold)
            tuned_runs[origin] = tuned
            x = tuned[[c for c in ["part_id", "decision", "risk_score"] if c in tuned.columns]].copy()
            x["origin_h"] = float(origin); decision_frames.append(x)
        progressive["runs_final_policy"] = tuned_runs
        progressive["decisions"] = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
        flags = progressive["decisions"][progressive["decisions"]["decision"].astype(str).isin({"REVIEW","REJECT"})].sort_values(["part_id","origin_h"])
        progressive["lead_time"] = (pd.DataFrame({"part_id": test["part_id"].astype(str).unique()})
            .merge(flags.drop_duplicates("part_id", keep="first")[["part_id","origin_h","decision","risk_score"]].rename(columns={"origin_h":"first_flag_time_h"}), on="part_id", how="left"))
        progressive["lead_time"]["lead_time_h"] = float(target_horizon) - pd.to_numeric(progressive["lead_time"]["first_flag_time_h"], errors="coerce")
        prog_metrics = progressive_metrics(progressive, test, target_horizon)
        leads = lead_time_table(progressive, test, target_horizon)
        lead_summary = summarize_lead_time(leads, target_horizon)
        prog_metrics["seed"] = current_seed
        all_progressive.append(prog_metrics)
        prog_metrics.to_csv(run_dir / "progressive_metrics.csv", index=False)
        leads.to_csv(run_dir / "lead_time.csv", index=False)

        # Mechanism + ablation.
        mech = mechanism_table(test_final, test, target_horizon)
        mech["seed"] = current_seed
        mech.to_csv(run_dir / "mechanism_metrics.csv", index=False)
        all_mechanisms.append(mech)

        method_ops = _fit_method_operating_points(_attach_labels(val_run.anomaly, val, target_horizon), val, max_reject_rate)
        abl = _ablation_table(_attach_labels(test_run.anomaly, test, target_horizon), test, method_ops)
        abl["seed"] = current_seed
        abl.to_csv(run_dir / "module_a_ablation.csv", index=False)
        all_ablation.append(abl)

        # Forecast comparison on untouched test predictions.
        fcmp = forecast_comparison(test, test_run.forecast, target_horizon)
        fcmp["seed"] = current_seed
        fcmp.to_csv(run_dir / "forecast_comparison.csv", index=False)
        all_forecast.append(fcmp)

        # Robustness: measure actual performance changes, not just missingness.
        robust_rows = []
        for scenario, frame in robustness_scenarios(test.head(min(5000, len(test))).copy(), current_seed).items():
            rr = screen_dataframe(frame, run_dir / "robustness" / scenario, artifacts=artifacts, as_of_h=24.0, target_horizon=target_horizon, auto_train_missing=False, render_explanations=False)
            # Apply the frozen validation operating point, preserving OOD/quality gates.
            rr_final = redecide_screening(rr.screening, safe_max=safe_threshold, reject_min=reject_threshold)
            metrics = system_disposition_metrics(rr_final, frame, target_horizon) if f"future_defective_{int(target_horizon)}h" in frame.columns else {}
            metrics.update({"scenario": scenario, "rows": len(frame), "input_missing_fraction": float(frame.isna().mean().mean())})
            robust_rows.append(metrics)
        rob = pd.DataFrame(robust_rows)
        rob.to_csv(run_dir / "robustness_metrics.csv", index=False)
        all_robustness.append(rob)

        seed_row = {
            "seed": current_seed,
            "validation_threshold": reject_threshold,
            "validation_safe_max": safe_threshold,
            "validation": val_metrics,
            "system": test_metrics,
            "latent_escape": esc,
            "lead_time": lead_summary,
            "forecast": fcmp.to_dict(orient="records"),
        }
        seed_rows.append(seed_row)

        # Persist the exact test disposition and its truth metadata for audit.
        test_final.to_csv(run_dir / "test_screening_final_policy.csv", index=False)
        with open(run_dir / "selection.json", "w", encoding="utf-8") as fh:
            json.dump({"seed": current_seed, "threshold": reject_threshold, "safe_max": safe_threshold, "calibration": calibration, "validation_metrics": val_metrics}, fh, indent=2, default=str)

    # Aggregate across seeds only after every seed is independently evaluated.
    def aggregate_nested(rows: list[dict[str, Any]], section: str) -> pd.DataFrame:
        flat = []
        for r in rows:
            x = dict(r[section]); x["seed"] = r["seed"]; flat.append(x)
        d = pd.DataFrame(flat)
        numeric = [c for c in d.columns if c != "seed" and pd.api.types.is_numeric_dtype(d[c])]
        out = []
        for c in numeric:
            out.append({"metric": c, "mean": float(d[c].mean()), "std": float(d[c].std(ddof=1)) if len(d) > 1 else 0.0, "min": float(d[c].min()), "max": float(d[c].max())})
        return pd.DataFrame(out)

    system_agg = aggregate_nested(seed_rows, "system")
    escape_agg = aggregate_nested(seed_rows, "latent_escape")
    lead_agg = aggregate_nested(seed_rows, "lead_time")
    system_agg.to_csv(outdir / "system_metrics_mean_std.csv", index=False)
    escape_agg.to_csv(outdir / "latent_escape_mean_std.csv", index=False)
    lead_agg.to_csv(outdir / "lead_time_mean_std.csv", index=False)

    mech_all = pd.concat(all_mechanisms, ignore_index=True) if all_mechanisms else pd.DataFrame()
    mech_agg = mech_all.groupby("mechanism", as_index=False).agg(
        parts=("parts", "mean"), future_defective=("future_defective", "mean"),
        recall=("recall", "mean"), recall_std=("recall", "std"),
        false_negative_rate=("false_negative_rate", "mean"),
        false_positive_rate=("false_positive_rate", "mean"),
        escalation_rate=("escalation_rate", "mean"), escape_recall=("escape_recall", "mean"),
    ) if not mech_all.empty else pd.DataFrame()
    if not mech_agg.empty: mech_agg["recall_std"] = mech_agg["recall_std"].fillna(0.0)
    mech_agg.to_csv(outdir / "mechanism_metrics_mean.csv", index=False)

    f_all = pd.concat(all_forecast, ignore_index=True) if all_forecast else pd.DataFrame()
    if not f_all.empty:
        f_agg = f_all.groupby("model", as_index=False).agg(
            n=("n", "mean"), mae=("mae", "mean"), mae_std=("mae", "std"), rmse=("rmse", "mean"),
            conformal_coverage=("conformal_coverage", "mean"), mean_interval_width=("mean_interval_width", "mean"),
            limit_crossing_recall=("limit_crossing_recall", "mean"), dangerous_case_mae=("dangerous_case_mae", "mean"),
        )
        for c in ["mae_std"]:
            if c in f_agg: f_agg[c] = f_agg[c].fillna(0.0)
    else: f_agg = pd.DataFrame()
    f_agg.to_csv(outdir / "forecast_metrics_mean.csv", index=False)

    if all_progressive:
        prog_all = pd.concat(all_progressive, ignore_index=True)
    else:
        prog_all = pd.DataFrame()
    if not prog_all.empty:
        prog_agg = prog_all.groupby("origin_h", as_index=False).agg(
            recall=("recall", "mean"), recall_std=("recall", "std"), false_positive_rate=("false_positive_rate", "mean"),
            escalation_rate=("flag_rate", "mean"), escape_recall=("escape_recall", "mean"),
        )
        prog_agg["recall_std"] = prog_agg["recall_std"].fillna(0.0)
    else: prog_agg = pd.DataFrame()
    prog_agg.to_csv(outdir / "progressive_metrics_mean.csv", index=False)

    rob_all = pd.concat(all_robustness, ignore_index=True) if all_robustness else pd.DataFrame()
    if not rob_all.empty:
        rob_agg = rob_all.groupby("scenario", as_index=False).mean(numeric_only=True)
    else: rob_agg = pd.DataFrame()
    rob_agg.to_csv(outdir / "robustness_metrics_mean.csv", index=False)

    abl_all = pd.concat(all_ablation, ignore_index=True) if all_ablation else pd.DataFrame()
    abl_agg = abl_all.groupby("method", as_index=False).agg(recall=("recall", "mean"), false_positive_rate=("false_positive_rate", "mean"), flag_rate=("flag_rate", "mean"), latent_escape_recall=("latent_escape_recall", "mean")) if not abl_all.empty else pd.DataFrame()
    abl_agg.to_csv(outdir / "ablation_metrics_mean.csv", index=False)

    report = {
        "protocol": {
            "train_calibration_only": True,
            "blind_test": True,
            "group_split": "lot",
            "n_seeds": len(seeds),
            "origins_h": list(DEFAULT_ORIGINS),
            "target_horizon_h": float(target_horizon),
            "max_escalation_rate_for_threshold_selection": float(max_reject_rate),
            "note": "All headline test metrics are measured from final frozen dispositions; no test threshold fitting is performed.",
        },
        "seeds": seed_rows,
        "system_mean_std": system_agg.to_dict(orient="records"),
        "latent_escape_mean_std": escape_agg.to_dict(orient="records"),
        "mechanism": mech_agg.to_dict(orient="records"),
        "forecast": f_agg.to_dict(orient="records"),
        "progressive": prog_agg.to_dict(orient="records"),
        "robustness": rob_agg.to_dict(orient="records"),
        "ablation": abl_agg.to_dict(orient="records"),
    }
    (outdir / "benchmark_results.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Judge-facing Markdown.
    def md(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "_No measured results._"
        return df[[c for c in cols if c in df.columns]].to_markdown(index=False)

    lines = [
        "# SIH 26170 — Winner-Grade Blind Benchmark",
        "",
        "## Protocol",
        f"- Seeds: {len(seeds)} ({', '.join(map(str, seeds))})",
        "- Calibration: validation lots only",
        "- OOD reference: training lots only",
        "- Final test: untouched for threshold selection",
        "- Decision flag for safety recall/FPR: REVIEW or REJECT",
        "",
        "## System performance (mean ± std components)",
        md(system_agg, ["metric", "mean", "std", "min", "max"]),
        "",
        "## Latent-escape performance",
        md(escape_agg, ["metric", "mean", "std", "min", "max"]),
        "",
        "## Mechanism breakdown",
        md(mech_agg, ["mechanism", "parts", "future_defective", "recall", "recall_std", "false_positive_rate", "escape_recall"]),
        "",
        "## Module B forecast comparison",
        md(f_agg, ["model", "mae", "mae_std", "rmse", "conformal_coverage", "mean_interval_width", "limit_crossing_recall", "dangerous_case_mae"]),
        "",
        "## Progressive early-warning curve",
        md(prog_agg, ["origin_h", "recall", "recall_std", "false_positive_rate", "escalation_rate", "escape_recall"]),
        "",
        "## Module-A ablation/comparison",
        md(abl_agg, ["method", "recall", "false_positive_rate", "flag_rate", "latent_escape_recall"]),
        "",
        "## Robustness",
        md(rob_agg, ["scenario", "recall", "false_positive_rate", "escalation_rate", "reject_rate", "unknown_rate", "input_missing_fraction"]),
        "",
        "## Interpretation rule",
        "No target metric is hard-coded. Results are accepted exactly as measured. Any claim of superiority must be supported by the corresponding held-out baseline and uncertainty statistics.",
    ]
    (outdir / "benchmark_table.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="SIH 26170 winner-grade blind benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("input")
    b.add_argument("--output-dir", default="reports/benchmark")
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--n-seeds", type=int, default=5)
    b.add_argument("--target-horizon", type=float, default=168.0)
    b.add_argument("--target-metric", choices=["future_recall", "escape_recall"], default="escape_recall")
    b.add_argument("--max-reject-rate", type=float, default=0.25)
    a = ap.parse_args()
    result = run_master_benchmark(a.input, a.output_dir, seed=a.seed, n_seeds=a.n_seeds, target_horizon=a.target_horizon, target_metric=a.target_metric, max_reject_rate=a.max_reject_rate)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# --- Compatibility/audit helpers used by the project's regression tests and final report. ---
def leakage_audit(df: pd.DataFrame, as_of_h: float = 24.0) -> dict[str, Any]:
    future_cols = []
    for c in df.columns:
        m = re.fullmatch(r"(?:value|target|latent)_(\d+(?:\.\d+)?)h", str(c))
        if m and float(m.group(1)) > float(as_of_h):
            future_cols.append(str(c))
    label_like = [c for c in df.columns if c in {"future_defective_48h","future_defective_96h","future_defective_168h","future_defective_240h","future_defective_300h","latent_defect_label","defect_state","failure_mode","primary_failure_mode","high_but_safe","measurement_only_anomaly"}]
    return {"as_of_h": float(as_of_h), "future_observation_columns": future_cols, "label_columns": label_like, "leakage_detected": bool(future_cols or label_like)}


def physics_sanity(df: pd.DataFrame) -> dict[str, Any]:
    bad = []
    for c, lo, hi in [("temperature_C", -100, 500), ("burnin_temperature_C", -100, 500), ("voltage_V", -1e5, 1e5), ("current_A", -1e5, 1e5)]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if (x.notna() & ((x < lo) | (x > hi))).any(): bad.append(c)
    return {"pass": not bad, "invalid_columns": bad}


def adversarial_suite(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    out = {"baseline": df.copy()}
    rng = np.random.default_rng(seed)
    renamed = df.copy()
    if "parameter" in renamed.columns: renamed["parameter"] = renamed["parameter"].astype(str).str.upper()
    out["renamed"] = renamed
    for frac in (.20, .40, .60):
        z = df.copy()
        cols = [c for c in z.columns if re.fullmatch(r"value_(?:12|24|48|72|96|120|144|168)h", str(c))]
        if cols and len(z):
            ix = rng.choice(len(z), max(1, int(frac*len(z))), replace=False)
            z.loc[z.index[ix], cols] = np.nan
        out[f"missing_{int(frac*100)}pct"] = z
    if "value_24h" in df.columns:
        base = pd.to_numeric(df["value_24h"], errors="coerce")
        z = df.copy(); z["value_24h"] = base * 1.15; out["shifted_mean"] = z
        z = df.copy(); z["value_24h"] = base * 2.0; out["shifted_variance"] = z
        z = df.copy(); z["value_24h"] = base + rng.normal(0, 0.05, len(z)) * base.abs().fillna(1).to_numpy(); out["noise_5pct"] = z
        z = df.copy(); mask = rng.random(len(z)) < 0.2; z.loc[mask, "value_24h"] = np.nan; out["informative_missing"] = z
    z = df.copy()
    value_cols = [c for c in z.columns if re.fullmatch(r"value_\d+(?:\.\d+)?h", str(c))]
    if len(value_cols) >= 3:
        z[value_cols[1]] = np.nan
    out["irregular_readpoints"] = z
    z = df.copy()
    if "parameter" in z.columns: z["parameter"] = "unknown_physical_quantity"
    out["unknown_parameter"] = z
    z = df.copy()
    if "value_24h" in z.columns: z["value_24h"] = pd.to_numeric(z["value_24h"], errors="coerce") * 1000.0
    out["unit_scale"] = z
    return out


def lot_health(df: pd.DataFrame, anomaly: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["lot_id"] if c in df.columns]
    if not keys:
        return pd.DataFrame()
    a = anomaly.copy(); f = forecast.copy()
    a["part_id"] = a["part_id"].astype(str); f["part_id"] = f["part_id"].astype(str)
    parts = df[["part_id", "lot_id"]].drop_duplicates("part_id")
    aa = parts.merge(a[["part_id","anomaly_risk"]] if "anomaly_risk" in a.columns else parts[["part_id"]].assign(anomaly_risk=0.0), on="part_id", how="left")
    ff = parts.merge(f[["part_id","failure_risk"]] if "failure_risk" in f.columns else parts[["part_id"]].assign(failure_risk=0.0), on="part_id", how="left")
    m = aa.merge(ff, on=["part_id","lot_id"], how="outer")
    return m.groupby("lot_id", as_index=False).agg(parts=("part_id","nunique"), mean_anomaly_risk=("anomaly_risk","mean"), mean_failure_risk=("failure_risk","mean"))


def what_if_projection(value_0h: float, value_24h: float, horizon_h: float = 168.0) -> pd.DataFrame:
    slope = (float(value_24h) - float(value_0h)) / 24.0
    t = np.arange(0.0, float(horizon_h) + 1.0, 24.0)
    return pd.DataFrame({"time_h": t, "projected_value": float(value_0h) + slope * t})


def escape_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    y = _binary_truth(x, "future_defective_168h") if "future_defective_168h" in x.columns else _binary_truth(x, "future_defective")
    absfail = _bool_series(x, "absolute_fail_168h")
    x["future_defective"] = y
    x["absolute_pass"] = ~absfail
    x["latent_escape"] = x["absolute_pass"] & y.eq(1)
    x["group"] = np.select([absfail & y.eq(1), absfail & y.eq(0), (~absfail) & y.eq(1), (~absfail) & y.eq(0)], ["obvious_failure", "obvious_absolute_fail_future_safe", "latent_escape", "absolute_pass_future_safe"], default="unknown")
    keep=[c for c in ["part_id","lot_id","group","absolute_pass","future_defective","latent_escape","defect_state","failure_mode","split","high_but_safe"] if c in x.columns]
    return x[keep].copy()


def lead_time_from_screening(canonical: pd.DataFrame, screening: pd.DataFrame, first_flag_h: float = 24.0, horizon_h: float = 168.0) -> pd.DataFrame:
    parts = canonical[["part_id"]].drop_duplicates().copy()
    parts["part_id"] = parts["part_id"].astype(str)
    flagged = screening[screening["decision"].astype(str).isin({"REVIEW","REJECT"})][["part_id"]].drop_duplicates("part_id") if "decision" in screening.columns else pd.DataFrame(columns=["part_id"])
    flagged["first_flag_time_h"] = float(first_flag_h)
    out = parts.merge(flagged,on="part_id",how="left")
    out["lead_time_h"] = float(horizon_h)-pd.to_numeric(out["first_flag_time_h"],errors="coerce")
    return out
