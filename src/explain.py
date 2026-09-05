"""Universal Engineering Explainability Layer.

Translates internal machine learning representations, statistical evidence channels,
and safety policies into a hierarchical, causally ranked engineering audit narrative.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd


def _json(v: Any) -> dict:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return v if isinstance(v, dict) else {}


def _num(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return default if not np.isfinite(x) else x
    except Exception:
        return default


def _trace(row: pd.Series) -> dict:
    t = _json(row.get("trace_json"))
    return t if isinstance(t, dict) else (_json(row.get("audit_trace_json")) if isinstance(_json(row.get("audit_trace_json")), dict) else {})


def _parameter(row: pd.Series, trace: dict) -> str:
    for k in ("parameter", "semantic_type", "physical_quantity"):
        v = row.get(k, None)
        if v is not None and not pd.isna(v) and str(v) not in {"", "nan", "None"}:
            return str(v)
    for path in (("module_a", "parameter"), ("module_b", "parameter")):
        x = trace
        for k in path:
            x = x.get(k) if isinstance(x, dict) else None
        if x:
            return str(x)
    return "measured parameter"


def _unit(row: pd.Series, feature_row: Mapping[str, Any] | None = None) -> str:
    if feature_row and feature_row.get("unit") not in {None, "", "nan"}:
        return str(feature_row.get("unit"))
    return str(row.get("unit")) if row.get("unit") not in {None, "", "nan"} else ""


def _format(v: Any, unit: str = "") -> str:
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "not available"
    if isinstance(v, (int, float, np.integer, np.floating)):
        fv = float(v)
        s = f"{fv:.4g}"
        if fv.is_integer():
            s += ".0"
    else:
        s = str(v)
    return f"{s} {unit}".strip()


def _feature_get(feature_row: Mapping[str, Any] | None, *names: str) -> Any:
    if not feature_row:
        return None
    for n in names:
        if n in feature_row and pd.notna(feature_row[n]):
            return feature_row[n]
    return None


def _counterfactuals(row: pd.Series, trace: dict, feature_row: Mapping[str, Any] | None = None) -> list[str]:
    a = trace.get("module_a", {}) or {}
    f = trace.get("module_b", {}) or {}
    o = trace.get("ood", {}) or {}
    out = []
    param = _parameter(row, trace)
    unit = _unit(row, feature_row)
    value24 = _feature_get(feature_row, "value_24h", "value_asof")
    
    if f.get("limit_cross"):
        lim = _feature_get(feature_row, "absolute_limit_upper", "engineering_limit_upper")
        if value24 is not None:
            out.append(f"If the {param} value at 24 h had remained within the lot-normal range, the future-risk warning would be weaker.")
        elif lim is not None:
            out.append(f"If the projected upper value stayed below {_format(lim, unit)}, the future-limit warning would disappear.")
        else:
            out.append("If the projected upper value remained inside the engineering-safe region, this warning would disappear.")
    elif _num(a.get("population", a.get("population_evidence", 0))) >= 0.65:
        rz = _feature_get(feature_row, "population_robust_z")
        out.append(f"If the {param} measurement returned toward the lot reference population" + (f" (robust deviation near zero rather than {_format(rz)})" if rz is not None else "") + ", the population warning would weaken.")
    elif _num(a.get("temporal", a.get("temporal_evidence", 0))) >= 0.65:
        out.append(f"If the early {param} drift matched the healthy reference trajectory, the temporal warning would weaken.")
    elif o.get("status") in {"MODERATE", "SEVERE"}:
        out.append("If validated reference data for this same component and test domain were available, domain uncertainty would reduce.")
    else:
        out.append("No single variable dominates this decision enough to claim a unique counterfactual.")
    return out[:2]


def _specific_counterfactual(row: pd.Series, trace: dict, feature_row: Mapping[str, Any] | None = None) -> str:
    a = trace.get("module_a", {}) or {}
    f = trace.get("module_b", {}) or {}
    param = _parameter(row, trace)
    unit = _unit(row, feature_row)
    pop = _num(a.get("population", a.get("population_evidence", 0)))
    if pop >= 0.65 and feature_row:
        med = _feature_get(feature_row, "lot_median")
        mad = _feature_get(feature_row, "lot_mad")
        if med is not None and mad is not None:
            scale = 1.4826 * abs(float(mad))
            boundary = float(med) + 3.0 * scale
            return f"If the 24 h {param} value were at or below approximately {_format(boundary, unit)}, the strong lot-relative warning would be reduced."
    if f.get("limit_cross") and feature_row:
        lim = _feature_get(feature_row, "absolute_limit_upper", "engineering_limit_upper")
        if lim is not None:
            return f"If the upper 168 h prediction stayed below {_format(lim, unit)}, the future-limit warning would disappear."
    cf = _counterfactuals(row, trace, feature_row)
    return cf[0] if cf else "No single evidence change was sufficient to define a safe counterfactual."


def _pattern_attribution(row: Mapping[str, Any]) -> dict[str, Any]:
    candidates = []
    vals = [
        ("population", "Lot-relative population shift", "the measurement is unusual compared with peer components"),
        ("temporal", "Abnormal early drift", "the measured trajectory changes faster or differently than the healthy reference"),
        ("multivariate", "Cross-parameter combination anomaly", "the combination of parameters is unusual even when individual values may look acceptable"),
        ("failure_risk", "Elevated future-failure risk", "the forecast suggests potentially unsafe future behaviour"),
        ("ood_score", "Reference-domain novelty", "the system has limited evidence that this case belongs to the learned operating domain"),
    ]
    for key, name, desc in vals:
        v = _num(row.get(key, row.get({"population": "population_evidence", "temporal": "temporal_evidence", "multivariate": "multivariate_component_score", "failure_risk": "failure_risk", "ood_score": "ood_score"}.get(key, key), 0)))
        if v > 0:
            candidates.append({"pattern": name, "score": v, "meaning": desc})
    return {"ranked": sorted(candidates, key=lambda x: x["score"], reverse=True)[:5], "note": "Hierarchical evidence ranking; not arbitrary SHAP attribution."}


def _lead_time(row: pd.Series) -> float | None:
    v = row.get("lead_time_h", np.nan)
    return float(v) if pd.notna(v) and np.isfinite(float(v)) else None


def explain_row(row: pd.Series, feature_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    trace = _trace(row)
    a = trace.get("module_a", {}) or {}
    f = trace.get("module_b", {}) or {}
    o = trace.get("ood", {}) or {}
    q = trace.get("data_quality", {}) or {}
    pol = trace.get("policy", {}) or {}
    
    decision = str(row.get("decision", "UNKNOWN"))
    risk = _num(row.get("risk_score"))
    confidence = str(row.get("confidence", "LOW"))
    param = _parameter(row, trace)
    unit = _unit(row, feature_row)
    
    # 1. Authoritative Physical Evidence
    facts = []
    for label, key in [("Component family", "component_family"), ("Component type", "part_type"), ("Parameter", "parameter"), ("Physical quantity", "physical_quantity"), ("Unit", "unit")]:
        v = _feature_get(feature_row, key) if feature_row else row.get(key)
        if v is not None and not pd.isna(v):
            facts.append(f"{label}: {v}.")
            
    for label, keys in [("Value at 0 h", ("value_0h",)), ("Value at 24 h", ("value_24h", "value_asof")), ("Lot robust deviation", ("population_robust_z",)), ("Early slope", ("slope_0_24", "observable_slope_0_24"))]:
        v = _feature_get(feature_row, *keys)
        if v is not None:
            facts.append(f"{label}: {_format(v, unit)}.")
            
    if a.get("absolute_violation"):
        facts.append("An authoritative engineering limit is violated at the available screening origin.")
    if q.get("status") != "PASS":
        facts.extend([str(x) for x in q.get("reasons", [])[:3]])

    # 2. Ordered Hierarchical Findings
    findings = []
    if a.get("absolute_violation"):
        findings.append("CRITICAL: Absolute engineering limit breached at 24 h.")
    if _num(a.get("temporal", a.get("temporal_evidence"))) >= 0.50:
        findings.append(f"Early drift: precursor acceleration evidence score is {_num(a.get('temporal', a.get('temporal_evidence'))):.2f}.")
    if _num(a.get("population", a.get("population_evidence"))) >= 0.50:
        findings.append(f"Lot comparison: elevated robust deviation ({_num(a.get('population', a.get('population_evidence'))):.2f} MAD above peer median).")
    if _num(a.get("multivariate", a.get("multivariate_component_score"))) >= 0.50:
        findings.append(f"Cross-parameter novelty detected (evidence score: {_num(a.get('multivariate', a.get('multivariate_component_score'))):.2f}).")
    if f.get("target_available") or f.get("max_prediction") is not None:
        findings.append(f"168 h drift forecast: {_format(f.get('max_prediction'), unit)} (90% conformal interval: [{_format(f.get('max_prediction', 0) - f.get('uncertainty', 0), unit)}, {_format(f.get('max_upper'), unit)}]).")
    if f.get("limit_cross"):
        findings.append("The upper forecast bound projects a critical engineering limit exceedance by 168 h.")
    if o.get("status") in {"MODERATE", "SEVERE"}:
        findings.append(f"Domain knowledge OOD status: {o.get('status')}. Contextual or trajectory novelty detected.")

    # 3. Policy Justification
    rules = []
    if decision == "REJECT":
        rules.append("Safety policy dispositioned REJECT based on absolute limit violation or multiple independent high-risk channels.")
    elif decision == "REVIEW":
        rules.append("Safety policy dispositioned REVIEW: suspicious trajectory, OOD novelty, or forecast limit crossing requires human sign-off.")
    elif decision == "UNKNOWN":
        rules.append("Safety policy dispositioned UNKNOWN: data quality gaps or insufficient evidence to confirm flight clearance.")
    else:
        rules.append("Safety policy dispositioned SAFE: all parameter measurements, temporal drift rates, and forecasts are within acceptable statistical boundaries.")

    pattern = _pattern_attribution({**row.to_dict(), **a, **f})
    cf = _counterfactuals(row, trace, feature_row)
    lead = _lead_time(row)
    next_test = row.get("recommended_next_test") or row.get("next_best_test") or "Continue standard screening burn-in protocol."

    headline = {
        "SAFE": "Flight-ready: No material parametric anomalies or drift concerns identified.",
        "REVIEW": "Inspection required: Evidence contains temporal drift or contextual uncertainty requiring engineering review.",
        "REJECT": "Screening rejection: High confidence latent defect precursor or engineering limit breach detected.",
        "UNKNOWN": "Screening incomplete: Missing observations or untrusted data quality prevents automatic disposition."
    }.get(decision, "System cannot establish an automated screening disposition.")

    forecast_sentence = ""
    if f.get("target_available") or f.get("max_prediction") is not None:
        forecast_sentence = f" Predicted 168 h value is {_format(f.get('max_prediction'), unit)} (Upper Bound: {_format(f.get('max_upper'), unit)})."
    if f.get("limit_cross"):
        forecast_sentence += " Critical: The upper conformal bound crosses specification limits."
        
    summary = f"{headline} {param} evaluated across peer population, temporal acceleration, and 168 h drift projection.{forecast_sentence}".strip()
    if cf:
        summary += f" Counterfactual sensitivity: {cf[0]}"

    facts_text = " ".join(facts[:8]) if facts else "No direct telemetry facts supplied."
    findings_text = " ".join(findings[:8]) if findings else "Normal parametric operation across all evidence channels."
    policy_text = " ".join(rules[:5])

    return {
        "part_id": str(row.get("part_id")),
        "decision": decision,
        "risk_score": risk,
        "confidence": confidence,
        "parameter": param,
        "summary": summary,
        "first_year_summary": summary,
        "first_year_explanation": summary,
        "facts": facts[:12],
        "model_findings": findings[:12],
        "policy_reasoning": rules[:8],
        "counterfactuals": cf,
        "specific_counterfactual": _specific_counterfactual(row, trace, feature_row),
        "lead_time_h": lead,
        "lead_time": f"{lead:.1f} h early-warning lead time" if lead is not None else "144.0 h (screened at 24 h)",
        "recommended_next_test": str(next_test),
        "pattern_attribution": pattern,
        "top_reasons": (findings + facts + rules)[:12],
        "audit_trace": trace,
        "facts_plain_english": facts_text,
        "model_findings_plain_english": findings_text,
        "policy_reason_plain_english": policy_text,
        "why_this_decision": policy_text,
    }


def _feature_rows(features: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if features is None or features.empty or "part_id" not in features:
        return {}
    d = features.copy()
    d["part_id"] = d["part_id"].astype(str)
    return {str(pid): g.iloc[0].to_dict() for pid, g in d.groupby("part_id", sort=False)}


def explain_frame(screening: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.DataFrame:
    fmap = _feature_rows(features)
    rows = [explain_row(r, fmap.get(str(r.get("part_id")))) for _, r in screening.iterrows()]
    out = pd.DataFrame(rows)
    for c in ["facts", "model_findings", "policy_reasoning", "counterfactuals", "top_reasons"]:
        out[c] = out[c].apply(lambda x: json.dumps(x, default=str))
    out["pattern_attribution_json"] = out.pop("pattern_attribution").apply(lambda x: json.dumps(x, default=str))
    out["audit_trace_json"] = out.pop("audit_trace").apply(lambda x: json.dumps(x, default=str))
    return out


def shap_explain(model_artifact: str | Path, feature_frame: pd.DataFrame, max_rows: int = 250) -> pd.DataFrame:
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("SHAP is not installed. Install shap to enable supplementary attribution.") from exc
    payload = joblib.load(model_artifact)
    model = payload
    if isinstance(payload, dict):
        if payload.get("primary") is not None:
            model = payload["primary"]
        elif payload.get("parameter_detectors"):
            model = next(iter(payload["parameter_detectors"].values()))
    X = feature_frame.copy().iloc[:max_rows]
    if hasattr(model, "named_steps") and "pre" in model.named_steps and "model" in model.named_steps:
        pre = model.named_steps["pre"]
        est = model.named_steps["model"]
        Xt = pre.transform(X)
        names = list(pre.get_feature_names_out()) if hasattr(pre, "get_feature_names_out") else [f"f{i}" for i in range(Xt.shape[1])]
    else:
        est = model
        Xt = np.asarray(X)
        names = [str(c) for c in X.columns]
        
    try:
        vals = shap.TreeExplainer(est).shap_values(Xt)
    except Exception as exc:
        raise RuntimeError(f"SHAP TreeExplainer failed: {exc}") from exc
    if isinstance(vals, list):
        vals = vals[0]
    vals = np.asarray(vals)
    mean_abs = np.mean(np.abs(vals), axis=0)
    signed = np.mean(vals, axis=0)
    return pd.DataFrame({
        "feature": names[:len(mean_abs)],
        "mean_abs_shap": mean_abs,
        "direction": np.where(signed >= 0, "increases failure risk", "lowers failure risk")
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def render_report(screening: pd.DataFrame, output_path: str | Path, features: pd.DataFrame | None = None) -> pd.DataFrame:
    out = explain_frame(screening, features)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Universal engineering explainability")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("screening")
    r.add_argument("output")
    r.add_argument("--features", default=None)
    s = sub.add_parser("shap")
    s.add_argument("model")
    s.add_argument("features")
    s.add_argument("output")
    a = ap.parse_args()
    if a.cmd == "render":
        out = render_report(pd.read_csv(a.screening), a.output, pd.read_csv(a.features) if a.features else None)
        print(out[["part_id", "decision", "risk_score", "confidence", "summary", "specific_counterfactual", "lead_time", "recommended_next_test"]].head(20).to_json(orient="records", indent=2))
        return 0
    out = shap_explain(a.model, pd.read_csv(a.features))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.head(20).to_json(orient="records", indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
