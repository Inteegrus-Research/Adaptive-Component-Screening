#!/usr/bin/env python3
"""Honest local sensitivity/counterfactual demo for latent-escape candidates.

The script perturbs only the requested observed parameter toward its 0 h baseline,
re-runs the real pipeline, reports channel-level evidence and finds the smallest
observed-value change that changes the disposition. It never claims causal proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import screen_dataframe, _default_paths


def _pick_column(df: pd.DataFrame, token: str) -> str:
    exact = [c for c in df.columns if str(c).lower() == token.lower()]
    if exact:
        return exact[0]
    raise ValueError(f"Required column {token!r} not found")


def _first_finite(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(x.iloc[0]) if len(x) else None


def _summary(row: pd.Series) -> dict:
    trace = row.get("trace_json", {})
    if isinstance(trace, str):
        try: trace = json.loads(trace)
        except Exception: trace = {}
    a = trace.get("module_a", {}) or {}
    f = trace.get("module_b", {}) or {}
    o = trace.get("ood", {}) or {}
    return {
        "decision": str(row.get("decision")),
        "risk_score": float(row.get("risk_score", 0.0)),
        "population_evidence": float(a.get("population", a.get("population_evidence", 0.0))),
        "temporal_evidence": float(a.get("temporal", a.get("temporal_evidence", 0.0))),
        "multivariate_evidence": float(a.get("multivariate", a.get("multivariate_component_score", 0.0))),
        "failure_risk": float(f.get("failure_risk", 0.0) or 0.0),
        "ood_score": float(o.get("score", 0.0) or 0.0),
        "ood_status": str(o.get("status", "UNKNOWN")),
    }


def run_counterfactual_demo(input_path: str, output_dir: str, part_id: str, parameter: str, steps: int = 25) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    part_df = df[df["part_id"].astype(str).eq(str(part_id))].copy()
    if part_df.empty:
        raise ValueError(f"Part {part_id} not found.")
    if "parameter" in part_df.columns:
        hit = part_df[part_df["parameter"].astype(str).str.lower().eq(str(parameter).lower())]
        if len(hit): part_df = hit.copy()
    value_col = _pick_column(part_df, "value_24h")
    base_col = _pick_column(part_df, "value_0h")
    if len(part_df) != 1:
        # Wide benchmark rows are normally one parameter per part. Select the requested parameter deterministically.
        if "parameter" in part_df.columns:
            hit = part_df[part_df["parameter"].astype(str).str.lower().eq(str(parameter).lower())]
            if len(hit) == 1: part_df = hit.copy()
        if len(part_df) != 1:
            raise ValueError("Counterfactual demo requires exactly one row for the selected part/parameter.")

    original = _first_finite(part_df, value_col); baseline = _first_finite(part_df, base_col)
    if original is None or baseline is None:
        raise ValueError("Selected parameter needs finite value_0h and value_24h observations.")

    outdir = Path(output_dir) / "counterfactual_demo"; outdir.mkdir(parents=True, exist_ok=True)
    artifacts = _default_paths()
    base_run = screen_dataframe(part_df, outdir / "baseline", artifacts=artifacts, as_of_h=24.0, render_explanations=True)
    base = _summary(base_run.screening.iloc[0])

    print(f"--- Counterfactual Sensitivity: {part_id} / {parameter} ---")
    print(f"Baseline disposition at 24 h: {base['decision']}")
    print(f"Original 24 h value: {original:.6g}; 0 h baseline: {baseline:.6g}")
    print("This is an empirical one-variable sensitivity test, not a causal proof.\n")

    values = np.linspace(original, baseline, int(steps) + 1)[1:]
    records = []
    transition = None
    for i, value in enumerate(values, 1):
        trial = part_df.copy(); trial[value_col] = float(value)
        run = screen_dataframe(trial, outdir / f"step_{i:02d}", artifacts=artifacts, as_of_h=24.0, render_explanations=False)
        row = run.screening.iloc[0]
        m = _summary(row); m["step"] = i; m["value_24h"] = float(value); m["delta_from_original"] = float(value-original)
        records.append(m)
        print(f"step={i:02d} value={value:.6g} risk={m['risk_score']:.4f} population={m['population_evidence']:.4f} temporal={m['temporal_evidence']:.4f} multi={m['multivariate_evidence']:.4f} OOD={m['ood_score']:.4f} decision={m['decision']}")
        if m["decision"] != base["decision"] and transition is None:
            transition = m

    result = {"part_id": part_id, "parameter": parameter, "baseline": base, "transition": transition, "records": records}
    pd.DataFrame(records).to_csv(outdir / "sensitivity_curve.csv", index=False)
    (outdir / "counterfactual_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    if transition:
        print(">>> DECISION BOUNDARY FOUND <<<")
        print(f"A simulated 24 h value of {transition['value_24h']:.6g} changed the disposition from {base['decision']} to {transition['decision']}.")
        print(f"Risk: {base['risk_score']:.4f} -> {transition['risk_score']:.4f}")
        print("Channel evidence changed as follows:")
        for k in ("population_evidence", "temporal_evidence", "multivariate_evidence", "failure_risk", "ood_score"):
            print(f"  {k}: {base[k]:.4f} -> {transition[k]:.4f}")
        print(f"If the 24 h {parameter} measurement had been approximately {transition['value_24h']:.6g}, the actual policy would have produced {transition['decision']} under the same artifacts.")
    else:
        print("No disposition transition occurred in the requested sweep.")
        print("The result should be interpreted by inspecting which evidence channel remains dominant; a flat aggregate score is not proof that the perturbed variable is irrelevant.")


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="reports/demo")
    ap.add_argument("--part-id", required=True)
    ap.add_argument("--parameter", required=True)
    ap.add_argument("--steps", type=int, default=25)
    args=ap.parse_args()
    run_counterfactual_demo(args.input, args.output_dir, args.part_id, args.parameter, args.steps)
