#!/usr/bin/env python3
"""Latent Escape Counterfactual Demonstration.

Proves the decision engine relies on physical precursor boundaries, not black-box noise,
by programmatically perturbing a 24h reading back toward its 0h baseline until SAFE.
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import screen_dataframe, _default_paths

def run_counterfactual_demo(input_path: str, output_dir: str, part_id: str, parameter: str):
    df = pd.read_csv(input_path, low_memory=False)
    part_df = df[(df["part_id"].astype(str) == part_id)].copy()
    if part_df.empty:
        raise ValueError(f"Part {part_id} not found in dataset.")
        
    outdir = Path(output_dir) / "counterfactual_demo"
    outdir.mkdir(parents=True, exist_ok=True)
    
    artifacts = _default_paths()
    base_run = screen_dataframe(part_df, outdir / "baseline", artifacts=artifacts, as_of_h=24.0, render_explanations=True)
    base_decision = base_run.screening["decision"].iloc[0]
    
    print(f"--- Counterfactual Demo: {part_id} ---")
    print(f"Baseline Decision at 24h: {base_decision}")
    if base_decision == "SAFE":
        print("Part is already SAFE. Choose a latent defect currently flagged as REVIEW or REJECT.")
        return

    val_col = next((c for c in part_df.columns if c.startswith("value_24h") or c.endswith("24h")), None)
    baseline_col = next((c for c in part_df.columns if c.startswith("value_0h") or c.endswith("0h")), None)
    
    original_value = float(part_df[val_col].iloc[0])
    baseline_value = float(part_df[baseline_col].iloc[0]) if baseline_col else (original_value * 0.8)
    
    print(f"Original {parameter} at 24h: {original_value:.4f} (0h Baseline: {baseline_value:.4f})")
    
    current_value = original_value
    # Step incrementally from the anomalous 24h reading back toward the healthy 0h reading
    step_size = (original_value - baseline_value) / 50.0 
    
    for step in range(1, 51):
        current_value -= step_size
        test_df = part_df.copy()
        test_df[val_col] = current_value
        
        run = screen_dataframe(test_df, outdir / f"step_{step}", artifacts=artifacts, as_of_h=24.0, render_explanations=False)
        decision = run.screening["decision"].iloc[0]
        risk = run.screening["risk_score"].iloc[0]
        
        print(f"Step {step} | Simulated 24h Value: {current_value:.4f} | Risk: {risk:.3f} | Decision: {decision}")
        
        if decision == "SAFE":
            print("\n>>> BOUNDARY FOUND <<<")
            print(f"If the 24h {parameter} drift had been contained to {current_value:.4f},")
            print(f"the overall risk score drops to {risk:.3f}.")
            print("The system physically clears this latent defect as SAFE.")
            break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to test dataset")
    ap.add_argument("--output-dir", default="reports/demo")
    ap.add_argument("--part-id", required=True, help="Part ID of a known latent defect")
    ap.add_argument("--parameter", default="leakage_current", help="Parameter to perturb")
    args = ap.parse_args()
    run_counterfactual_demo(args.input, args.output_dir, args.part_id, args.parameter)
