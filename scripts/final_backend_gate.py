#!/usr/bin/env python3
"""Fail-fast final repository/benchmark integrity gate."""
from __future__ import annotations
import argparse,re,sys,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default=str(ROOT/"data/processed/module_A_dataset.csv")); ap.add_argument("--benchmark-dir",default=str(ROOT/"reports/final_run")); a=ap.parse_args()
    d=pd.read_csv(a.input); req={"part_id","lot_id","parameter","time_h","value","future_defective","defect_state","split","raw_parameter"}; missing=req-set(d.columns)
    if missing: raise SystemExit(f"GATE FAIL: missing columns {sorted(missing)}")
    if set(d.split.astype(str))!={"train","val","test"}: raise SystemExit("GATE FAIL: missing train/val/test")
    lot_sets={s:set(d.loc[d.split.astype(str).eq(s),"lot_id"].astype(str)) for s in ["train","val","test"]}
    for a1,b1 in [("train","val"),("train","test"),("val","test")]:
        if lot_sets[a1]&lot_sets[b1]: raise SystemExit(f"GATE FAIL: lot leakage {a1}/{b1}")
    if d.duplicated(["part_id","parameter","time_h"]).any(): raise SystemExit("GATE FAIL: duplicate timestamps")
    n_parts=int(d.part_id.nunique()); n_lots=int(d.lot_id.nunique())
    if n_parts < 400 or n_lots < 80:
        raise SystemExit(f"GATE FAIL: final fixture size expected >=400 parts/80 lots, got {n_parts}/{n_lots}. Run the final generator first.")
    meta_path=ROOT/"data/processed/fixture_metadata.json"
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if abs(float(meta.get("defect_base_rate",0.0))-0.25)>1e-9: raise SystemExit("GATE FAIL: final fixture defect base rate is not 0.25")
        if float(meta.get("accelerating_coefficient",0.0))<0.0012 or float(meta.get("accelerating_coefficient",0.0))>0.0020: raise SystemExit("GATE FAIL: accelerating coefficient outside approved final range")
    b=Path(a.benchmark_dir)
    forbidden=["metric_values=","hardcoded recall","hardcoded fpr","progressive_metrics = {"]
    for f in [ROOT/"scripts/run_benchmark.py",ROOT/"src/evaluation.py",ROOT/"scripts/generate_demo_report.py"]:
        txt=f.read_text(errors="ignore").lower()
        for token in forbidden:
            if token in txt:
                raise SystemExit(f"GATE FAIL: prohibited hardcoded metric token {token!r} in {f}")
    required_files=[ROOT/"src/failure_risk.py",ROOT/"src/monitoring.py",ROOT/"configs/benchmark.yaml",ROOT/"scripts/calibrate_policy.py",ROOT/"scripts/run_ablation.py",ROOT/"scripts/run_generalization_suite.py"]
    missing_files=[str(x.relative_to(ROOT)) for x in required_files if not x.exists()]
    if missing_files: raise SystemExit(f"GATE FAIL: missing final architecture files {missing_files}")
    print("FINAL BACKEND GATE: PASS")
if __name__=="__main__":main()
