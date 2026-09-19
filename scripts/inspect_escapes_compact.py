#!/usr/bin/env python3
"""Compact, descriptive inspection of defective SAFE test escapes.

This script is descriptive only. It never tunes thresholds.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

COLS=[
    "part_id","lot_id","component_family","defect_state","future_defective","origin_h",
    "decision","reason_codes","decision_score","evidence_score","failure_risk",
    "temporal_rank","anomaly_score","isolation_forest","temporal","multivariate",
    "max_acceleration_normalized","hard_violation","predicted_crossing","forecast_uncertainty",
    "ood_status","ood_distance","actionable_failure_risk",
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",default="reports/final_run"); ap.add_argument("--top",type=int,default=50); args=ap.parse_args()
    run=Path(args.run_dir)
    all_rows=[]
    for sdir in sorted(run.glob("seed_*")):
        p=sdir/"test_triage.csv"
        if not p.exists(): continue
        d=pd.read_csv(p)
        y=pd.to_numeric(d.get("future_defective",0),errors="coerce").fillna(0).astype(int)
        esc=d[(y==1)&d.decision.astype(str).eq("SAFE")].copy()
        if esc.empty:
            print(f"{sdir.name}: no defective SAFE escapes")
            continue
        keep=[c for c in COLS if c in esc.columns]
        esc=esc[keep].sort_values([c for c in ["decision_score","failure_risk","evidence_score"] if c in esc.columns],ascending=False).head(args.top)
        esc.to_csv(sdir/"critical_escapes_compact.csv",index=False)
        all_rows.append(esc.assign(seed=int(sdir.name.removeprefix("seed_"))))
        print(f"\n=== {sdir.name}: {len(esc)} defective SAFE escapes ===")
        print(esc.to_string(index=False))
    if all_rows:
        pd.concat(all_rows,ignore_index=True).to_csv(run/"critical_escapes_compact_all.csv",index=False)
        print(f"\nWROTE: {run/'critical_escapes_compact_all.csv'}")

if __name__=="__main__": raise SystemExit(main())
