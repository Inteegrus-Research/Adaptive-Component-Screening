#!/usr/bin/env python3
"""Final reporting gate. It does not change anything or tune the test set."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",default="reports/final_run"); args=ap.parse_args()
    run=Path(args.run_dir)
    p=run/"benchmark_results_table.csv"
    if not p.exists(): raise SystemExit(f"Missing {p}; run the benchmark first.")
    d=pd.read_csv(p)
    cols=[c for c in ["seed","recall","fpr","precision","pr_auc","review_burden","safe_rate","reject_rate","unknown_rate","critical_escapes","coverage_95","ood_severe_rate"] if c in d.columns]
    print("=== FINAL 3-SEED RESULTS ===")
    print(d[cols].to_string(index=False))
    agg={c:float(d[c].mean()) for c in cols if c!="seed" and pd.api.types.is_numeric_dtype(d[c])}
    print("\n=== MEAN ===")
    print(json.dumps(agg,indent=2))
    checks=[]
    if "recall" in d: checks.append(("mean_recall>=0.90",agg["recall"]>=.90))
    if "fpr" in d: checks.append(("mean_fpr<=0.10",agg["fpr"]<=.10))
    if "review_burden" in d: checks.append(("mean_review<=0.20",agg["review_burden"]<=.20))
    if "critical_escapes" in d: checks.append(("total_critical_escapes==0",float(d.critical_escapes.sum())==0))
    if "coverage_95" in d: checks.append(("mean_coverage>=0.90",agg["coverage_95"]>=.90))
    print("\n=== TARGET GATE ===")
    for name,ok in checks: print(f"{'PASS' if ok else 'NOT_PASS':9s} {name}")
    print("\nThis gate only reports held-out results. It performs no threshold tuning.")

if __name__=="__main__": raise SystemExit(main())
