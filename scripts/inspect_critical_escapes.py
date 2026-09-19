#!/usr/bin/env python3
"""Show component-level defective SAFE escapes from a completed run."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",default="reports/final_run"); ap.add_argument("--top",type=int,default=50); args=ap.parse_args()
    run=Path(args.run_dir); outs=[]
    for sdir in sorted(run.glob("seed_*")):
        p=sdir/"test_triage.csv"
        if not p.exists(): p=sdir/"test_screen.csv"
        if not p.exists(): continue
        d=pd.read_csv(p); y=pd.to_numeric(d.get("future_defective",0),errors="coerce").fillna(0).astype(int); dec=d.get("decision",pd.Series("SAFE",index=d.index)).astype(str)
        esc=d.loc[(y==1)&dec.eq("SAFE")].copy()
        if esc.empty: print(f"{sdir.name}: no defective components classified SAFE"); continue
        esc["seed"]=int(sdir.name.removeprefix("seed_")); esc.sort_values([c for c in ["failure_risk","evidence_score","anomaly_score"] if c in esc.columns],inplace=True)
        esc=esc.head(args.top); esc.to_csv(sdir/"critical_escapes.csv",index=False); outs.append(esc); print(f"\n=== {sdir.name}: {len(esc)} escapes ===\n",esc.to_string(index=False))
    if outs: pd.concat(outs,ignore_index=True).to_csv(run/"critical_escapes_all.csv",index=False)
    return 0
if __name__=="__main__": raise SystemExit(main())
