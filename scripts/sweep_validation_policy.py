#!/usr/bin/env python3
"""Validation-only operating-point sweep for V5 evidence score."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

WEIGHTS={"isolation_forest":0.45,"temporal":0.25,"anomaly_score":0.15,"multivariate":0.10,"failure_risk":0.05}

def score(df):
    s=np.zeros(len(df))
    for c,w in WEIGHTS.items(): s+=w*pd.to_numeric(df.get(c,0),errors="coerce").fillna(0).to_numpy(float)
    return np.clip(s,0,1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",default="reports/final_run"); ap.add_argument("--grid-step",type=float,default=.005); args=ap.parse_args()
    run=Path(args.run_dir); summary=[]
    for sdir in sorted(run.glob("seed_*")):
        p=sdir/"validation_triage.csv"
        if not p.exists(): p=sdir/"validation_screen.csv"
        if not p.exists(): print(f"{sdir.name}: validation triage missing"); continue
        d=pd.read_csv(p); y=pd.to_numeric(d.get("future_defective",0),errors="coerce").fillna(0).astype(int).to_numpy(); sc=score(d)
        rows=[]
        for t in np.arange(.20,1.0001,args.grid_step):
            flag=sc>=t; rec=float(flag[y==1].mean()) if np.any(y==1) else 0; fpr=float(flag[y==0].mean()) if np.any(y==0) else 1; review=float(flag.mean())
            rows.append({"threshold":round(float(t),4),"recall":rec,"fpr":fpr,"review_burden":review,"critical_escapes":int(((y==1)&~flag).sum())})
        tab=pd.DataFrame(rows); feasible=tab[(tab.recall>=.90)&(tab.fpr<=.10)&(tab.review_burden<=.20)&(tab.critical_escapes==0)]
        if feasible.empty:
            tab["violation"]=np.maximum(.90-tab.recall,0)+np.maximum(tab.fpr-.10,0)+np.maximum(tab.review_burden-.20,0)+(tab.critical_escapes>0).astype(float)
            chosen=tab.sort_values(["violation","fpr","review_burden"]).iloc[0].to_dict(); status="NO_FEASIBLE_POINT"
        else:
            chosen=feasible.sort_values(["recall","fpr","review_burden"],ascending=[False,True,True]).iloc[0].to_dict(); status=f"{len(feasible)} feasible points"
        tab.to_csv(sdir/"validation_evidence_sweep.csv",index=False); (sdir/"validation_policy_candidate.json").write_text(json.dumps({"seed":int(sdir.name.removeprefix("seed_")),"status":status,"selected_validation_policy":chosen,"test_used":False},indent=2)); print(f"\n=== {sdir.name} ===\n{status}\n{json.dumps(chosen,indent=2)}")
        summary.append({**chosen,"seed":int(sdir.name.removeprefix("seed_")),"feasible":not feasible.empty})
    if summary: pd.DataFrame(summary).to_csv(run/"validation_policy_summary.csv",index=False); print(f"\nWROTE: {run/'validation_policy_summary.csv'}")
if __name__=="__main__": raise SystemExit(main())
