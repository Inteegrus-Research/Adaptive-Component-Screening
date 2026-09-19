#!/usr/bin/env python3
"""Validation-only diagnostic for the V6 safety policy.

Reads existing validation_triage.csv files only. It does not read test rows,
change thresholds in the project, or train any model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

WEIGHTS={
    "isolation_forest":0.45,
    "temporal":0.25,
    "anomaly_score":0.15,
    "multivariate":0.10,
    "failure_risk":0.05,
}

def num(df,c,default=0.0):
    return pd.to_numeric(df.get(c,pd.Series(default,index=df.index)),errors="coerce").fillna(default).to_numpy(float)

def rank_ref(x,ref,cap=.995):
    ref=np.sort(np.asarray(ref,float)[np.isfinite(ref)])
    if ref.size==0: return np.zeros(len(x))
    z=(np.searchsorted(ref,np.asarray(x,float),side="right")-0.5)/max(len(ref),1)
    z[~np.isfinite(x)]=0.0
    return np.clip(z,0,cap)

def run_one(path,step=.005):
    d=pd.read_csv(path)
    y=pd.to_numeric(d["future_defective"],errors="coerce").fillna(0).astype(int).to_numpy()
    healthy=y==0; defective=y==1
    evidence=np.zeros(len(d))
    for c,w in WEIGHTS.items(): evidence += w*np.clip(num(d,c),0,1)
    evidence=np.clip(evidence,0,1)
    temporal_raw=num(d,"max_acceleration_normalized")
    temporal_ref=np.sort(temporal_raw[healthy][np.isfinite(temporal_raw[healthy])])
    temporal_rank=rank_ref(temporal_raw,temporal_ref)
    risk=np.clip(num(d,"failure_risk"),0,1)
    score=np.maximum.reduce([evidence,risk,temporal_rank])
    rows=[]
    for t in np.unique(np.r_[np.arange(.30,.951,step),np.quantile(score,np.linspace(.30,.995,100))]):
        if not np.isfinite(t): continue
        rows.append({"score_threshold":float(t)})
    # For deployment we use distinct review/reject thresholds; sweep the compact grid.
    out=[]
    for review_t in [r["score_threshold"] for r in rows]:
        for reject_t in [x for x in [0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95] if x>review_t]:
            hard=d.get("hard_violation",pd.Series(False,index=d.index)).astype(bool).to_numpy()
            unc=np.clip(num(d,"forecast_uncertainty",1.0),0,1)
            crossing=d.get("predicted_crossing",pd.Series(False,index=d.index)).astype(bool).to_numpy() & (unc<.40)
            missing=np.clip(num(d,"current_missing_fraction"),0,1)
            unknown=missing>.66
            reject=hard|crossing|(score>=reject_t)
            unknown=unknown & ~reject
            review=(~reject)&(~unknown)&(score>=review_t)
            action=reject|unknown|review
            recall=float(action[defective].mean()) if defective.any() else 0
            fpr=float(action[healthy].mean()) if healthy.any() else 1
            burden=float(review.mean())
            escapes=int((defective&~action).sum())
            violation=max(.90-recall,0)+max(fpr-.10,0)+max(burden-.20,0)+(1.0 if escapes else 0.0)
            out.append({"review_t":float(review_t),"reject_t":float(reject_t),"recall":recall,"fpr":fpr,"review_burden":burden,"critical_escapes":escapes,"violation":violation})
    tab=pd.DataFrame(out).sort_values(["violation","critical_escapes","fpr","review_burden","reject_t","review_t"])
    feasible=tab[(tab.recall>=.90)&(tab.fpr<=.10)&(tab.review_burden<=.20)&(tab.critical_escapes==0)]
    result={
        "rows":int(len(d)),"positives":int(defective.sum()),"negatives":int(healthy.sum()),
        "temporal_reference_n":int(len(temporal_ref)),
        "evidence_pr_auc":float(average_precision_score(y,evidence)) if len(np.unique(y))>1 else None,
        "risk_pr_auc":float(average_precision_score(y,risk)) if len(np.unique(y))>1 else None,
        "temporal_rank_pr_auc":float(average_precision_score(y,temporal_rank)) if len(np.unique(y))>1 else None,
        "v6_decision_score_pr_auc":float(average_precision_score(y,score)) if len(np.unique(y))>1 else None,
        "v6_decision_score_roc_auc":float(roc_auc_score(y,score)) if len(np.unique(y))>1 else None,
        "feasible_points":int(len(feasible)),
        "best":tab.iloc[0].to_dict() if len(tab) else None,
        "best_feasible":feasible.iloc[0].to_dict() if len(feasible) else None,
    }
    tab.to_csv(path.parent/"validation_v6_policy_sweep.csv",index=False)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",default="reports/final_run"); args=ap.parse_args()
    run=Path(args.run_dir); summary=[]
    for sdir in sorted(run.glob("seed_*")):
        p=sdir/"validation_triage.csv"
        if not p.exists():
            print(f"[MISSING] {p}"); continue
        r=run_one(p); r["seed"]=int(sdir.name.removeprefix("seed_")); summary.append(r)
        print(f"\n=== {sdir.name} ===")
        print(json.dumps(r,indent=2,default=str))
    if summary:
        (run/"v6_validation_diagnostic.json").write_text(json.dumps(summary,indent=2))
        pd.DataFrame(summary).to_csv(run/"v6_validation_diagnostic.csv",index=False)
        print(f"\nWROTE: {run/'v6_validation_diagnostic.csv'}")
        print("TEST DATA WAS NOT READ OR USED BY THIS DIAGNOSTIC.")

if __name__=="__main__": raise SystemExit(main())
