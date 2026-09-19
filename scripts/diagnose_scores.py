#!/usr/bin/env python3
"""Post-run diagnostics; never tunes the test set."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SCORES=["evidence_score","robust_population","robust_PAT","isolation_forest","temporal","multivariate","anomaly_score","failure_risk","acceleration_normalized","ood_score"]

def _labels(df):
    return pd.to_numeric(df.get("future_defective",0),errors="coerce").fillna(0).astype(int).to_numpy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",default="reports/final_run"); ap.add_argument("--out",default=None); args=ap.parse_args()
    run=Path(args.run_dir); rows=[]
    for sdir in sorted(run.glob("seed_*")):
        p=sdir/"test_triage.csv"
        if not p.exists(): p=sdir/"test_screen.csv"
        if not p.exists(): print(f"[skip] {sdir}"); continue
        df=pd.read_csv(p); y=_labels(df); seed=int(sdir.name.removeprefix("seed_"))
        print(f"\n=== SEED {seed} / TEST DIAGNOSTICS ===")
        print(f"rows={len(df)} positives={int(y.sum())} negatives={int((y==0).sum())}")
        for col in SCORES:
            if col not in df: continue
            x=pd.to_numeric(df[col],errors="coerce").fillna(0).to_numpy(float); ok=np.isfinite(x); xx=x[ok]; yy=y[ok]
            a=float(average_precision_score(yy,xx)) if len(np.unique(yy))>1 else np.nan
            r=float(roc_auc_score(yy,xx)) if len(np.unique(yy))>1 else np.nan
            hm=float(xx[yy==0].mean()) if np.any(yy==0) else np.nan; dm=float(xx[yy==1].mean()) if np.any(yy==1) else np.nan
            print(f"{col:24s} PR-AUC={a:.4f} ROC-AUC={r:.4f} healthy_mean={hm:.4f} defective_mean={dm:.4f} q95={np.quantile(xx,.95):.4f} q99={np.quantile(xx,.99):.4f}")
            rows.append({"seed":seed,"score":col,"pr_auc":a,"roc_auc":r,"healthy_mean":hm,"defective_mean":dm,"q95":np.quantile(xx,.95),"q99":np.quantile(xx,.99)})
        if "failure_risk" in df:
            rr=pd.to_numeric(df.failure_risk,errors="coerce").fillna(0.0); print("risk unique values:",int(rr.nunique()),"of",len(rr))
        if {"anomaly_score","failure_risk"}.issubset(df.columns):
            corr=np.corrcoef(pd.to_numeric(df.anomaly_score,errors="coerce").fillna(0.0),pd.to_numeric(df.failure_risk,errors="coerce").fillna(0.0))[0,1]
            print("corr(anomaly_score, failure_risk):",float(corr))
        if "decision" in df: print("decisions:",df.decision.value_counts(dropna=False).to_dict())
    out=Path(args.out) if args.out else run/"ml_diagnostics.csv"; pd.DataFrame(rows).to_csv(out,index=False); print(f"\nWROTE: {out}"); print("Test diagnostics are descriptive only and were not used for tuning.")

if __name__=="__main__": raise SystemExit(main())
