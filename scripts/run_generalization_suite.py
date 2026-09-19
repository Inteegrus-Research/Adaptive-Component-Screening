#!/usr/bin/env python3
"""Leakage-safe generalization stress suite.

Two protocols are implemented:
1) leave-one-family-out: the held-out family's train/validation rows are removed;
   only its test-lot rows are scored. Because the family is outside the learned
   anomaly/OOD domain, the primary outcome is safe routing (UNKNOWN/MODERATE OOD),
   not an inflated physical-risk claim.
2) failure-mechanism holdout: the selected mechanism is removed from train/validation,
   while the normal test split remains untouched; this measures whether screening can
   detect a mechanism not used during supervised calibration.

This suite does not modify the canonical benchmark artifacts.
"""
from __future__ import annotations
import argparse, json, tempfile, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ingest import ingest_csv
from src.evaluation import _score_pipeline


def _evaluate(df: pd.DataFrame, seed:int, name:str, output_dir:Path) -> dict:
    tmpdir=Path(tempfile.mkdtemp(prefix="acs_gen_", dir=output_dir))
    csv=tmpdir/f"{name}.csv"
    df.to_csv(csv,index=False)
    long_df,_,_=ingest_csv(csv)
    r=_score_pipeline(long_df,24,seed)
    te=r["test"]
    y=te.future_defective.to_numpy(int)
    actionable=te.decision.astype(str).isin(["REVIEW","REJECT","UNKNOWN"]).to_numpy()
    positives=y==1; negatives=y==0
    return {
        "protocol":name,
        "parts":int(len(te)),
        "positive_parts":int(positives.sum()),
        "recall":float(actionable[positives].mean()) if positives.any() else None,
        "healthy_action_rate":float(actionable[negatives].mean()) if negatives.any() else None,
        "safe_rate":float(te.decision.eq("SAFE").mean()),
        "review_rate":float(te.decision.eq("REVIEW").mean()),
        "reject_rate":float(te.decision.eq("REJECT").mean()),
        "unknown_rate":float(te.decision.eq("UNKNOWN").mean()),
        "ood_moderate_or_severe":float(te.ood_status.isin(["MODERATE","SEVERE"]).mean()),
        "critical_escapes":int(((y==1)&te.decision.astype(str).eq("SAFE")).sum()),
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default=str(ROOT/"data/processed/module_A_dataset.csv"))
    ap.add_argument("--output",default=str(ROOT/"reports/final_run/generalization_suite.json"))
    ap.add_argument("--seed",type=int,default=20260831)
    ap.add_argument("--protocol",choices=["family","mechanism","both"],default="both")
    args=ap.parse_args()
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(args.input)
    rows=[]
    workdir=Path(tempfile.mkdtemp(prefix="acs_generalization_",dir=out.parent))

    if args.protocol in {"family","both"}:
        for fam in sorted(base.component_family.astype(str).unique()):
            x=base.copy()
            # Keep only non-held-out-family train/val and held-out-family test.
            keep=(~x.component_family.astype(str).eq(fam) & x.split.astype(str).isin(["train","val"])) | (x.component_family.astype(str).eq(fam) & x.split.astype(str).eq("test"))
            x=x.loc[keep].copy()
            rows.append(_evaluate(x,args.seed,f"leave_one_family_out::{fam}",workdir))

    if args.protocol in {"mechanism","both"}:
        mechanisms=[m for m in ["LATENT_ACCELERATING","LATENT_CHANGE_POINT","HARD_EARLY_FAILURE"] if m in set(base.defect_state.astype(str))]
        for mech in mechanisms:
            x=base.copy()
            # Remove the held-out mechanism from train/validation; keep test unchanged.
            keep=~(x.defect_state.astype(str).eq(mech) & x.split.astype(str).isin(["train","val"]))
            x=x.loc[keep].copy()
            rows.append(_evaluate(x,args.seed,f"failure_mechanism_holdout::{mech}",workdir))

    shutil_target=[]
    # Remove temporary per-run data after collecting metrics.
    import shutil
    shutil.rmtree(workdir,ignore_errors=True)
    out.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print(json.dumps(rows,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
