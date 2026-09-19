#!/usr/bin/env python3
"""Counterfactual sensitivity demo using the real current pipeline."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.pipeline import ACSPipeline

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output-dir',default='reports/demo'); ap.add_argument('--part-id',required=True); ap.add_argument('--parameter',required=True); ap.add_argument('--steps',type=int,default=25); args=ap.parse_args()
    raw=pd.read_csv(args.input)
    part=raw.loc[raw['part_id'].astype(str).eq(str(args.part_id)) & raw['parameter'].astype(str).eq(str(args.parameter))].copy()
    if part.empty: raise SystemExit('Requested part/parameter not found.')
    lo=float(pd.to_numeric(part['value'],errors='coerce').min()); hi=float(pd.to_numeric(part['value'],errors='coerce').max())
    base_value=float(pd.to_numeric(part.loc[part['time_h'].eq(24),'value'],errors='coerce').iloc[0]) if part['time_h'].eq(24).any() else hi
    sweep=np.linspace(base_value*0.8,base_value*1.2,max(args.steps,3))
    rows=[]
    pipe=ACSPipeline()
    for v in sweep:
        altered=raw.copy(); mask=altered['part_id'].astype(str).eq(str(args.part_id)) & altered['parameter'].astype(str).eq(str(args.parameter)) & altered['time_h'].eq(24)
        if mask.any(): altered.loc[mask,'value']=v
        result=pipe.run(altered)
        row=result.triage.loc[result.triage['component_id'].astype(str).eq(str(args.part_id)) & result.triage['parameter'].astype(str).eq(str(args.parameter))]
        if not row.empty:
            r=row.iloc[-1]
            rows.append({'value_24h':float(v),'decision':str(r.get('decision')),'anomaly_score':float(r.get('anomaly_score',0)),'failure_risk':float(r.get('failure_risk',0)),'ood_score':float(r.get('ood_score',0))})
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'sensitivity_curve.csv',index=False)
    (out/'counterfactual_summary.json').write_text(json.dumps({'part_id':args.part_id,'parameter':args.parameter,'base_value_24h':base_value,'value_range':[lo,hi]},indent=2),encoding='utf-8')
    print('COUNTERFACTUAL DEMO COMPLETE')
    return 0
if __name__=='__main__': raise SystemExit(main())
