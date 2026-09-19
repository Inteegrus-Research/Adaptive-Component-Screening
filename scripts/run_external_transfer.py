#!/usr/bin/env python3
"""Optional external-transfer smoke test using the same ingestion/pipeline contract."""
from __future__ import annotations
import argparse, io, json, sys
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.pipeline import ACSPipeline

def load_csv(source:str)->pd.DataFrame:
    p=Path(source)
    if p.exists(): return pd.read_csv(p)
    with urlopen(source,timeout=60) as r: return pd.read_csv(io.BytesIO(r.read()))

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output-dir',default='reports/external_transfer'); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    result=ACSPipeline().run(load_csv(args.input),target_horizon_h=168.0)
    result.triage.to_csv(out/'triage.csv',index=False)
    result.feature_table.to_csv(out/'features.csv',index=False)
    (out/'capability_manifest.json').write_text(json.dumps(result.capability_manifest.__dict__,indent=2,default=str),encoding='utf-8')
    print(f'EXTERNAL TRANSFER COMPLETE: {out}')
    return 0
if __name__=='__main__': raise SystemExit(main())
