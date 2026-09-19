#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.explain import shap_explain

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--features',required=True); ap.add_argument('--output',required=True); ap.add_argument('--max-rows',type=int,default=250); a=ap.parse_args()
    out=shap_explain(a.model,pd.read_csv(a.features),max_rows=a.max_rows)
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); out.to_csv(p,index=False); print(out.head(15).to_string(index=False)); return 0
if __name__=='__main__': raise SystemExit(main())
