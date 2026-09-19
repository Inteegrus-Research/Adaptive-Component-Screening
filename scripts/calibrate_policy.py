#!/usr/bin/env python3
"""Validation-only policy calibration smoke command.
The master benchmark performs this calibration internally; this script runs one
seed through the headline origin and persists the generated policy separately.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ingest import ingest_csv
from src.evaluation import _score_pipeline, asdict_policy

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="data/processed/module_A_dataset.csv"); ap.add_argument("--seed",type=int,default=20260831); ap.add_argument("--output",default="models/calibration/safety_policy.json"); a=ap.parse_args()
    df,_,_=ingest_csv(a.input); r=_score_pipeline(df,24,a.seed); p=asdict_policy(r["policy"]); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(p,indent=2),encoding="utf-8"); print(json.dumps(p,indent=2))
if __name__=="__main__":main()
