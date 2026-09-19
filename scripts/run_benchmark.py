#!/usr/bin/env python3
"""Canonical final benchmark entry point."""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from src.evaluation import run_master_benchmark

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default=str(ROOT/"data/processed/module_A_dataset.csv"))
    ap.add_argument("--output-dir",default=str(ROOT/"reports/final_run"))
    ap.add_argument("--config",default=str(ROOT/"configs/benchmark.yaml"))
    ap.add_argument("--seeds",nargs="+",type=int,default=[20260831,20260832,20260833])
    ap.add_argument("--origins",nargs="+",type=int,default=[12,24,48,72,96,120,144,168])
    ap.add_argument("--bootstrap",type=int,default=1000)
    args=ap.parse_args()
    print(f"configuration_sha256: {__import__('hashlib').sha256(Path(args.config).read_bytes()).hexdigest()}")
    result=run_master_benchmark(Path(args.input),Path(args.output_dir),args.seeds,args.origins,args.bootstrap,Path(args.config))
    print(json.dumps(result.get("headline_aggregate",{}),indent=2,default=str))
    print(f"FINAL BENCHMARK COMPLETE: {args.output_dir}")

if __name__=="__main__": main()
