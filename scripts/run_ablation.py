#!/usr/bin/env python3
"""Generate the detector ablation table using the frozen benchmark protocol."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation import run_master_benchmark

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="data/processed/module_A_dataset.csv"); ap.add_argument("--output-dir",default="reports/final_run_ablation"); ap.add_argument("--seed",type=int,default=20260831); a=ap.parse_args()
    run_master_benchmark(Path(a.input),Path(a.output_dir),[a.seed],[24],1000,Path("configs/benchmark.yaml"))
if __name__=="__main__":main()
