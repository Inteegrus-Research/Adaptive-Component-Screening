#!/usr/bin/env python3
"""One-command final SIH 26170 run. No test-set tuning or post-hoc threshold edits."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    print("\n$ "+" ".join(map(str,cmd)), flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default='reports/final_submission')
    ap.add_argument('--seeds',nargs='+',type=int,default=[20260831,20260832,20260833])
    ap.add_argument('--origins',nargs='+',type=int,default=[12,24,48,72,96,120,144,168])
    ap.add_argument('--bootstrap',type=int,default=1000)
    args=ap.parse_args()
    gen=ROOT/'data/reference/sih26170_generator_gen2_final.py'
    run([sys.executable,str(gen),'--profile','final','--parts','400','--lots','80','--seed','26170'])
    run([sys.executable,'scripts/run_benchmark.py','--input',str(ROOT/'data/processed/module_A_dataset.csv'),'--output-dir',args.output_dir,'--config',str(ROOT/'configs/benchmark.yaml'),'--seeds',*map(str,args.seeds),'--origins',*map(str,args.origins),'--bootstrap',str(args.bootstrap)])
    run([sys.executable,'scripts/diagnose_scores.py','--run-dir',args.output_dir])
    run([sys.executable,'scripts/inspect_critical_escapes.py','--run-dir',args.output_dir])
    run([sys.executable,'scripts/final_success_gate.py','--run-dir',args.output_dir])

if __name__=='__main__': main()
