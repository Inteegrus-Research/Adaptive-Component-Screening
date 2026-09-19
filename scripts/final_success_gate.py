#!/usr/bin/env python3
"""Final submission gate. Reports actual held-out metrics; never tunes them."""
from __future__ import annotations
import argparse,json
from pathlib import Path

TARGETS={
    'recall':('>=',0.90),
    'fpr':('<=',0.10),
    'review_burden':('<=',0.20),
    'critical_escapes':('==',0.0),
    'coverage_95':('>=',0.90),
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',default='reports/final_submission'); a=ap.parse_args()
    p=Path(a.run_dir)/'benchmark_results.json'
    if not p.exists(): raise SystemExit(f'FINAL GATE FAIL: missing {p}')
    data=json.loads(p.read_text()); m=data.get('headline_aggregate',{}).get('mean',{})
    print('=== FINAL SUBMISSION GATE ===')
    ok=True
    for k,(op,t) in TARGETS.items():
        v=float(m.get(k,float('nan')))
        passed=(v>=t) if op=='>=' else (v<=t) if op=='<=' else abs(v-t)<1e-12
        ok &= passed
        print(f'{k}: {v}  target {op} {t}  [{"PASS" if passed else "FAIL"}]')
    if not ok: raise SystemExit('FINAL GATE FAIL: measured held-out aggregate does not meet all submission targets.')
    print('FINAL GATE PASS: submission metrics meet all configured targets.')

if __name__=='__main__': main()
