#!/usr/bin/env python3
"""Final, leakage-safe benchmark campaign.

CLI wrapper that delegates to the master evaluation engine, ensuring thresholds
are tuned only on validation data and evaluated cleanly on untouched test lots.
"""
import argparse
import json
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import run_master_benchmark

SEED = 20260831

def main():
    ap = argparse.ArgumentParser(description="Run the master evaluation benchmark.")
    ap.add_argument("--input", required=True, help="Path to full canonical/wide dataset with splits")
    ap.add_argument("--output-dir", default="reports/benchmark")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--target-horizon", type=float, default=168.0)
    ap.add_argument("--target-metric", choices=["future_recall", "escape_recall"], default="escape_recall")
    ap.add_argument("--max-reject-rate", type=float, default=0.25)
    args = ap.parse_args()

    result = run_master_benchmark(
        input_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        n_seeds=args.n_seeds,
        target_horizon=args.target_horizon,
        target_metric=args.target_metric,
        max_reject_rate=args.max_reject_rate
    )
    
    serialized_result = {
        k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v) 
        for k, v in result.items()
    }
    print(json.dumps(serialized_result, indent=2, default=str))

if __name__ == "__main__":
    main()


