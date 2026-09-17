#!/usr/bin/env python3
"""CLI wrapper for the V4 leakage-safe benchmark campaign."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_master_benchmark  # noqa: E402


SEED = 20260831


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, pd.DataFrame):
        return _serialize(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _serialize(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the ACS V4 master benchmark."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV containing train/val/test partitions.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--target-horizon",
        type=float,
        default=168.0,
    )
    parser.add_argument(
        "--target-metric",
        choices=["future_recall", "escape_recall"],
        default="escape_recall",
    )
    parser.add_argument(
        "--max-reject-rate",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional second JSON output path.",
    )
    args = parser.parse_args()

    result = run_master_benchmark(
        input_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        n_seeds=args.n_seeds,
        target_horizon=args.target_horizon,
        target_metric=args.target_metric,
        max_reject_rate=args.max_reject_rate,
    )

    payload = _serialize(result)

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        json_path.write_text(
            json.dumps(
                payload,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            payload,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
