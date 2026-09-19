#!/usr/bin/env python3
"""Generate a report from real ACS output tables only.

This script intentionally contains no synthetic benchmark numbers. It is a reporting
utility for an already-completed screening/benchmark run.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening", default=None, help="CSV produced by a real screening/benchmark run")
    ap.add_argument("--benchmark", default=None, help="benchmark_results.json")
    ap.add_argument("--output", default=str(ROOT / "reports" / "demo" / "DEMO_REPORT.md"))
    args = ap.parse_args()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)

    parts = None
    if args.screening:
        p = Path(args.screening)
        if not p.exists(): raise SystemExit(f"Missing screening file: {p}")
        parts = pd.read_csv(p)

    result = None
    if args.benchmark:
        p = Path(args.benchmark)
        if not p.exists(): raise SystemExit(f"Missing benchmark JSON: {p}")
        result = json.loads(p.read_text(encoding="utf-8"))

    lines = [
        "# ACS Demonstration / Run Report",
        "",
        "This report is generated strictly from supplied ACS output artifacts. It does not contain fabricated benchmark values.",
        "",
    ]
    if parts is not None:
        lines += [f"## Screening rows: {len(parts)}", ""]
        if "decision" in parts:
            counts = parts["decision"].astype(str).value_counts().to_dict()
            lines += ["### Decision distribution", "", "| Decision | Count |", "|---|---:|"]
            lines += [f"| {k} | {v} |" for k, v in counts.items()]
            lines += [""]
        if "future_defective" in parts and "decision" in parts:
            y = pd.to_numeric(parts["future_defective"], errors="coerce").fillna(0).astype(int)
            flag = parts["decision"].astype(str).isin(["REVIEW", "REJECT", "UNKNOWN"])
            lines += ["### Observed screening metrics", "", f"- Recall (observed): {float((flag & (y == 1)).sum() / max((y == 1).sum(), 1)):.4f}", f"- Healthy action rate: {float((flag & (y == 0)).sum() / max((y == 0).sum(), 1)):.4f}", ""]
    if result is not None:
        lines += ["## Benchmark aggregate", ""]
        lines += ["```json", json.dumps(result.get("headline_aggregate", {}), indent=2, default=str), "```", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
