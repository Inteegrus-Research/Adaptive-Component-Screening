#!/usr/bin/env python3
"""Final backend integrity gate for SIH 26170.

This gate checks software integrity and benchmark prerequisites. It deliberately
never asserts a target recall/FPR/MAE value; empirical performance must come from
an untouched test evaluation.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    ROOT / "src/anomaly.py",
    ROOT / "src/forecast.py",
    ROOT / "src/safety.py",
    ROOT / "src/features.py",
    ROOT / "src/ingest.py",
    ROOT / "src/pipeline.py",
    ROOT / "src/explain.py",
    ROOT / "src/evaluation.py",
    ROOT / "configs/models.yaml",
    ROOT / "configs/parameters.yaml",
    ROOT / "configs/policy.yaml",
    ROOT / "retrain_models.sh",
    ROOT / "scripts/demo_latent_escape.py",
    ROOT / "scripts/generate_attributions.py",
]


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        print("MISSING REQUIRED FILES:")
        print("\n".join(missing))
        return 2

    py_files = [p for p in REQUIRED if p.suffix == ".py"]
    for p in py_files:
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))

    bash = subprocess.run(["bash", "-n", str(ROOT / "retrain_models.sh")], capture_output=True, text=True)
    if bash.returncode:
        print(bash.stderr, file=sys.stderr)
        return bash.returncode

    required_tokens = {
        ROOT / "src/evaluation.py": [
            "false_positive_rate", "false_review_rate", "automatic_reject_burden",
            "high_safe_false_disposition_rate", "normal_aging_false_alarm_rate",
            "LATENT_CHANGE_POINT", "LATENT_ACCELERATING", "LATENT_ABRUPT",
            "progressive_screen_dataframe", "blind_test", "selection_split",
        ],
        ROOT / "src/safety.py": [
            "ood_profile_v4", "0.10 * physical", "0.20 * context", "0.70 * population",
            "set(_operational_schema(profile.expected_columns))",
            "redecide_screening",
        ],
        ROOT / "src/forecast.py": [
            "target_", "target_abs", "baseline_value_0h", "precursor_strength",
            "_persist", "_linear", "nominal_interval_coverage",
        ],
        ROOT / "src/pipeline.py": [
            "value_24h_is_proxy", "feature_availability_manifest", "progressive_screen_dataframe",
        ],
        ROOT / "src/explain.py": ["shap_explain", "specific_counterfactual", "pattern_attribution"],
    }
    for p, tokens in required_tokens.items():
        text = p.read_text(encoding="utf-8")
        absent = [t for t in tokens if t not in text]
        if absent:
            print(f"Missing required implementation markers in {p}: {absent}")
            return 3

    print("FINAL BACKEND GATE: PASS")
    print("No empirical performance target was assumed or asserted.")
    print("Run the blind benchmark to obtain the actual submission metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
