#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python data/reference/sih26170_generator_gen2_final.py --profile final --parts 400 --lots 80 --seed 26170
python scripts/final_backend_gate.py
python scripts/run_benchmark.py --config configs/benchmark.yaml --seeds 20260831 20260832 20260833 --origins 12 24 48 72 96 120 144 168 --bootstrap 1000
python scripts/run_generalization_suite.py
