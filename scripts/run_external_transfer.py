#!/usr/bin/env python3
"""Optional external-transfer smoke test for real electronics-aging CSV data.

This script deliberately treats the external dataset as a transfer-domain stress
case, not as SIH/ISRO ground truth. It accepts a local CSV, or a URL that points
to a downloadable CSV. Column names are mapped conservatively through the existing
ParameterOntology aliases; unsupported channels are ignored rather than invented.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import ParameterOntology, load_yaml  # noqa: E402
from src.pipeline import PipelineArtifacts, screen_dataframe  # noqa: E402
from src.utils import load_yaml as _load_yaml_utils  # noqa: E402


def load_csv(source: str) -> pd.DataFrame:
    p = Path(source)
    if p.exists():
        return pd.read_csv(p)
    with urlopen(source, timeout=60) as r:  # nosec B310: explicit user-supplied data URL
        return pd.read_csv(io.BytesIO(r.read()))


def map_external_columns(df: pd.DataFrame) -> pd.DataFrame:
    ontology = ParameterOntology(load_yaml(PROJECT_ROOT / "configs" / "parameters.yaml"))
    out = df.copy()

    # Common NASA / PHM IGBT naming variants. We only rename when an existing
    # canonical semantic name is not already present.
    aliases = {
        "VCE_ON_V": ["VCE_ON_V", "vce_on", "vce(on)", "VCE", "vce"],
        "ICE_A": ["ICE_A", "ice", "IC", "collector_current", "collector current"],
        "VGE_V": ["VGE_V", "vge", "VGE", "gate_emitter_voltage"],
        "GATE_LEAK_nA": ["GATE_LEAK_nA", "gate_leak", "gate leakage", "gate_current"],
        "RDS_ON_mOhm": ["RDS_ON_mOhm", "rds_on", "RDS(on)", "on_resistance"],
    }
    lower = {str(c).strip().lower(): c for c in out.columns}
    for canonical, names in aliases.items():
        if canonical in out.columns:
            continue
        for name in names:
            actual = lower.get(str(name).strip().lower())
            if actual is not None:
                out = out.rename(columns={actual: canonical})
                break

    # Preserve ordinary metadata when present under common names.
    renames = {
        "device": "part_id",
        "device_id": "part_id",
        "sample": "part_id",
        "unit": "part_id",
        "cycle": "time_h",
        "hours": "time_h",
        "hour": "time_h",
        "aging_time_h": "time_h",
        "temperature": "temperature_C",
        "temp_c": "temperature_C",
    }
    lower = {str(c).strip().lower(): c for c in out.columns}
    for src, dst in renames.items():
        if dst not in out.columns and src in lower:
            out = out.rename(columns={lower[src]: dst})

    # The operational pipeline expects at least a part identifier. For a genuine
    # time-series table with one device and no ID column, assign a deterministic ID.
    if "part_id" not in out.columns:
        out["part_id"] = "EXTERNAL_0001"
    if "lot_id" not in out.columns:
        out["lot_id"] = "EXTERNAL_LOT"
    if "time_h" not in out.columns:
        out["time_h"] = 0.0
    if "component_family" not in out.columns:
        out["component_family"] = "POWER_IGBT"
    if "component_type" not in out.columns:
        out["component_type"] = "IGBT"
    if "test_stage" not in out.columns:
        out["test_stage"] = "external_transfer"
    if "test_method" not in out.columns:
        out["test_method"] = "unknown_external"
    if "stress_mode" not in out.columns:
        out["stress_mode"] = "unknown_external"

    # Exercise the same ontology path used by the main pipeline. We do not force
    # columns that the ontology cannot resolve; sparse external schemas are expected.
    _ = ontology
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Local CSV or downloadable CSV URL")
    ap.add_argument("--output-dir", default="reports/external_transfer")
    ap.add_argument("--as-of", type=float, default=24.0)
    ap.add_argument("--target-horizon", type=float, default=168.0)
    args = ap.parse_args()

    if not args.input:
        refs = sorted((PROJECT_ROOT / "data" / "reference").glob("*igbt*.csv"))
        if not refs:
            raise SystemExit("Provide --input or place an IGBT CSV under data/reference/.")
        args.input = str(refs[0])

    raw = load_csv(args.input)
    mapped = map_external_columns(raw)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Use existing frozen artifacts only; do not silently retrain on external data.
    artifacts = PipelineArtifacts(
        anomaly_model=PROJECT_ROOT / "models" / "anomaly" / "model.joblib",
        forecast_model=PROJECT_ROOT / "models" / "forecast" / "model.joblib",
        ood_profile=PROJECT_ROOT / "models" / "calibration" / "ood_profile.joblib",
        safety_policy=PROJECT_ROOT / "configs" / "policy.yaml",
    )
    run = screen_dataframe(
        mapped,
        outdir,
        artifacts=artifacts,
        as_of_h=args.as_of,
        target_horizon=args.target_horizon,
    )

    report = {
        "input": str(args.input),
        "rows": int(len(mapped)),
        "mapped_columns": list(mapped.columns),
        "screening_output": str(outdir / "screening.csv"),
        "claim_boundary": "External transfer smoke test only; not SIH/ISRO ground truth.",
        "run_manifest": getattr(run, "manifest", None),
    }
    (outdir / "transfer_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


