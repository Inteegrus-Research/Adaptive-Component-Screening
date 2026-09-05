from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import _model_input_from_canonical, screen_dataframe


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "processed" / "module_A_dataset.csv"
ANOMALY = ROOT / "models" / "anomaly" / "model.joblib"
FORECAST = ROOT / "models" / "forecast" / "model.joblib"
OOD = ROOT / "models" / "calibration" / "ood_profile.joblib"


@pytest.fixture
def tiny_raw() -> pd.DataFrame:
    return pd.DataFrame({
        "part_id": ["P1", "P1", "P1", "P1"],
        "lot_id": ["L1"] * 4,
        "component_family": ["MONOLITHIC_IC"] * 4,
        "part_type": ["MMIC"] * 4,
        "parameter": ["IDDQ_uA"] * 4,
        "unit": ["uA"] * 4,
        "value_0h": [10.0] * 4,
        "value_24h": [10.2] * 4,
        "value_96h": [10.5] * 4,
        "value_168h": [11.0] * 4,
        "absolute_limit_upper": [50.0] * 4,
    })


def test_prepare_canonical_wide_and_irregular_origin():
    canonical = pd.DataFrame({
        "part_id": ["P1", "P1", "P1"],
        "lot_id": ["L1"] * 3,
        "component_family": ["MONOLITHIC_IC"] * 3,
        "component_type": ["MMIC"] * 3,
        "parameter": ["quiescent_current"] * 3,
        "semantic_type": ["quiescent_current"] * 3,
        "physical_quantity": ["current"] * 3,
        "unit": ["uA"] * 3,
        "test_stage": ["BURN_IN"] * 3,
        "test_method": ["POWER_BURN_IN"] * 3,
        "stress_mode": ["TEMPERATURE"] * 3,
        "time_h": [0.0, 12.0, 168.0],
        "value": [10.0, 10.4, 12.0],
    })
    wide = _model_input_from_canonical(canonical)
    assert wide.loc[0, "value_0h"] == 10.0
    assert wide.loc[0, "value_24h"] == 10.4
    assert wide.loc[0, "adapted_forecast_origin"] == 1
    assert wide.loc[0, "forecast_origin_h"] == 12.0
    assert wide.loc[0, "target_168h"] != wide.loc[0, "target_168h"]  # NaN


def test_pipeline_requires_artifacts_when_auto_train_disabled(tmp_path, tiny_raw):
    with pytest.raises(Exception, match="Required model artifacts"):
        screen_dataframe(
            tiny_raw,
            tmp_path,
            artifacts=type("Artifacts", (), {
                "anomaly_model": tmp_path / "a.joblib",
                "forecast_model": tmp_path / "f.joblib",
                "ood_profile": tmp_path / "o.joblib",
            })(),
            auto_train_missing=False,
        )


@pytest.mark.skipif(not (REFERENCE.exists() and ANOMALY.exists() and FORECAST.exists() and OOD.exists()), reason="reference artifacts unavailable")
def test_live_end_to_end_reference(tmp_path):
    # Keep this integration test intentionally small while using the actual trained artifacts.
    raw = pd.read_csv(REFERENCE).head(200)
    run = screen_dataframe(raw, tmp_path)
    assert not run.screening.empty
    required = {"part_id", "decision", "risk_score", "confidence", "ood_status"}
    assert required.issubset(run.screening.columns)
    assert not run.explanations.empty
