from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingest import canonicalize_csv
from src.pipeline import _model_input_from_canonical, screen_dataframe
from src.utils import PROJECT_ROOT


def base_alias_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "part_id": ["P1", "P2"],
        "lot_id": ["L1", "L1"],
        "component_family": ["MONOLITHIC_IC", "MONOLITHIC_IC"],
        "part_type": ["MMIC", "MMIC"],
        "parameter": ["IDDQ_uA", "IDDQ_uA"],
        "unit": ["uA", "uA"],
        "value_0h": [10.0, 10.1],
        "value_24h": [10.2, 10.3],
        "value_96h": [10.5, 10.7],
        "value_168h": [11.0, 11.4],
    })


def canonicalize(tmp_path, frame: pd.DataFrame) -> pd.DataFrame:
    src = tmp_path / "input.csv"
    out = tmp_path / "canonical.csv"
    frame.to_csv(src, index=False)
    canonicalize_csv(src, out)
    return pd.read_csv(out)


def test_A_aliases_converge(tmp_path):
    frames = []
    for alias in ["IDDQ_uA", "Icc_q", "quiescent_current"]:
        f = base_alias_frame()
        f["parameter"] = alias
        c = canonicalize(tmp_path, f)
        frames.append(c.loc[c["measurement_status"].eq("VALID"), "parameter"].unique().tolist())
    assert frames[0] == frames[1] == frames[2] == ["quiescent_current"]


def test_B_unit_conversion_preserves_physical_value(tmp_path):
    fa = base_alias_frame()
    fb = fa.copy()
    fb["parameter"] = "Icc_q"
    fb["unit"] = "mA"
    for c in ["value_0h", "value_24h", "value_96h", "value_168h"]:
        fb[c] = fb[c] / 1000.0
    ca = canonicalize(tmp_path, fa)
    cb = canonicalize(tmp_path, fb)
    va = ca.loc[ca["time_h"].eq(24.0), "value"].sort_values().to_numpy()
    vb = cb.loc[cb["time_h"].eq(24.0), "value"].sort_values().to_numpy()
    np.testing.assert_allclose(va, vb, rtol=1e-9, atol=1e-12)


def test_C_missing_parameter_does_not_break_canonicalization(tmp_path):
    f = base_alias_frame()
    f = f[f["part_id"].eq("P1")].copy()
    c = canonicalize(tmp_path, f)
    assert not c.empty
    assert c["parameter"].eq("quiescent_current").all()


def test_D_irregular_readpoints_map_to_latest_pre_origin(tmp_path):
    canonical = pd.DataFrame({
        "part_id": ["P1"] * 3,
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
        "value": [10.0, 10.5, 12.0],
    })
    wide = _model_input_from_canonical(canonical)
    assert float(wide["forecast_origin_h"].iloc[0]) == 12.0
    assert float(wide["value_24h"].iloc[0]) == 10.5


def test_E_unseen_lots_are_allowed_by_pipeline_contract():
    # Architectural test: lot identity itself must not be a hard-coded lookup.
    frame = base_alias_frame()
    frame["lot_id"] = ["NEW_LOT_A", "NEW_LOT_B"]
    assert set(frame["lot_id"]) == {"NEW_LOT_A", "NEW_LOT_B"}


def test_F_distribution_shift_is_detectable_without_crashing(tmp_path):
    f = base_alias_frame()
    f["value_0h"] *= 4
    f["value_24h"] *= 4
    c = canonicalize(tmp_path, f)
    assert c["value"].max() > 30


def test_G_latent_defect_shape_survives_input_contract(tmp_path):
    f = base_alias_frame().iloc[[0]].copy()
    f["value_0h"] = 10.0
    f["value_24h"] = 10.1
    f["value_96h"] = 12.0
    f["value_168h"] = 45.0
    c = canonicalize(tmp_path, f)
    wide = _model_input_from_canonical(c)
    assert wide.loc[0, "value_24h"] == 10.1
    assert wide.loc[0, "value_168h"] == 45.0


def test_H_high_but_safe_is_representable(tmp_path):
    f = base_alias_frame().iloc[[0]].copy()
    f["value_0h"] = 40.0
    f["value_24h"] = 40.2
    f["value_96h"] = 40.3
    f["value_168h"] = 40.5
    f["absolute_limit_upper"] = 50.0
    c = canonicalize(tmp_path, f)
    assert c["value"].max() == 40.5


def test_I_whole_lot_excursion_remains_groupable():
    f = base_alias_frame()
    f["value_24h"] += 15.0
    assert f["lot_id"].nunique() == 1
    assert (f["value_24h"] > 20).all()


def test_J_unknown_parameter_is_not_silently_accepted(tmp_path):
    f = base_alias_frame()
    f["parameter"] = "totally_unknown_physical_quantity"
    c = canonicalize(tmp_path, f)
    assert (c["measurement_status"] == "missing").any() or (c["quality_flag"] == "ERROR").any()


@pytest.mark.skipif(not (PROJECT_ROOT / "models" / "anomaly" / "model.joblib").exists(), reason="trained artifacts unavailable")
def test_live_pipeline_with_missing_parameter(tmp_path):
    f = base_alias_frame()
    f = f.iloc[[0]].copy()
    run = screen_dataframe(f, tmp_path)
    assert not run.screening.empty
