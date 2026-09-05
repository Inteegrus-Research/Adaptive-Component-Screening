import numpy as np
import pandas as pd

from src.features import FeatureEngine, validate_features


def canonical_fixture():
    rows = []
    obs = 0
    for lot, base in [("L1", 10.0), ("L2", 20.0)]:
        for part_i in range(3):
            part = f"{lot}_P{part_i}"
            for t, value in [(0, base), (24, base + 1 + part_i), (96, base + 4 + 2 * part_i), (168, base + 8 + 3 * part_i)]:
                obs += 1
                rows.append({
                    "observation_id": f"O{obs}",
                    "source_file": "fixture.csv",
                    "source_row_number": obs,
                    "source_column": "value",
                    "source_format": "wide",
                    "part_id": part,
                    "lot_id": lot,
                    "wafer_id": f"W{lot[-1]}",
                    "test_run_id": f"R{lot[-1]}",
                    "component_family": "MONOLITHIC_IC",
                    "component_type": "DIGITAL_IC",
                    "parameter": "quiescent_current",
                    "raw_parameter": "IDDQ_uA",
                    "semantic_type": "quiescent_current",
                    "physical_quantity": "current",
                    "value": value,
                    "value_original": value,
                    "unit": "uA",
                    "raw_unit": "uA",
                    "unit_conversion_factor": 1.0,
                    "time_h": float(t),
                    "test_stage": "BURN_IN",
                    "test_method": "POWER_BURN_IN",
                    "temperature_C": 125.0,
                    "burnin_temperature_C": 125.0,
                    "voltage_V": 5.0,
                    "current_A": 0.01,
                    "stress_mode": "POWER_BURN_IN",
                    "measurement_status": "VALID",
                    "censoring": "NONE",
                    "quality_flag": "VALID",
                    "mapping_method": "exact_alias",
                    "unit_source": "explicit",
                    "quality_detail": None,
                })
    return pd.DataFrame(rows)


def test_population_temporal_and_quality_features():
    df = canonical_fixture()
    out = FeatureEngine.from_project_config().build(df)
    assert len(out) == len(df)
    for col in ["lot_mean", "lot_median", "lot_mad", "lot_iqr", "lot_robust_z", "lot_percentile", "lot_relative_deviation"]:
        assert col in out.columns
    for col in ["temporal_delta", "temporal_slope", "temporal_acceleration", "temporal_curvature", "normalized_drift", "trajectory_deviation", "change_point_indicator"]:
        assert col in out.columns
    for col in ["measurement_quality_score", "repeat_count", "quality_is_censored"]:
        assert col in out.columns
    ok, errors = validate_features(out)
    assert ok, errors


def test_irregular_and_missing_readpoints_survive():
    df = canonical_fixture()
    df = df[~((df["part_id"] == "L1_P0") & (df["time_h"] == 96.0))].copy()
    df.loc[df.index[0], "value"] = np.nan
    out = FeatureEngine.from_project_config().build(df)
    # Missing value must not be silently converted into a real measurement.
    assert out.loc[out.index[0], "quality_is_missing"] == 1
    # The remaining part has a 0 -> 24 -> 168 irregular trajectory.
    g = out[out["part_id"].eq("L1_P0")].sort_values("time_h")
    assert g["observed_readpoint_count"].max() == 2


def test_repeated_measurements_are_summarized_not_deleted():
    df = canonical_fixture()
    extra = df.iloc[[1]].copy()
    extra["observation_id"] = "REPEAT"
    extra["value"] = extra["value"].iloc[0] + 0.2
    out = FeatureEngine.from_project_config().build(pd.concat([df, extra], ignore_index=True))
    rep = out[out["part_id"].eq(df.iloc[1]["part_id"]) & out["time_h"].eq(df.iloc[1]["time_h"])]
    assert len(rep) == 2
    assert rep["repeat_count"].eq(2).all()
    assert rep["repeat_std"].notna().all()


def test_context_codes_are_deterministic():
    df = canonical_fixture()
    e = FeatureEngine.from_project_config()
    a = e.build(df)
    b = e.build(df)
    assert a["context_parameter_code"].equals(b["context_parameter_code"])
    assert a["context_component_family_code"].equals(b["context_component_family_code"])
