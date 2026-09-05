import pandas as pd
from src.ingest import Canonicalizer, ParameterOntology, SchemaProfiler, _project_configs, alias_demo, validate_canonical


def objects():
    cfg = _project_configs(); ontology = ParameterOntology(cfg["parameters"])
    return ontology, SchemaProfiler(ontology), Canonicalizer(ontology)


def test_alias_convergence():
    t = alias_demo()
    assert t["canonical_parameter"].tolist() == ["quiescent_current"] * 3


def test_long_units_and_canonical_fields():
    _, p, c = objects()
    df = pd.DataFrame({"Serial": ["P1", "P1"], "Lot": ["L1", "L1"], "Parameter": ["IDDQ", "IDDQ"], "Value": [0.01, 0.011], "Units": ["mA", "mA"], "Hours": [0, 24]})
    out = c.canonicalize(df, p.profile(df), "test.csv")
    assert out["parameter"].eq("quiescent_current").all()
    assert out["unit"].eq("uA").all()
    assert out["value"].tolist() == [10.0, 11.0]
    assert validate_canonical(out)[0]


def test_wide_readpoints():
    _, p, c = objects()
    df = pd.DataFrame({"part_id": ["P1"], "lot_id": ["L1"], "parameter": ["IDDQ_uA"], "value_0h": [10.0], "value_24h": [11.0], "value_168h": [15.0]})
    out = c.canonicalize(df, p.profile(df), "wide.csv")
    assert set(out["time_h"]) == {0.0, 24.0, 168.0}
    assert out["parameter"].eq("quiescent_current").all()


def test_censoring():
    _, p, c = objects()
    df = pd.DataFrame({"part_id": ["P1"], "lot_id": ["L1"], "parameter": ["IIN_LEAK"], "value": [">500"], "unit": ["nA"], "time_h": [24]})
    out = c.canonicalize(df, p.profile(df), "censor.csv")
    assert out.iloc[0]["censoring"] == "GT"
    assert out.iloc[0]["measurement_status"] == "ABOVE_RANGE"


def test_unknown_parameter_is_quarantined():
    _, p, c = objects()
    df = pd.DataFrame({"part_id": ["P1"], "lot_id": ["L1"], "parameter": ["MYSTERY_SENSOR"], "value": [1.0], "unit": ["V"], "time_h": [0]})
    out = c.canonicalize(df, p.profile(df), "unknown.csv")
    assert pd.isna(out.iloc[0]["parameter"])
    assert out.iloc[0]["quality_flag"] == "ERROR"


def test_timestamp_is_inferred_per_part_not_globally():
    _, p, c = objects()
    df = pd.DataFrame({
        "part_id": ["P1", "P1", "P2", "P2"],
        "lot_id": ["L1"] * 4,
        "parameter": ["IDDQ", "IDDQ", "IDDQ", "IDDQ"],
        "value": [10, 11, 20, 21],
        "unit": ["uA"] * 4,
        "timestamp": ["2026-01-01 00:00", "2026-01-02 00:00", "2026-02-01 00:00", "2026-02-02 00:00"],
    })
    out = c.canonicalize(df, p.profile(df), "timestamps.csv")
    assert out["time_h"].tolist() == [0.0, 24.0, 0.0, 24.0]


def test_duplicate_measurements_are_flagged_not_deleted():
    _, p, c = objects()
    df = pd.DataFrame({
        "part_id": ["P1", "P1"], "lot_id": ["L1", "L1"], "parameter": ["IDDQ", "IDDQ"],
        "value": [10.0, 10.1], "unit": ["uA", "uA"], "time_h": [0, 0],
    })
    out = c.canonicalize(df, p.profile(df), "duplicate.csv")
    assert len(out) == 2
    assert out["quality_flag"].eq("LIMITED").all()
    assert out["quality_detail"].str.contains("duplicate_observation_key").all()
