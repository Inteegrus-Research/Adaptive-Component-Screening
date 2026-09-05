import pandas as pd
from src.anomaly import _canonicalize_state, _robust_lot_features, _robust_score_to_probability, aggregate_part_anomaly


def test_anomaly_features_respect_as_of_boundary():
    d = pd.DataFrame({
        "part_id": ["a"], "lot_id": ["L1"], "parameter": ["IDDQ_uA"],
        "value_0h": [10.0], "value_24h": [12.0], "value_96h": [99.0], "value_168h": [999.0]
    })
    x = _canonicalize_state(d, as_of_h=24)
    assert x.loc[0, "value_asof"] == 12.0
    assert "value_168h" not in x.columns or x.loc[0].get("value_168h", None) == 999.0


def test_lot_relative_features_are_finite_for_normal_population():
    d = pd.DataFrame({
        "part_id": [str(i) for i in range(20)], "lot_id": ["L1"]*20,
        "component_family": ["MONOLITHIC_IC"]*20, "parameter": ["IDDQ_uA"]*20,
        "value_asof": [10+i*0.1 for i in range(20)]
    })
    x = _robust_lot_features(d)
    assert x["population_robust_z"].notna().sum() >= 19
    assert x["population_percentile"].between(0,1).all()


def test_probability_bounds():
    z = pd.Series([-10, 0, 10], dtype=float)
    p = _robust_score_to_probability(z)
    assert ((p >= 0) & (p <= 1)).all()


def test_part_aggregation_is_conservative():
    d = pd.DataFrame({
        "part_id": ["a","a","b"], "lot_id": ["L","L","L"], "component_family": ["X"]*3,
        "part_type": ["Y"]*3, "parameter": ["p1","p2","p1"],
        "anomaly_score": [0.2,0.9,0.4], "population_anomaly_score":[0.1,0.8,0.3],
        "temporal_anomaly_score":[0.2,0.7,0.4], "multivariate_anomaly_score":[0.1,0.9,0.4],
        "anomaly_support_count":[0,3,1]
    })
    x=aggregate_part_anomaly(d).set_index("part_id")
    assert x.loc["a","anomaly_score"] == 0.9
