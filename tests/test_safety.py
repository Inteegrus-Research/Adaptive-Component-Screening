from pathlib import Path
import pandas as pd
from src.safety import assess_screening, fit_ood_profile


def _input():
    return pd.DataFrame([
        {"part_id":"P1","lot_id":"L1","parameter":"quiescent_current","component_family":"MONOLITHIC_IC","value_0h":10.0,"value_24h":10.2,"test_method":"POWER_BURN_IN","stress_mode":"BURN_IN","temperature_C":25.0,"burnin_temperature_C":125.0,"voltage_V":5.0},
        {"part_id":"P2","lot_id":"L1","parameter":"quiescent_current","component_family":"MONOLITHIC_IC","value_0h":10.0,"value_24h":25.0,"test_method":"POWER_BURN_IN","stress_mode":"BURN_IN","temperature_C":25.0,"burnin_temperature_C":125.0,"voltage_V":5.0},
    ])


def _anomaly(hard=False, high=False):
    return pd.DataFrame([
        {"part_id":"P1","lot_id":"L1","component_family":"MONOLITHIC_IC","anomaly_score":0.10,"calibrated_risk_score":0.10,"absolute_violation":0,"anomaly_support_count":0,"population_evidence":0.1,"temporal_evidence":0.1,"multivariate_component_score":0.1},
        {"part_id":"P2","lot_id":"L1","component_family":"MONOLITHIC_IC","anomaly_score":0.90 if high else 0.2,"calibrated_risk_score":0.90 if high else 0.2,"absolute_violation":1 if hard else 0,"anomaly_support_count":3 if high else 0,"population_evidence":0.9 if high else 0.1,"temporal_evidence":0.8 if high else 0.1,"multivariate_component_score":0.9 if high else 0.1},
    ])


def _forecast():
    return pd.DataFrame([
        {"part_id":"P1","prediction_168h":10.4,"prediction_lower":9.8,"prediction_upper":11.0,"absolute_limit_upper":50.0,"limit_exceedance_probability_proxy":0.01},
        {"part_id":"P2","prediction_168h":45.0,"prediction_lower":39.0,"prediction_upper":56.0,"absolute_limit_upper":50.0,"limit_exceedance_probability_proxy":0.80},
    ])


def test_ood_profile_and_safe_case(tmp_path):
    inp = _input()
    profile, _ = fit_ood_profile(inp)
    out = assess_screening(_anomaly(), _forecast(), inp, profile)
    p1 = out[out.part_id == "P1"].iloc[0]
    assert p1.decision == "SAFE"
    assert p1.ood_status in {"LOW", "MODERATE"}


def test_hard_limit_is_non_overridable(tmp_path):
    inp = _input(); profile, _ = fit_ood_profile(inp)
    out = assess_screening(_anomaly(hard=True, high=True), _forecast(), inp, profile)
    p2 = out[out.part_id == "P2"].iloc[0]
    assert p2.decision == "REJECT"
    assert bool(p2.hard_limit_violation)


def test_high_risk_is_reject_or_review_not_safe(tmp_path):
    inp = _input(); profile, _ = fit_ood_profile(inp)
    out = assess_screening(_anomaly(high=True), _forecast(), inp, profile)
    p2 = out[out.part_id == "P2"].iloc[0]
    assert p2.decision in {"REJECT", "REVIEW"}
    assert p2.decision != "SAFE"


def test_ood_detects_unseen_semantics_and_schema_change():
    inp = _input()
    profile, _ = fit_ood_profile(inp)
    shifted = inp.copy()
    shifted["component_family"] = "UNSEEN_FAMILY"
    shifted["parameter"] = "unknown_parameter"
    shifted = shifted.drop(columns=["voltage_V"])
    o = __import__("src.safety", fromlist=["assess_data_ood"]).assess_data_ood(profile, shifted)
    assert o["status"] == "SEVERE"
    assert any("unseen" in r.lower() or "missing" in r.lower() for r in o["parts"][0]["reasons"])


def test_bad_quality_produces_unknown():
    inp = _input()
    inp.loc[inp.index.repeat(2), "quality_flag"] = "ERROR"
    profile, _ = fit_ood_profile(inp)
    out = assess_screening(_anomaly(), _forecast(), inp, profile)
    assert set(out["decision"]).issubset({"UNKNOWN", "REVIEW", "REJECT"})
    assert "UNKNOWN" in set(out["decision"])
