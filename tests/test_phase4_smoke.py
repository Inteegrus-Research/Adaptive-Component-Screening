from pathlib import Path
import sqlite3
import pandas as pd
from src.safety import fit_ood_profile, assess_screening
from src.explain import explain_frame


def test_reference_phase4_smoke():
    db = Path('/mnt/data/sih26170_gen2_reference/sih26170_gen2.db')
    if not db.exists():
        return
    with sqlite3.connect(db) as con:
        inp = pd.read_sql_query('SELECT * FROM module_A_dataset LIMIT 8000', con)
        try:
            lab = pd.read_sql_query('SELECT * FROM labels', con)
        except Exception:
            lab = pd.DataFrame()
    # The test uses the existing benchmark outputs only if they exist; otherwise
    # construct a minimal model-evidence frame so Phase 4's policy path is tested.
    parts = inp[[c for c in ['part_id','lot_id','component_family','parameter'] if c in inp]].drop_duplicates('part_id').copy()
    anomaly = parts.assign(anomaly_score=0.1, calibrated_risk_score=0.1, absolute_violation=0, anomaly_support_count=0,
                           population_evidence=0.1, temporal_evidence=0.1, multivariate_component_score=0.1)
    forecast = parts.assign(prediction_168h=float('nan'), prediction_lower=float('nan'), prediction_upper=float('nan'),
                            limit_exceedance_probability_proxy=float('nan'))
    profile, _ = fit_ood_profile(inp)
    out = assess_screening(anomaly, forecast, inp, profile)
    assert len(out) == inp.part_id.nunique()
    exp = explain_frame(out)
    assert len(exp) == len(out)
    assert {'decision','risk_score','summary'}.issubset(exp.columns)
