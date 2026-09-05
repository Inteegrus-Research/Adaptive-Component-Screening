from pathlib import Path
import sqlite3
import pandas as pd
from src.anomaly import fit_anomaly, score_anomaly
from src.forecast import fit_forecast, predict_forecast


def test_reference_training_smoke(tmp_path):
    db = Path('/mnt/data/sih26170_gen2_reference/sih26170_gen2.db')
    if not db.exists():
        return
    with sqlite3.connect(db) as con:
        state = pd.read_sql_query('SELECT * FROM component_parameter_state', con)
        b = pd.read_sql_query('SELECT * FROM module_B_train', con)
    a_path = tmp_path/'state.csv'; b_path=tmp_path/'b.csv'
    state.to_csv(a_path,index=False); b.to_csv(b_path,index=False)
    am, ametrics = fit_anomaly(a_path, tmp_path/'a.joblib')
    assert (tmp_path/'a.joblib').exists(); assert ametrics['train_rows'] > 100
    scored = score_anomaly(am, a_path)
    assert 'anomaly_score' in scored.columns
    fm, fmetrics, vp = fit_forecast(b_path, tmp_path/'f.joblib')
    assert (tmp_path/'f.joblib').exists(); assert fmetrics['mae'] >= 0
    assert 'prediction_168h' in vp.columns
