import numpy as np
import pandas as pd
from src.forecast import _ensure_features, _linear_baseline, evaluate_forecast, _fit_conformal_intervals


def test_forecast_feature_engineering_uses_0_24_only():
    d = pd.DataFrame({"part_id":["a"],"lot_id":["L1"],"value_0h":[10.0],"value_24h":[12.0],"target_168h":[20.0]})
    x = _ensure_features(d)
    assert x.loc[0,"delta_0_24"] == 2.0
    assert x.loc[0,"slope_0_24"] == 2.0/24


def test_linear_baseline():
    d = pd.DataFrame({"value_0h":[10.0],"value_24h":[12.0]})
    p = _linear_baseline(d, 168.0)
    assert np.isclose(p[0], 12.0 + (2.0/24.0)*144.0)


def test_forecast_metrics():
    ref = pd.DataFrame({"target_168h":[10.0,20.0]})
    pred = pd.DataFrame({"prediction_168h":[11.0,18.0],"prediction_lower":[9,17],"prediction_upper":[13,22]})
    m = evaluate_forecast(ref,pred)
    assert m["mae"] == 1.5
    assert 0 <= m["conformal_coverage"] <= 1


def test_conformal_90_uses_upper_residual_tail():
    y = np.arange(100, dtype=float)
    pred = np.zeros(100, dtype=float)
    q = _fit_conformal_intervals(y, pred, coverage=0.90)
    assert q > 80
