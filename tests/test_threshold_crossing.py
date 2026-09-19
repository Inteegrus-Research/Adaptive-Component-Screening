import numpy as np

def test_linear_crossing_math():
    cur, pred, lo, hi = 10.0, 20.0, 0.0, 15.0
    frac=(hi-cur)/(pred-cur); t=24+(168-24)*frac
    assert 24 < t < 168
