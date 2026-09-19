import numpy as np, pandas as pd
from src.failure_risk import FailureRiskModel

def test_risk_range():
    n=40; d=pd.DataFrame({"future_defective":[0]*20+[1]*20,"anomaly_score":np.linspace(0.1,0.9,n),"robust_population":np.linspace(.1,.9,n),"isolation_forest":np.linspace(.1,.9,n),"temporal":np.linspace(.1,.9,n),"multivariate":np.linspace(.1,.9,n),"max_acceleration_normalized":np.linspace(.01,.2,n),"max_slope_normalized":np.linspace(.01,.2,n),"max_change_point_score":np.linspace(0,5,n),"joint_exceedance_count":np.zeros(n),"forecast_risk":np.linspace(0,1,n),"forecast_uncertainty":np.full(n,.1),"predicted_crossing":np.zeros(n),"crossing_time_h":np.full(n,np.nan),"current_missing_fraction":np.zeros(n)})
    m=FailureRiskModel(1).fit(d.iloc[:30],d.iloc[30:]); p=m.predict(d.iloc[30:]); assert np.all((p>=0)&(p<=1))
