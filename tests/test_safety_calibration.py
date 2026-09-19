import pandas as pd
from src.safety import SafetyPolicy

def test_hard_violation_reject():
    v=pd.DataFrame({"future_defective":[0,0,1,1],"anomaly_score":[.1,.1,.8,.9],"failure_risk":[.1,.1,.8,.9],"max_acceleration_normalized":[.1]*4,"forecast_risk":[0,0,0,1],"hard_violation":[False,False,False,True]})
    p=SafetyPolicy(1).calibrate(v); out=p.apply(v,pd.DataFrame({"ood_status":["LOW"]*4})); assert out.iloc[3].decision=="REJECT"
