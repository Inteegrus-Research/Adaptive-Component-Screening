import json
import pandas as pd
from src.explain import explain_row


def test_universal_explanation_has_three_layers():
    trace={"module_a":{"population":0.9,"temporal":0.8,"multivariate":0.7,"absolute_violation":False},
           "module_b":{"target_available":True,"max_prediction":43.0,"max_upper":52.0,"limit_cross":True},
           "ood":{"status":"LOW"},"data_quality":{"status":"PASS"}}
    row=pd.Series({'part_id':'P1','decision':'REVIEW','risk_score':0.81,'confidence':'HIGH','trace_json':json.dumps(trace),
                   'reasons_json':json.dumps(['engineering review required'])})
    x=explain_row(row)
    assert x['first_year_explanation']
    assert x['model_findings']
    assert x['why_this_decision']
    assert x['audit_trace']
