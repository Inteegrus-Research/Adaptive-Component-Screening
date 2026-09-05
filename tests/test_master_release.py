import pandas as pd
from src.evaluation import adversarial_suite, leakage_audit, optimize_threshold, escape_matrix
from src.explain import explain_frame


def test_master_threshold_attribute_and_fn_priority():
    r=optimize_threshold([0,0,1,1],[0.1,0.2,0.7,0.8],fn_cost=100,fp_cost=1,max_reject_rate=1.0)
    assert r.fn == 0 and r.recall == 1


def test_adversarial_alias_exists():
    d=pd.DataFrame({'part_id':['P1'],'lot_id':['L1'],'parameter':['x'],'value_0h':[1.0],'value_24h':[1.1]})
    out=adversarial_suite(d)
    assert 'unknown_parameter' in out and 'unit_scale' in out


def test_leakage_field_is_explicit():
    d=pd.DataFrame({'value_0h':[1.0],'value_24h':[1.1],'value_168h':[2.0]})
    r=leakage_audit(d,24)
    assert r['leakage_detected'] is True


def test_escape_matrix_marks_absolute_pass_future_defect():
    d=pd.DataFrame({'part_id':['P1'],'lot_id':['L1'],'future_defective':[1],'absolute_fail_168h':[0]})
    assert escape_matrix(d).iloc[0]['group'] == 'latent_escape'


def test_explanation_has_universal_fields():
    d=pd.DataFrame([{
        'part_id':'P1','decision':'REVIEW','risk_score':0.72,'confidence':'HIGH',
        'trace_json':'{"module_a":{"population":0.8},"module_b":{"max_prediction":43.0,"max_upper":51.0,"limit_cross":true}}'
    }])
    out=explain_frame(d)
    for c in ['summary','specific_counterfactual','lead_time','recommended_next_test','pattern_attribution_json','audit_trace_json']:
        assert c in out.columns
