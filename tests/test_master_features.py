import json
import numpy as np
import pandas as pd
from src.evaluation import optimize_threshold, leakage_audit, physics_sanity, adversarial_suite, lot_health, what_if_projection


def base():
    rows=[]
    for lot in ['L1','L2']:
        for i in range(10):
            p=f'{lot}-P{i}'
            rows.append({'part_id':p,'lot_id':lot,'component_family':'MONOLITHIC_IC','parameter':'quiescent_current','unit':'uA',
                         'value_0h':10+i*0.01,'value_24h':10.2+i*0.01,'value_96h':10.5+i*0.01,'value_168h':11+i*0.01,
                         'absolute_limit_upper':50.0,'temperature_C':25.0,'burnin_temperature_C':125.0,'voltage_V':5.0})
    return pd.DataFrame(rows)


def test_threshold_prefers_low_fn():
    y=[0,0,1,1]; s=[0.1,0.2,0.7,0.8]
    r=optimize_threshold(y,s,fn_cost=100,fp_cost=1)
    assert r.fn==0 and r.recall==1


def test_leakage_audit_catches_future_columns():
    d=base().drop(columns=['value_96h','value_168h']); assert leakage_audit(d)['leakage_detected'] is False
    assert leakage_audit(base())['leakage_detected'] is True


def test_physics_sanity():
    assert physics_sanity(base())['pass'] is True
    d=base(); d.loc[0,'temperature_C']=5000
    assert physics_sanity(d)['pass'] is False


def test_adversarial_suite_has_core_scenarios():
    s=adversarial_suite(base())
    for k in ['baseline','renamed','missing_20pct','missing_40pct','missing_60pct','shifted_mean','shifted_variance','noise_5pct','informative_missing','irregular_readpoints']:
        assert k in s


def test_lot_health_and_projection():
    d=base(); a=pd.DataFrame({'part_id':d.part_id,'anomaly_risk':0.1}); f=pd.DataFrame({'part_id':d.part_id,'failure_risk':0.05})
    lh=lot_health(d,a,f); assert len(lh)==2
    proj=what_if_projection(10,11); assert proj.iloc[-1].projected_value==17.0
