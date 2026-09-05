import pandas as pd
from src.safety import fit_ood_profile, assess_data_ood, assess_screening


def frame():
    rows=[]
    for i in range(50):
        rows.append({'part_id':f'P{i}','lot_id':'L1','component_family':'MONOLITHIC_IC','parameter':'quiescent_current','unit':'uA',
                     'value_0h':10+i*0.01,'value_24h':10.1+i*0.01,'temperature_C':25.0,'burnin_temperature_C':125.0,'voltage_V':5.0})
    return pd.DataFrame(rows)


def test_ood_v3_is_parameter_conditional():
    d=frame(); p,_=fit_ood_profile(d)
    r=assess_data_ood(p,d)
    assert r['status']=='LOW'
    unknown=d.copy(); unknown['parameter']='mystery_quantity'
    rr=assess_data_ood(p,unknown)
    assert rr['status']=='SEVERE'


def test_ood_missing_context_is_not_physical_novelty():
    d=frame().drop(columns=['burnin_temperature_C'])
    p,_=fit_ood_profile(frame())
    r=assess_data_ood(p,d)
    assert all('burnin_temperature_C' not in str(x.get('reasons')) for x in r['parts']) or r['score'] < 1.0
