from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'demo_report'
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / 'benchmark_kaggle').mkdir(parents=True)

rng = np.random.default_rng(26170)
times = np.array([0, 12, 24, 48, 72, 96, 120, 144, 168], dtype=float)

param_meta = {
    'quiescent_current': ('current', 'uA', 85.0, 115.0, 'MONOLITHIC_IC', 'ANALOG_IC'),
    'input_leakage_current': ('current', 'nA', 38.0, 62.0, 'MONOLITHIC_IC', 'MMIC'),
    'supply_operating_current': ('current', 'mA', 14.0, 19.0, 'MONOLITHIC_IC', 'REGULATOR'),
    'propagation_delay': ('time', 'ns', 6.0, 10.0, 'MONOLITHIC_IC', 'DIGITAL_IC'),
    'threshold_voltage': ('voltage', 'V', 1.05, 1.65, 'DISCRETE_SEMICONDUCTOR', 'POWER_MOSFET'),
    'drain_source_leakage': ('current', 'uA', 9.0, 18.0, 'DISCRETE_SEMICONDUCTOR', 'DIODE'),
    'drain_source_on_resistance': ('resistance', 'mOhm', 48.0, 78.0, 'DISCRETE_SEMICONDUCTOR', 'POWER_MOSFET'),
    'collector_emitter_on_voltage': ('voltage', 'V', 1.2, 2.1, 'DISCRETE_SEMICONDUCTOR', 'IGBT'),
    'capacitance': ('capacitance', 'nF', 9.2, 11.8, 'CAPACITOR', 'CERAMIC_MLCC'),
    'leakage_current': ('current', 'uA', 1.8, 4.2, 'CAPACITOR', 'SOLID_TANTALUM'),
    'esr': ('resistance', 'Ohm', 0.42, 0.95, 'CAPACITOR', 'SOLID_TANTALUM'),
    'impedance': ('impedance', 'Ohm', 1.4, 2.8, 'CAPACITOR', 'CERAMIC_MLCC'),
    'burnin_temperature': ('temperature', 'C', 118.0, 132.0, 'POWER_REG', 'REGULATOR'),
    'applied_voltage': ('voltage', 'V', 4.5, 5.5, 'POWER_REG', 'REGULATOR'),
    'applied_current': ('current', 'mA', 240.0, 330.0, 'POWER_REG', 'REGULATOR'),
}

family_specs = {
    'DIGITAL_LOGIC': ['quiescent_current','input_leakage_current','propagation_delay'],
    'MMIC_ANALOG': ['quiescent_current','input_leakage_current','supply_operating_current'],
    'POWER_REG': ['supply_operating_current','burnin_temperature','applied_current'],
    'CAPACITOR': ['capacitance','leakage_current','esr'],
    'SENSOR_IF': ['input_leakage_current','supply_operating_current','propagation_delay'],
    'POWER_SEMICONDUCTOR': ['threshold_voltage','drain_source_leakage','drain_source_on_resistance'],
}

types = {
    'DIGITAL_LOGIC':'DIGITAL_IC','MMIC_ANALOG':'MMIC','POWER_REG':'REGULATOR','CAPACITOR':'CERAMIC_MLCC','SENSOR_IF':'ADC','POWER_SEMICONDUCTOR':'POWER_MOSFET'
}

records=[]; feature_rows=[]; anomaly_rows=[]; forecast_rows=[]; screening_rows=[]; explanation_rows=[]
state_for_part={}
part_counter=1

# Deliberate demonstration classes: healthy, review-latent, review-change-point,
# reject-early, unknown-quality. The values are synthetic presentation data.
classes = ['HEALTHY_STABLE']*38 + ['LATENT_ACCELERATING']*9 + ['LATENT_CHANGE_POINT']*6 + ['HARD_EARLY_FAILURE']*4 + ['UNKNOWN_DATA_QUALITY']*3
rng.shuffle(classes)

for family, params in family_specs.items():
    for i in range(10):
        state = classes[(part_counter-1) % len(classes)]
        parameter = params[i % len(params)]
        _, unit, lo, hi, canonical_family, component_type = param_meta.get(parameter, ('unknown','unit',1,2,family,types[family]))
        if family == 'SENSOR_IF':
            component_type='ADC'
        if family == 'POWER_SEMICONDUCTOR':
            component_type='POWER_MOSFET'
        part_id=f'{family}_{part_counter:03d}'
        lot=f'LOT-{1+(part_counter-1)//10:02d}'
        test_run=f'RUN-26-{1+(part_counter-1)//20:02d}'
        profile='PWR-BI-168H-A'
        base=lo + (hi-lo)*float(rng.uniform(0.28,0.62))
        nominal_slope=(hi-lo)/168.0 * float(rng.uniform(-0.06,0.08))
        # make selected hero components explicit and memorable
        if part_id.endswith('002') and family=='MMIC_ANALOG':
            parameter='input_leakage_current'; unit='nA'; lo,hi=38.0,62.0; base=43.0; state='LATENT_ACCELERATING'
        if part_id.endswith('007') and family=='MMIC_ANALOG':
            parameter='supply_operating_current'; unit='mA'; lo,hi=14.0,19.0; base=16.2; state='HARD_EARLY_FAILURE'
        if part_id.endswith('004') and family=='CAPACITOR':
            parameter='leakage_current'; unit='uA'; lo,hi=1.8,4.2; base=2.15; state='LATENT_CHANGE_POINT'
        if parameter=='threshold_voltage':
            lo,hi=1.05,1.65
        limit_upper=hi
        limit_lower=lo
        vals=[]
        for t in times:
            noise=(hi-lo)*0.010
            if state=='HEALTHY_STABLE':
                v=base + nominal_slope*t + rng.normal(0,noise)
            elif state=='LATENT_ACCELERATING':
                v=base + nominal_slope*t + (hi-lo)*0.00034*(t**1.65) + rng.normal(0,noise*1.2)
            elif state=='LATENT_CHANGE_POINT':
                extra=max(0,t-72)*(hi-lo)/1500.0
                v=base + nominal_slope*t + extra + rng.normal(0,noise*1.15)
            elif state=='HARD_EARLY_FAILURE':
                v=base + (hi-lo)*0.0135*t + rng.normal(0,noise*0.8)
                if t>=24: v += (hi-lo)*0.10
            else:
                v=base + nominal_slope*t + rng.normal(0,noise)
                if t in (72,144): v=np.nan
            vals.append(float(v) if math.isfinite(v) else np.nan)
        vals=np.array(vals,float)
        # Keep values plausible and create a selected late-crossing demo component.
        if state=='HARD_EARLY_FAILURE': vals[2]=min(hi*1.03, vals[2]); vals[3:]=np.maximum(vals[3:], hi*1.05)
        if part_id=='MMIC_ANALOG_002': vals=np.array([42.8,43.6,44.4,46.2,48.0,50.7,54.2,57.9,60.8])
        if part_id=='MMIC_ANALOG_007': vals=np.array([16.1,16.5,18.3,20.0,20.6,21.1,21.4,21.7,21.8])
        if part_id=='CAPACITOR_004': vals=np.array([2.10,2.12,2.18,2.19,2.23,2.41,2.92,3.38,3.75])

        v0=float(vals[0]) if np.isfinite(vals[0]) else None
        v24=float(vals[2]) if np.isfinite(vals[2]) else None
        valid=[v for v in vals if np.isfinite(v)]
        median=float(np.median(valid))
        mad=float(np.median(np.abs(np.array(valid)-median))) or max((hi-lo)*0.02,1e-6)
        slope=float((v24-v0)/24.0) if v0 is not None and v24 is not None else 0.0
        slope_late=float((valid[-1]-v24)/(168-24)) if v24 is not None else 0.0
        robust_z=float((v24-median)/(1.4826*mad)) if v24 is not None else 0.0
        pop_score=min(1.0,abs(robust_z)/4.0)
        temporal_score=min(1.0,abs(slope_late-slope)/(max(abs(hi-lo)/168.0,1e-6))*0.35)
        if state=='LATENT_ACCELERATING': temporal_score=max(temporal_score,.72)
        if state=='LATENT_CHANGE_POINT': temporal_score=max(temporal_score,.58)
        multivariate=min(1.0,0.12+0.58*pop_score+0.22*temporal_score+rng.uniform(0,.08))
        abs_score=0.98 if state=='HARD_EARLY_FAILURE' else float(np.clip(max(0,(max(valid)-hi)/(max(hi-lo,1e-6))*8),0,1))
        ood=0.04 if state!='UNKNOWN_DATA_QUALITY' else 0.58
        ood_status='LOW' if ood<0.25 else 'MODERATE'
        anomaly_risk=float(np.clip(0.28*pop_score+0.34*temporal_score+0.28*multivariate+0.10*abs_score,0,1))
        uncertainty=0.08 if state!='UNKNOWN_DATA_QUALITY' else 0.46
        failure_risk=float(np.clip(0.65*anomaly_risk+0.28*ood+0.07*uncertainty,0,1))
        if state=='HARD_EARLY_FAILURE': risk=0.96; decision='REJECT'; confidence='HIGH'
        elif state=='LATENT_ACCELERATING': risk=max(0.76,failure_risk+0.15); decision='REVIEW'; confidence='MODERATE'
        elif state=='LATENT_CHANGE_POINT': risk=max(0.62,failure_risk+0.08); decision='REVIEW'; confidence='MODERATE'
        elif state=='UNKNOWN_DATA_QUALITY': risk=0.48; decision='UNKNOWN'; confidence='LOW'
        else: risk=float(np.clip(failure_risk,0.08,0.42)); decision='SAFE'; confidence='HIGH'
        forecast_extra=0.0
        if state=='LATENT_ACCELERATING': forecast_extra=(hi-lo)*0.22
        elif state=='LATENT_CHANGE_POINT': forecast_extra=(hi-lo)*0.10
        elif state=='HARD_EARLY_FAILURE': forecast_extra=(hi-lo)*0.14
        elif state=='UNKNOWN_DATA_QUALITY': forecast_extra=0
        last=valid[-1]
        persistence=last + rng.normal(0,(hi-lo)*0.012)
        linear=last + slope_late*(168-max(times[np.isfinite(vals)]))
        ridge=last + slope_late*125
        gb=last + slope_late*144 + forecast_extra
        if part_id=='MMIC_ANALOG_002':
            persistence=60.2; linear=67.8; ridge=68.5; gb=72.9
        if part_id=='MMIC_ANALOG_007':
            persistence=21.8; linear=24.3; ridge=23.9; gb=24.8
        prediction=float(np.clip(gb if decision!='SAFE' else persistence + rng.normal(0,(hi-lo)*0.02), lo*0.7, hi*1.3))
        half=float(max((hi-lo)*0.025,abs(prediction-last)*0.11+0.01))
        if state=='UNKNOWN_DATA_QUALITY': half*=1.8
        lower=prediction-half; upper=prediction+half
        cross=bool(upper>=hi)
        prob=float(np.clip((upper-hi)/(half+1e-6)*0.5+0.15 if cross else 0.02,0,1))
        if state=='LATENT_ACCELERATING' and part_id=='MMIC_ANALOG_002': cross=True; prob=0.74; prediction=68.8; lower=65.9; upper=71.7
        if state=='HARD_EARLY_FAILURE': cross=True; prob=0.98

        for t,v in zip(times,vals):
            records.append({
                'part_id':part_id,'lot_id':lot,'wafer_id':f'WAF-{lot[-2:]}-{part_counter:02d}','test_run_id':test_run,
                'component_family':family,'component_type':component_type,'parameter':parameter,'semantic_type':parameter,
                'physical_quantity':param_meta[parameter][0],'unit':unit,'profile_id':profile,'test_stage':'BURN_IN','test_method':'POWER_BURN_IN',
                'stress_mode':'TEMPERATURE','temperature_C':125.0+rng.normal(0,0.8),'burnin_temperature_C':125.0,
                'voltage_V':5.0 if 'current' in param_meta[parameter][0] else 3.3,'current_A':0.18,
                'time_h':float(t),'value':None if not np.isfinite(v) else round(float(v),6),
                'measurement_status':'MISSING' if not np.isfinite(v) else 'VALID','censoring':'NONE' if np.isfinite(v) else 'NONE',
                'absolute_limit_lower':limit_lower,'absolute_limit_upper':limit_upper,
                'engineering_limit_lower':limit_lower,'engineering_limit_upper':limit_upper,
                'defect_state':state,'latent_defect_label':int(state in {'LATENT_ACCELERATING','LATENT_CHANGE_POINT','HARD_EARLY_FAILURE'}),
                'high_but_safe':int(state=='HEALTHY_STABLE' and risk>0.35),'measurement_only_anomaly':int(state=='UNKNOWN_DATA_QUALITY'),
                'precursor_strength':round(float(temporal_score),4),'precursor_effect':'latent_drift' if state.startswith('LATENT') else 'none',
                'failure_mode':state,'primary_failure_mode':state,
            })

        trace={
            'module_a':{'parameter':parameter,'population':round(pop_score,3),'temporal':round(temporal_score,3),'multivariate':round(multivariate,3),'absolute_limits':round(abs_score,3),'component_family':family,'component_type':component_type,'lot_id':lot},
            'module_b':{'parameter':parameter,'unit':unit,'forecast_origin_h':24,'value_0h':v0,'value_24h':v24,'value_asof':v24,'prediction_168h':prediction,'failure_mode':state,'target_horizon_h':168,'prediction_lower':lower,'prediction_upper':upper,'predicted_limit_exceedance':cross},
            'ood':{'status':ood_status,'score':round(ood,3)},
            'data_quality':{'status':'PASS' if state!='UNKNOWN_DATA_QUALITY' else 'DEGRADED','reasons':[] if state!='UNKNOWN_DATA_QUALITY' else ['Late readpoints missing; confidence reduced.']},
            'policy':{'decision':decision,'reason':'Safety policy evaluates independent evidence channels, future trajectory and domain trust before disposition.'},
            'fusion':{'supporting_evidence_count':int(sum(s>=0.5 for s in [pop_score,temporal_score,multivariate,abs_score]))}
        }
        screening_rows.append({
            'part_id':part_id,'lot_id':lot,'component_family':family,'component_type':component_type,'parameter':parameter,'unit':unit,
            'decision':decision,'risk_score':round(risk,3),'confidence':confidence,'ood_status':ood_status,'ood_score':round(ood,3),
            'anomaly_risk':round(anomaly_risk,3),'failure_risk':round(failure_risk,3),'uncertainty_score':round(uncertainty,3),
            'failure_mode':state,'value_0h':v0,'value_24h':v24,'hard_limit_violation':state=='HARD_EARLY_FAILURE',
            'near_limit':bool(upper>=hi*0.94),'supporting_evidence_count':trace['fusion']['supporting_evidence_count'],'trace':json.dumps(trace)
        })
        feature_rows.append({'part_id':part_id,'lot_id':lot,'parameter':parameter,'unit':unit,'component_family':family,'component_type':component_type,
            'value_0h':v0,'value_24h':v24,'population_robust_z':round(robust_z,3),'slope_0_24':round(slope,6),'slope_24_168':round(slope_late,6),
            'slope_acceleration':round(slope_late-slope,6),'lot_median':round(median,6),'lot_mad':round(mad,6),'latest_value':round(last,6),
            'latest_percent_from_median':round((last-median)/max(abs(median),1e-9)*100,3),'volatility':round(float(np.nanstd(vals)),6),
            'available_readpoints':int(np.isfinite(vals).sum()),'missing_fraction':round(float(np.isnan(vals).mean()),3),'absolute_limit_upper':limit_upper,'absolute_limit_lower':limit_lower,
            'engineering_limit_upper':hi,'engineering_limit_lower':lo,'temperature_C':125.0,'voltage_V':5.0 if 'current' in param_meta[parameter][0] else 3.3,
            'stress_duration_h':168.0})
        anomaly_rows.append({'part_id':part_id,'parameter':parameter,'population':round(pop_score,4),'temporal':round(temporal_score,4),'multivariate':round(multivariate,4),'absolute_limits':round(abs_score,4),'isolation_forest':round(min(1,0.65*anomaly_risk+0.15),4),'ensemble_risk':round(anomaly_risk,4),'calibrated_risk':round(risk,4),'evidence_channels':int(trace['fusion']['supporting_evidence_count'])})
        forecast_rows.append({'part_id':part_id,'parameter':parameter,'unit':unit,'forecast_origin_h':24.0,'target_horizon_h':168.0,'selected_forecast_model':'gradient_boosting' if decision!='SAFE' else 'persistence','prediction_168h':round(prediction,5),'prediction_lower':round(lower,5),'prediction_upper':round(upper,5),'prediction_interval_width':round(upper-lower,5),'conformal_half_width':round(half,5),'predicted_limit_exceedance':int(cross),'limit_exceedance_probability_proxy':round(prob,4),'safety_slope_excess':round(max(0,slope_late-(hi-lo)/168.0),6),'prediction_persistence':round(persistence,5),'prediction_linear':round(linear,5),'prediction_ridge':round(ridge,5),'prediction_gradient_boosting':round(gb,5)})
        facts=[f'Component family: {family}.',f'Parameter: {parameter}.',f'Physical quantity: {param_meta[parameter][0]}.',f'Unit: {unit}.']
        if v0 is not None: facts.append(f'Value at 0 h: {v0:.4g} {unit}.')
        if v24 is not None: facts.append(f'Value at 24 h: {v24:.4g} {unit}.')
        findings=[f'Population evidence score: {pop_score:.2f}.',f'Temporal drift evidence score: {temporal_score:.2f}.',f'Multivariate evidence score: {multivariate:.2f}.',f'168 h forecast: {prediction:.4g} {unit} with conformal upper bound {upper:.4g} {unit}.']
        if cross: findings.append('Upper forecast bound approaches or exceeds the engineering boundary.')
        if state=='UNKNOWN_DATA_QUALITY': findings.append('Data-quality degradation reduces confidence in automatic disposition.')
        policy=[f'Safety policy dispositioned {decision}: independent evidence, forecast uncertainty and domain trust are considered together.']
        cf=['If the measured trajectory returned toward the lot reference band, temporal and population warnings would weaken.' if decision!='SAFE' else 'No single evidence change dominates this safe decision.']
        explanation_rows.append({'part_id':part_id,'decision':decision,'risk_score':round(risk,3),'confidence':confidence,'parameter':parameter,
            'summary':f'{decision}: {parameter} evaluated across peer population, temporal behaviour, future trajectory and domain trust.',
            'facts':json.dumps(facts),'model_findings':json.dumps(findings),'policy_reasoning':json.dumps(policy),'counterfactuals':json.dumps(cf),
            'specific_counterfactual':cf[0],'lead_time_h':132.0 if decision in {'REVIEW','REJECT'} and state!='UNKNOWN_DATA_QUALITY' else None,
            'lead_time': '132.0 h early-warning lead time' if decision in {'REVIEW','REJECT'} else 'Not applicable',
            'recommended_next_test':'Continue burn-in / engineering review' if decision=='REVIEW' else ('Hold for engineering disposition' if decision=='REJECT' else 'Continue standard screening burn-in protocol.' if decision=='SAFE' else 'Resolve missing observations before automatic disposition.'),
            'pattern_attribution_json':json.dumps({'ranked':[{'pattern':'Temporal drift','score':temporal_score},{'pattern':'Population shift','score':pop_score},{'pattern':'Multivariate novelty','score':multivariate}]}),
            'audit_trace_json':json.dumps(trace),'why_this_decision':policy[0]})
        state_for_part[part_id]=state
        part_counter+=1

raw=pd.DataFrame(records); canonical=raw.copy()
canonical.insert(0,'observation_id',[f'OBS-{i:06d}' for i in range(1,len(canonical)+1)])
canonical['source_file']='demo_component_screening.csv'; canonical['source_row_number']=np.arange(1,len(canonical)+1); canonical['source_column']='value'; canonical['source_format']='long'; canonical['raw_parameter']=canonical['parameter']; canonical['value_original']=canonical['value']; canonical['raw_unit']=canonical['unit']; canonical['unit_conversion_factor']=1.0; canonical['mapping_method']='exact_alias'; canonical['unit_source']='explicit'; canonical['quality_flag']=np.where(canonical['measurement_status'].eq('VALID'),'OK','LIMITED'); canonical['quality_detail']=np.where(canonical['measurement_status'].eq('VALID'),'','missing_readpoint')
features=pd.DataFrame(feature_rows); anomaly=pd.DataFrame(anomaly_rows); forecast=pd.DataFrame(forecast_rows); screening=pd.DataFrame(screening_rows); explanations=pd.DataFrame(explanation_rows)

raw.to_csv(OUT/'input.csv',index=False); canonical.to_csv(OUT/'canonical.csv',index=False); features.to_csv(OUT/'features.csv',index=False); anomaly.to_csv(OUT/'anomaly.csv',index=False); forecast.to_csv(OUT/'forecast.csv',index=False); screening.to_csv(OUT/'screening.csv',index=False); explanations.to_csv(OUT/'explanations.csv',index=False)

# Manifest and active-source metadata.
manifest={'run_id':'demo_scenario_2026','source':'representative_synthetic_demo_dataset','rows':{'raw':int(len(raw)),'canonical':int(len(canonical)),'features':int(len(features)),'anomaly':int(len(anomaly)),'forecast':int(len(forecast)),'screening':int(len(screening))},'parts':int(screening.part_id.nunique()),'parameters':int(screening.parameter.nunique()),'as_of_h':24.0,'target_horizon_h':168.0,'schema_strategy':'schema-adaptive long-form canonicalization','demo_only':True,'claim_boundary':'Representative synthetic demonstration only; not ISRO operational data or validation.'}
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

# Benchmark artifacts: presentation-oriented, explicitly synthetic.
bench=OUT/'benchmark_kaggle'
pd.DataFrame([['recall',0.923,0.031],['precision',0.214,0.018],['fpr',0.087,0.009],['escape_recall',1.0,0.0]],columns=['metric','mean','std']).to_csv(bench/'system_metrics_mean_std.csv',index=False)
pd.DataFrame([['median_lead_time_h',132.0,7.2],['lead_detection_rate',0.923,0.031]],columns=['metric','mean','std']).to_csv(bench/'lead_time_mean_std.csv',index=False)
pd.DataFrame([['escape_recall',1.0,0.0],['future_defective_detected',12.0,0.0],['future_defective_total',13.0,0.0]],columns=['metric','mean','std']).to_csv(bench/'latent_escape_mean_std.csv',index=False)
pd.DataFrame([['HARD_EARLY_FAILURE',4,4,1.0,0.0],['LATENT_ACCELERATING',9,6,1.0,0.05],['LATENT_CHANGE_POINT',6,4,0.75,0.10],['HEALTHY_STABLE',38,0,np.nan,0.08],['UNKNOWN_DATA_QUALITY',3,0,np.nan,0.0]],columns=['mechanism','parts','future_defective','recall','fpr']).to_csv(bench/'mechanism_metrics_mean.csv',index=False)
pd.DataFrame([['persistence',0.242,0.410,0.94],['gradient_boosting',0.251,0.432,0.95],['ridge',0.329,0.618,0.93],['linear',0.881,1.402,0.89]],columns=['model','mae','rmse','coverage']).to_csv(bench/'forecast_metrics_mean.csv',index=False)
pd.DataFrame([['missing_20pct',0.90,0.19,'synthetic demo'],['missing_40pct',0.82,0.21,'synthetic demo'],['measurement_artifact',0.88,0.16,'synthetic demo'],['normal_aging',np.nan,0.09,'synthetic demo'] ],columns=['scenario','recall','flag_rate','notes']).to_csv(bench/'robustness_metrics.csv',index=False)

origins=[12,24,48,72,96,120,144,168]
prog_rows=[]
for seed in (42,43,44):
    sdir=bench/f'seed_{seed}'; sdir.mkdir()
    rows=[]
    for origin in origins:
        od=sdir/f'origin_{origin}h'; od.mkdir()
        sr=[]
        for _,r in screening.iterrows():
            state=r['failure_mode']; dec='SAFE'
            if state=='HARD_EARLY_FAILURE': dec='REJECT' if origin>=12 else 'REVIEW'
            elif state=='LATENT_ACCELERATING': dec='REVIEW' if origin>=24 else 'SAFE'
            elif state=='LATENT_CHANGE_POINT': dec='REVIEW' if origin>=72 else 'SAFE'
            elif state=='UNKNOWN_DATA_QUALITY': dec='UNKNOWN' if origin>=48 else 'REVIEW'
            elif state=='HEALTHY_STABLE': dec='SAFE'
            risk=float(r['risk_score']) if dec!='SAFE' else float(r['risk_score'])*0.55
            sr.append({'part_id':r['part_id'],'decision':dec,'risk_score':round(risk,4)})
        pd.DataFrame(sr).to_csv(od/'screening.csv',index=False)
    metrics=[]
    # Synthetic validation trajectory: progressively increasing recall while controlling FPR.
    metric_values={12:(0.70,0.03,0.18),24:(0.86,0.06,0.22),48:(0.92,0.08,0.24),72:(0.94,0.087,0.25),96:(0.95,0.09,0.25),120:(0.96,0.091,0.26),144:(0.97,0.092,0.26),168:(0.97,0.092,0.26)}
    for origin in origins:
        rec,fpr,flag=metric_values[origin]
        metrics.append({'origin_h':origin,'recall':rec,'fpr':fpr,'escape_recall':min(1.0,rec+0.03),'flag_rate':flag})
    pd.DataFrame(metrics).to_csv(sdir/'progressive_metrics.csv',index=False)

pd.DataFrame([{'origin_h':o,'recall':metric_values[o][0],'fpr':metric_values[o][1],'escape_recall':min(1.0,metric_values[o][0]+0.03),'flag_rate':metric_values[o][2]} for o in origins]).to_csv(bench/'progressive_metrics_mean.csv',index=False)

(OUT/'upload_metadata.json').write_text(json.dumps({'run_id':'demo_scenario_2026','original_filename':'demo_component_screening.csv','created_utc':'2026-09-06T00:00:00Z','mode':'demonstration','synthetic':True},indent=2),encoding='utf-8')
(OUT/'DEMO_REPORT_README.md').write_text('''# ACS Demonstration Report\n\nRepresentative synthetic component screening data for UI demonstration.\n\nThis dataset is intentionally rich: identifiers, lot/wafer/test context, semantic parameter metadata, units, burn-in observations, engineering limits, data quality, anomaly evidence, forecast uncertainty, policy decisions, explanations, and progressive benchmark artifacts are all included.\n\n**Not ISRO data. Not operational validation.** Replace this report with project-approved screening artifacts when entering the final stage.\n''',encoding='utf-8')
print('DEMO REPORT GENERATED', OUT)
print('parts',screening.part_id.nunique(),'rows',len(raw),'safe/review/reject/unknown',screening.decision.value_counts().to_dict())
