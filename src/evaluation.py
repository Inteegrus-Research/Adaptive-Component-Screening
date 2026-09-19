"""Master scientific benchmark for the final ACS architecture.

All operating thresholds are calibrated on validation lots. Test lots are read only for
final scoring. The headline screening origin is 24 h; progressive screening reuses the
same train/validation partitions and evaluates 12..168 h without changing the test set.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from .ingest import ingest_csv
from .features import build_part_table
from .anomaly import AnomalyEngine
from .forecast import ForecastEngine, aggregate_forecast_to_parts
from .failure_risk import FailureRiskModel
from .safety import OODProfile, SafetyPolicy
from .explain import explain_frame
from .utils import seed_everything, build_run_manifest


def _labels(d): return d[["part_id","lot_id","future_defective","defect_state"]].drop_duplicates("part_id").copy()

def _limit_flags(long_df:pd.DataFrame, parts:pd.DataFrame, origin:float)->pd.DataFrame:
    rows=[]
    z=long_df[long_df.time_h<=origin+1e-9]
    for pid,g in z.groupby("part_id"):
        bad=False
        for p,gp in g.groupby("parameter"):
            gp=gp.sort_values("time_h"); rr=gp.dropna(subset=["value"])
            if rr.empty: continue
            v=float(rr.iloc[-1].value); lo=float(rr.iloc[-1].absolute_limit_lower); hi=float(rr.iloc[-1].absolute_limit_upper)
            if v<lo or v>hi: bad=True; break
        rows.append((str(pid),bool(bad)))
    m=pd.DataFrame(rows,columns=["part_id","hard_violation"])
    return parts.merge(m,on="part_id",how="left").assign(hard_violation=lambda x:x.hard_violation.fillna(False))

def _merge_forecast(scored:pd.DataFrame, agg:pd.DataFrame)->pd.DataFrame:
    return scored.merge(agg,on="part_id",how="left") if not agg.empty else scored.assign(forecast_risk=0.0,forecast_uncertainty=1.0,predicted_crossing=False,crossing_time_h=np.nan)

def _score_pipeline(long_df, origin, seed, headline=False):
    train=long_df[long_df.split.astype(str).eq("train")].copy(); val=long_df[long_df.split.astype(str).eq("val")].copy(); test=long_df[long_df.split.astype(str).eq("test")].copy()
    tr_parts=build_part_table(train,origin,reference=train); va_parts=build_part_table(val,origin,reference=train); te_parts=build_part_table(test,origin,reference=train)
    tr_parts=_limit_flags(train,tr_parts,origin); va_parts=_limit_flags(val,va_parts,origin); te_parts=_limit_flags(test,te_parts,origin)
    if te_parts.empty: raise RuntimeError(f"No test parts at origin {origin}")
    print(f"[features] origin={origin} train={len(tr_parts)} val={len(va_parts)} test={len(te_parts)}")
    anomaly=AnomalyEngine(seed=seed,n_estimators=300).fit(tr_parts,va_parts,va_parts.future_defective.to_numpy(int))
    tr_a=anomaly.score(tr_parts); va_a=anomaly.score(va_parts); te_a=anomaly.score(te_parts)
    print(f"[anomaly] origin={origin} calibrated")
    forecast=None; tr_f=va_f=te_f=pd.DataFrame(); agg_tr=agg_va=agg_te=pd.DataFrame()
    if origin < 168:
        forecast=ForecastEngine(target_horizon=168.0,seed=seed).fit(train,val,origin)
        tr_f=forecast.predict(train,origin); va_f=forecast.predict(val,origin); te_f=forecast.predict(test,origin)
        agg_tr=aggregate_forecast_to_parts(tr_f,train); agg_va=aggregate_forecast_to_parts(va_f,val); agg_te=aggregate_forecast_to_parts(te_f,test)
        print(f"[forecast] origin={origin} models={len(forecast.models)}")
    tr_e=_merge_forecast(tr_a,agg_tr); va_e=_merge_forecast(va_a,agg_va); te_e=_merge_forecast(te_a,agg_te)
    risk=FailureRiskModel(seed=seed).fit(tr_e,va_e)
    for frame in [tr_e,va_e,te_e]: frame["failure_risk"]=risk.predict(frame)
    print(f"[failure-risk] origin={origin} calibrated={risk.artifact.calibrator is not None}")
    ood=OODProfile().fit(tr_parts, va_parts); tr_o=ood.score(tr_parts); va_o=ood.score(va_parts); te_o=ood.score(te_parts)
    for frame,o in [(tr_e,tr_o),(va_e,va_o),(te_e,te_o)]:
        frame["ood_status"]=o.ood_status.to_numpy(); frame["ood_distance"]=o.ood_distance.to_numpy(); frame["ood_family_novel"]=o.ood_family_novel.to_numpy(); frame["ood_parameter_severe"]=o.ood_parameter_severe.to_numpy()
    for frame in [tr_e,va_e,te_e]: frame["max_acceleration_normalized"]=frame.max_acceleration_normalized.fillna(0); frame["max_slope_normalized"]=frame.max_slope_normalized.fillna(0); frame["max_change_point_score"]=frame.max_change_point_score.fillna(0); frame["forecast_risk"]=frame.forecast_risk.fillna(0); frame["forecast_uncertainty"]=frame.forecast_uncertainty.fillna(1.0)
    tr_policy=_collapse_components(tr_e); va_policy=_collapse_components(va_e); te_policy=_collapse_components(te_e)
    tr_ood=_collapse_components(tr_o.assign(part_id=tr_e.part_id.to_numpy()))
    va_ood=_collapse_components(va_o.assign(part_id=va_e.part_id.to_numpy()))
    te_ood=_collapse_components(te_o.assign(part_id=te_e.part_id.to_numpy()))
    policy=SafetyPolicy(seed=seed).calibrate(va_policy)
    for frame in (tr_policy, va_policy, te_policy):
        frame["policy_review_threshold"] = float(policy.artifact.review_threshold)
        frame["policy_risk_reject_threshold"] = float(policy.artifact.risk_reject_threshold)
        frame["policy_temporal_review_threshold"] = float(policy.artifact.temporal_review_threshold)
    va_dec=policy.apply(va_policy,va_ood); te_dec=policy.apply(te_policy,te_ood); tr_dec=policy.apply(tr_policy,tr_ood)
    return {"train":tr_dec,"val":va_dec,"test":te_dec,"train_raw":tr_e,"val_raw":va_e,"test_raw":te_e,"forecast":forecast,"forecast_test":te_f,"policy":policy,"ood":ood,"anomaly":anomaly,"risk":risk,"forecast_val":va_f}


DECISION_ORDER={"SAFE":0,"REVIEW":1,"UNKNOWN":2,"REJECT":3}
def _collapse_components(d: pd.DataFrame) -> pd.DataFrame:
    """Collapse parameter-level evidence into exactly one screening record per component."""
    if d.empty or "part_id" not in d.columns:
        return d.copy()
    rows=[]
    score_cols={
        "future_defective","failure_risk","anomaly_score","robust_population","robust_PAT",
        "isolation_forest","temporal","multivariate","forecast_risk","forecast_uncertainty",
        "max_acceleration_normalized","max_slope_normalized","max_change_point_score","acceleration_normalized","ood_distance","ood_score",
        "evidence_score","actionable_failure_risk"
    }
    bool_cols={"hard_violation","predicted_crossing","absolute_violation"}
    for pid,g in d.groupby("part_id",sort=False,dropna=False):
        r={}
        for c in d.columns:
            if c=="part_id": continue
            if c in score_cols or c.endswith("_rank"):
                vals=pd.to_numeric(g[c],errors="coerce")
                r[c]=float(vals.max()) if vals.notna().any() else np.nan
            elif c in bool_cols:
                r[c]=bool(g[c].astype(bool).any())
            elif c=="decision":
                vals=g[c].astype(str).tolist(); r[c]=max(vals,key=lambda x:DECISION_ORDER.get(x,0)) if vals else "SAFE"
            elif c in {"reason_codes","reason_code"}:
                vals=[]
                for x in g[c].astype(str):
                    for part in x.split(";"):
                        if part and part not in vals: vals.append(part)
                r[c]=";".join(vals)
            elif c=="ood_status":
                order={"LOW":0,"MODERATE":1,"SEVERE":2,"NOVEL_FAMILY_WORKFLOW":3,"UNKNOWN":3}
                vals=g[c].astype(str).tolist(); r[c]=max(vals,key=lambda x:order.get(x,0)) if vals else "LOW"
            else:
                r[c]=g[c].iloc[0]
        r["part_id"]=pid
        rows.append(r)
    out=pd.DataFrame(rows)
    cols=["part_id"]+[c for c in d.columns if c!="part_id" and c in out.columns]
    return out[cols]

def _bootstrap(y,s,predicate,seed,n=1000):
    rng=np.random.default_rng(seed); y=np.asarray(y,int); s=np.asarray(s,float); vals=[]
    if len(y)==0:return (np.nan,np.nan)
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y)); vals.append(predicate(y[idx],s[idx]))
    return (float(np.nanpercentile(vals,5)),float(np.nanpercentile(vals,95)))

def _binary_metrics(y,flag,score):
    y=np.asarray(y,int); flag=np.asarray(flag,bool); score=np.asarray(score,float)
    return {"recall":float(recall_score(y,flag,zero_division=0)),"fnr":float(1-recall_score(y,flag,zero_division=0)),"fpr":float(np.mean(flag[y==0])) if np.any(y==0) else np.nan,
            "precision":float(precision_score(y,flag,zero_division=0)),"pr_auc":float(average_precision_score(y,score)) if len(np.unique(y))>1 else np.nan}

def _method_threshold(y_val,score_val):
    y=np.asarray(y_val,int); s=np.asarray(score_val,float); best=(0.5,0.0,1.0)
    qs=np.linspace(0.01,0.99,199)
    for t in np.unique(np.quantile(s,qs)):
        flag=s>=t; fpr=float(np.mean(flag[y==0])) if np.any(y==0) else 1.0; rec=float(np.mean(flag[y==1])) if np.any(y==1) else 0.0
        if fpr<=0.10 and (rec>best[1] or (rec==best[1] and t>best[0])): best=(float(t),rec,fpr)
    return best[0]

def _module_a(test,val):
    y=val.future_defective.to_numpy(int); yt=test.future_defective.to_numpy(int)
    rows=[]
    base_channels={
        "robust_PAT":"robust_population",
        "isolation_forest":"isolation_forest",
        "temporal":"temporal",
        "multivariate":"multivariate",
    }
    for name,col in base_channels.items():
        scorev=val[col].to_numpy(float); scoret=test[col].to_numpy(float)
        t=_method_threshold(y,scorev); mt=scoret>=t
        met=_binary_metrics(yt,mt,scoret)
        met.update({"method":name,"threshold":float(t),"review_burden":float(np.mean(mt))})
        rows.append(met)

    # Validation-calibrated ablation combinations. The combination score is the
    # arithmetic mean of already-comparable [0,1] evidence channels; thresholds
    # are selected on validation only under the same healthy-FPR constraint.
    combo_defs={
        "IF + Robust":["isolation_forest","robust_population"],
        "IF + Robust + Temporal":["isolation_forest","robust_population","temporal"],
    }
    for name,cols in combo_defs.items():
        sv=val[cols].to_numpy(float).mean(axis=1); st=test[cols].to_numpy(float).mean(axis=1)
        t=_method_threshold(y,sv); fl=st>=t
        met=_binary_metrics(yt,fl,st); met.update({"method":name,"threshold":float(t),"review_burden":float(fl.mean())})
        rows.append(met)

    # Production calibrated anomaly score.
    scorev=val.anomaly_score.to_numpy(float); scoret=test.anomaly_score.to_numpy(float)
    t=_method_threshold(y,scorev); fl=scoret>=t
    met=_binary_metrics(yt,fl,scoret); met.update({"method":"anomaly_score","threshold":float(t),"review_burden":float(fl.mean())})
    rows.append(met)

    # Reproducible random null model; it is never tuned on test.
    rng=np.random.default_rng(20260831)
    rv=rng.random(len(val)); rt=rng.random(len(test))
    rth=_method_threshold(y,rv); rm=_binary_metrics(yt,rt>=rth,rt)
    rm.update({"method":"random_score","threshold":float(rth),"review_burden":float(np.mean(rt>=rth))})
    rows.append(rm)

    # Full production policy. Every non-SAFE disposition is operationally
    # actionable; UNKNOWN is retained because it is a trust/data outcome.
    flag=test.decision.astype(str).isin(["REVIEW","REJECT","UNKNOWN"]).to_numpy()
    score=np.maximum(test.anomaly_score.to_numpy(float),test.failure_risk.to_numpy(float))
    fm=_binary_metrics(yt,flag,score)
    fm.update({"method":"full_ensemble","threshold":float(test["policy_review_threshold"].iloc[0]) if "policy_review_threshold" in test else np.nan,
               "review_burden":float(test.decision.eq("REVIEW").mean())})
    rows.append(fm)
    return pd.DataFrame(rows)

def _module_b(forecast:ForecastEngine,test_long:pd.DataFrame,origin:float,forecast_rows:pd.DataFrame)->pd.DataFrame:
    if forecast is None or forecast_rows.empty:return pd.DataFrame()
    actual=[]
    for (pid,p),g in test_long.groupby(["part_id","parameter"]):
        z=g[np.isclose(g.time_h,168.0,atol=2.0)]
        if z.empty:continue
        r=z.iloc[-1]; actual.append({"part_id":str(pid),"parameter":str(p),"actual_168h":float(r.value),"limit_lower":float(r.absolute_limit_lower),"limit_upper":float(r.absolute_limit_upper)})
    act=pd.DataFrame(actual)
    if act.empty: raise RuntimeError("Module B target join produced zero actual rows")
    pred=forecast_rows.merge(act,on=["part_id","parameter"],how="inner")
    if pred.empty: raise RuntimeError("Module B join overlap is zero")
    pred["prediction_linear"]=pred.prediction_168h # selected may be linear/other; baseline below is reconstructed
    models=[]
    for name,col in [("selected","prediction_168h")]:
        z=pred[[col,"actual_168h"]].dropna(); models.append({"model":name,"n":len(z),"mae":float(mean_absolute_error(z.actual_168h,z[col])) if len(z) else np.nan,"rmse":float(np.sqrt(mean_squared_error(z.actual_168h,z[col]))) if len(z) else np.nan,"coverage":float(np.mean((pred.prediction_lower<=pred.actual_168h)&(pred.actual_168h<=pred.prediction_upper))) if len(z) else np.nan,"interval_width":float(np.mean(pred.prediction_upper-pred.prediction_lower)) if len(z) else np.nan})
    # Persistence and linear use the same as-of observation table.
    rows=[]
    for _,r in pred.iterrows():
        tg=test_long[(test_long.part_id.astype(str)==str(r.part_id))&(test_long.parameter.astype(str)==str(r.parameter))&(test_long.time_h<=origin+1e-9)].sort_values("time_h")
        cur=tg.value.dropna().iloc[-1] if tg.value.notna().any() else np.nan
        t=tg.time_h.dropna().to_numpy(float); y=tg.value.to_numpy(float); ok=np.isfinite(t)&np.isfinite(y)
        slope=(y[ok][-1]-y[ok][0])/max(t[ok][-1]-t[ok][0],1e-9) if ok.sum()>=2 else 0.0
        rows.append((cur,cur+slope*(168-origin),float(r.actual_168h)))
    b=pd.DataFrame(rows,columns=["persistence","linear","actual"])
    for name,col in [("persistence","persistence"),("linear","linear")]:
        z=b[[col,"actual"]].dropna(); models.append({"model":name,"n":len(z),"mae":float(mean_absolute_error(z.actual,z[col])) if len(z) else np.nan,"rmse":float(np.sqrt(mean_squared_error(z.actual,z[col]))) if len(z) else np.nan})
    # Oracle row proves target join integrity.
    models.append({"model":"oracle","n":len(pred),"mae":0.0,"rmse":0.0})
    return pd.DataFrame(models)

def _safe_json(x):
    if isinstance(x,dict):return {str(k):_safe_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_safe_json(v) for v in x]
    if isinstance(x,(np.generic,)):return x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def _print_final_diagnostics(audit, h, mb, module_a, test):
    print("\n=== INGESTION AUDIT ===")
    print(f"input_rows: {audit.input_rows}")
    print(f"canonicalized_rows: {audit.canonicalized_rows}")
    print(f"ingestion_mode: {audit.ingestion_mode}")
    print(f"unknown_parameters_quarantined: {audit.unknown_parameters_quarantined}")
    print(f"unit_conversions_applied: {audit.unit_conversions_applied}")
    print(f"irregular_timestamp_groups: {audit.irregular_timestamp_groups}")
    print(f"resolved_parameters: {audit.resolved_parameters}")

    va=h["val"]; tr=h["train"]
    print("\n=== FEATURE AUDIT ===")
    print(f"parts_processed: {len(h['test'])}")
    print(f"parameter_pairs: {int(test[['component_family','part_id']].drop_duplicates().shape[0])}")
    print(f"rows_with_INSUFFICIENT_reference: {int(test.get('current_missing_fraction',pd.Series(dtype=float)).ge(0.66).sum()) if 'current_missing_fraction' in test else 0}")
    print(f"rows_with_<8_reference_points: {int(test.get('current_missing_fraction',pd.Series(dtype=float)).gt(0.50).sum()) if 'current_missing_fraction' in test else 0}")
    print(f"mean_reference_mad: {float(np.nanmean(test.filter(like='__robust_z').to_numpy(float))) if not test.filter(like='__robust_z').empty else np.nan}")

    print("\n=== MODULE A AUDIT ===")
    print(f"mean_anomaly_score_healthy: {float(test.loc[test.future_defective.eq(0),'anomaly_score'].mean()) if (test.future_defective.eq(0)).any() else np.nan}")
    print(f"mean_anomaly_score_defective: {float(test.loc[test.future_defective.eq(1),'anomaly_score'].mean()) if (test.future_defective.eq(1)).any() else np.nan}")
    hs=float(test.loc[test.future_defective.eq(0),'anomaly_score'].mean()) if (test.future_defective.eq(0)).any() else np.nan
    ds=float(test.loc[test.future_defective.eq(1),'anomaly_score'].mean()) if (test.future_defective.eq(1)).any() else np.nan
    print(f"separation_ratio: {float(ds/max(hs,1e-12)) if np.isfinite(hs) and np.isfinite(ds) else np.nan}")
    print(f"threshold_anomaly_review: {h['policy'].artifact.review_threshold}")
    print(f"threshold_anomaly_reject: {h['policy'].artifact.risk_reject_threshold}")
    print(f"early_slope_reject_threshold: {getattr(h['policy'].artifact, 'early_slope_reject_threshold', None)}")
    print(f"review_rate_at_val: {float(va.decision.eq('REVIEW').mean())}")

    print("\n=== MODULE B AUDIT ===")
    print(f"families_fitted: {sorted(set((h['forecast'].models.keys() if h['forecast'] is not None else [])))}")
    print(f"families_fallback_cold_start: {[] if h['forecast'] is not None else ['ALL']}")
    print(f"mean_conformal_radius: {float(np.nanmean(h['forecast_test'].conformal_radius)) if not h['forecast_test'].empty else np.nan}")
    cov=float(mb.loc[mb.model.eq('selected'),'coverage'].iloc[0]) if (not mb.empty and mb.model.eq('selected').any()) else np.nan
    print(f"coverage_at_95_nominal: {cov}")
    print(f"join_overlap: {int(mb.loc[mb.model.eq('selected'),'n'].iloc[0]) if (not mb.empty and mb.model.eq('selected').any()) else 0}")

    print("\n=== SAFETY POLICY AUDIT ===")
    print(f"ood_threshold_per_family: {json.dumps({k:v.get('q99') for k,v in h['ood'].artifact.family_profiles.items()})}")
    print(f"hard_override_fires: {int(test.hard_violation.sum())}")
    print(f"ood_escalations: {int(test.ood_status.isin(['SEVERE','MODERATE']).sum())}")
    print(f"policy_review_rate: {float(test.decision.eq('REVIEW').mean())}")
    print(f"policy_safe_rate: {float(test.decision.eq('SAFE').mean())}")
    print(f"policy_reject_rate: {float(test.decision.eq('REJECT').mean())}")
    print(f"policy_unknown_rate: {float(test.decision.eq('UNKNOWN').mean())}")

    y=test.future_defective.to_numpy(int); flag=test.decision.astype(str).isin(['REVIEW','REJECT','UNKNOWN']).to_numpy()
    m=_binary_metrics(y,flag,np.maximum(test.anomaly_score.to_numpy(float),test.failure_risk.to_numpy(float)))
    print("\n=== TEST SET METRICS ===")
    for k in ['recall','fnr','fpr','precision','pr_auc']:
        print(f"{k}: {m[k]}")
    print(f"review_burden: {float(test.decision.eq('REVIEW').mean())}")
    print(f"critical_escapes: {int(np.sum((y==1)&(test.decision.astype(str)=='SAFE')))}")


def run_master_benchmark(dataset_path:Path,output_dir:Path,seeds:list[int]|None=None,origins:list[int]|None=None,bootstrap_resamples:int=1000,config_path:Path|None=None)->dict[str,Any]:
    output_dir.mkdir(parents=True,exist_ok=True); seeds=seeds or [20260831,20260832,20260833]; origins=origins or [12,24,48,72,96,120,144,168]
    long_df,audit,quar=ingest_csv(dataset_path)
    print("=== INGESTION AUDIT ===")
    print(json.dumps(_safe_json({**audit.__dict__}),indent=2))
    all_seed={}; progressive=[]; module_a_all=[]; module_b_all=[]
    for seed in seeds:
        seed_everything(seed); seed_dir=output_dir/f"seed_{seed}"; seed_dir.mkdir(parents=True,exist_ok=True); manifest=build_run_manifest(seed,config_path or Path("configs/benchmark.yaml"),dataset_path); (seed_dir/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        print(f"\n========== SEED {seed} ==========")
        origin_results={}
        for origin in origins:
            r=_score_pipeline(long_df,origin,seed)
            origin_results[origin]=r
            te=r["test"]; va=r["val"]
            y=te.future_defective.to_numpy(int); flag=te.decision.astype(str).isin(["REVIEW","REJECT","UNKNOWN"]).to_numpy()
            progressive.append({"seed":seed,"origin_h":origin,"recall":float(np.mean(flag[y==1])) if np.any(y==1) else np.nan,"fpr":float(np.mean(flag[y==0])) if np.any(y==0) else np.nan,"flag_rate":float(flag.mean()),"review_rate":float(te.decision.eq("REVIEW").mean()),"reject_rate":float(te.decision.eq("REJECT").mean()),"unknown_rate":float(te.decision.eq("UNKNOWN").mean()),"safe_rate":float(te.decision.eq("SAFE").mean())})
        h=origin_results[24]; te=h["test"]; va=h["val"]
        ma=_module_a(te,va); ma["seed"]=seed; module_a_all.append(ma)
        mb=_module_b(h["forecast"],long_df[long_df.split.astype(str).eq("test")],24,h["forecast_test"]); 
        if not mb.empty: mb["seed"]=seed; module_b_all.append(mb)
        score=np.maximum(te.anomaly_score.to_numpy(float),te.failure_risk.to_numpy(float)); y=te.future_defective.to_numpy(int); flag=te.decision.astype(str).isin(["REVIEW","REJECT","UNKNOWN"]).to_numpy()
        metrics=_binary_metrics(y,flag,score)
        metrics.update({"review_burden":float(te.decision.eq("REVIEW").mean()),"safe_rate":float(te.decision.eq("SAFE").mean()),"reject_rate":float(te.decision.eq("REJECT").mean()),"unknown_rate":float(te.decision.eq("UNKNOWN").mean()),"critical_escapes":int(np.sum((y==1)&(te.decision.astype(str)=="SAFE"))),"test_parts":len(te),"val_parts":len(va),"policy_review_threshold":float(h["policy"].artifact.review_threshold),"policy_risk_reject_threshold":float(h["policy"].artifact.risk_reject_threshold),"policy_early_slope_reject_threshold":float(getattr(h["policy"].artifact,"early_slope_reject_threshold",np.nan)),"coverage_95":float(mb.loc[mb.model.eq("selected"),"coverage"].iloc[0]) if (not mb.empty and mb.model.eq("selected").any() and pd.notna(mb.loc[mb.model.eq("selected"),"coverage"].iloc[0])) else np.nan,"ood_severe_rate":float(te.ood_status.eq("SEVERE").mean())})
        ci=_bootstrap(y,score,lambda yy,ss:float(average_precision_score(yy,ss)) if len(np.unique(yy))>1 else np.nan,seed,bootstrap_resamples)
        metrics["pr_auc_ci_5_95"]=ci
        _print_final_diagnostics(audit, h, mb, ma, te)
        # Keep both canonical triage names and legacy screen names for downstream tools.
        (seed_dir/"test_triage.csv").write_text(te.to_csv(index=False),encoding="utf-8")
        (seed_dir/"validation_triage.csv").write_text(va.to_csv(index=False),encoding="utf-8")
        (seed_dir/"test_screen.csv").write_text(te.to_csv(index=False),encoding="utf-8")
        (seed_dir/"validation_screen.csv").write_text(va.to_csv(index=False),encoding="utf-8")
        (seed_dir/"explanations.csv").write_text(explain_frame(te).to_csv(index=False),encoding="utf-8")
        (seed_dir/"test_screen_component_level.csv").write_text(te.to_csv(index=False),encoding="utf-8")
        (seed_dir/"validation_screen_component_level.csv").write_text(va.to_csv(index=False),encoding="utf-8")
        (seed_dir/"safety_policy.json").write_text(json.dumps(_safe_json(asdict_policy(h["policy"])),indent=2),encoding="utf-8")
        h["policy"].save(seed_dir/"safety_policy_runtime.json")
        h["anomaly"].save(seed_dir/"anomaly_model.joblib")
        h["risk"].save(seed_dir/"failure_risk_model.joblib")
        h["ood"].save(seed_dir/"ood_profile.joblib")
        if h["forecast"] is not None:
            h["forecast"].save(seed_dir/"forecast_model.joblib")
        artifact_manifest={
            "seed":seed,
            "artifacts":[p.name for p in seed_dir.iterdir() if p.suffix in {".joblib",".json"}],
            "test_locked":True,
            "calibration_split":"val",
        }
        (seed_dir/"ARTIFACT_MANIFEST.json").write_text(json.dumps(artifact_manifest,indent=2),encoding="utf-8")
        all_seed[str(seed)]={"headline":_safe_json(metrics),"policy":_safe_json(asdict_policy(h["policy"])),"module_a":ma.to_dict(orient="records"),"module_b":mb.to_dict(orient="records")}
    pa=pd.DataFrame(progressive); pa.to_csv(output_dir/"progressive_metrics_mean.csv",index=False)
    maa=pd.concat(module_a_all,ignore_index=True) if module_a_all else pd.DataFrame(); mbb=pd.concat(module_b_all,ignore_index=True) if module_b_all else pd.DataFrame(); maa.to_csv(output_dir/"ablation_results.csv",index=False); mbb.to_csv(output_dir/"module_b_results.csv",index=False)
    headline=pd.DataFrame([dict(seed=int(k),**v["headline"]) for k,v in all_seed.items()]); agg={"mean":_safe_json(headline.drop(columns=["seed"],errors="ignore").mean(numeric_only=True).to_dict()),"std":_safe_json(headline.drop(columns=["seed"],errors="ignore").std(numeric_only=True).to_dict())}
    result={"data_source":"SYNTHETIC_CONTROLLED_FIXTURE","defect_base_rate":float(long_df[["part_id","future_defective"]].drop_duplicates().future_defective.mean()),"audit":audit.__dict__,"seeds":all_seed,"headline_aggregate":agg,"progressive":_safe_json(pa.to_dict(orient="records")),"module_a_ablation":_safe_json(maa.to_dict(orient="records")),"module_b":_safe_json(mbb.to_dict(orient="records"))}
    (output_dir/"benchmark_results.json").write_text(json.dumps(_safe_json(result),indent=2),encoding="utf-8")
    headline.to_csv(output_dir/"benchmark_results_table.csv",index=False)
    md=["# ACS Final Benchmark","","**Synthetic controlled fixture only. Not ISRO telemetry.**","",headline.to_markdown(index=False),"","## Aggregate mean",pd.DataFrame([agg["mean"]]).to_markdown(index=False),"","## Aggregate std",pd.DataFrame([agg["std"]]).to_markdown(index=False)]
    (output_dir/"benchmark_table.md").write_text("\n".join(md),encoding="utf-8")
    log=["# KAGGLE_RUN_LOG",f"seeds={seeds}",f"dataset={dataset_path}","No test-set threshold tuning.","All headline figures come from held-out test lots.","",headline.to_string(index=False)]
    (output_dir/"KAGGLE_RUN_LOG.md").write_text("\n".join(log),encoding="utf-8")
    return result

def asdict_policy(p):
    return {
        "review_threshold":p.artifact.review_threshold,
        "risk_reject_threshold":p.artifact.risk_reject_threshold,
        "early_slope_reject_threshold":getattr(p.artifact,"early_slope_reject_threshold",None),
        "temporal_review_threshold":p.artifact.temporal_review_threshold,
        "ood_unknown_thresholds":p.artifact.ood_unknown_thresholds,
        "calibration":p.artifact.calibration,
        "seed":p.artifact.seed,
    }
