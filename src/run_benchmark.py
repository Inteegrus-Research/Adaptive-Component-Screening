#!/usr/bin/env python3
"""Final, leakage-safe benchmark campaign.

Runs the operational pipeline on validation/test lots, tunes thresholds only on
validation data, evaluates multiple Module-A methods and Module-B predictors on
untouched test lots, and writes a judge-facing Markdown table.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation import (
    _binary_truth, _metrics, optimize_threshold, escape_matrix,
    robustness_suite, lead_time_from_screening,
)
from src.pipeline import PipelineArtifacts, screen_dataframe
from src.safety import fit_ood_profile
from src.forecast import load_forecast_model, _persist, _linear

SEED=20260831


def _part_labels(d):
    cols=[c for c in ["part_id","lot_id","defect_state","latent_defect_label","absolute_fail_168h","high_but_safe"] if c in d.columns]
    return d[cols].drop_duplicates("part_id").copy()


def _attach_label(score_df, labels):
    keep=[c for c in ["part_id","defect_state","latent_defect_label","absolute_fail_168h","high_but_safe","split"] if c in labels.columns]
    return score_df.merge(labels[keep].drop_duplicates("part_id"),on="part_id",how="left")


def _method_thresholds(val, methods):
    y=_binary_truth(val).to_numpy(int); out={}
    for name,col in methods.items():
        s=pd.to_numeric(val[col],errors="coerce").fillna(0).to_numpy(float)
        if name=="absolute_limits": out[name]=0.5
        else: out[name]=float(optimize_threshold(y,s,fn_cost=100,fp_cost=1,review_cost=.25,max_reject_rate=.20)["threshold"])
    return out


def _module_a_table(test, labels, thresholds):
    d=_attach_label(test,labels); y=_binary_truth(d).to_numpy(int)
    methods={
        "absolute_limits":"absolute_violation",
        "robust_PAT":"population_evidence",
        "isolation_forest":"parameter_isolation_evidence",
        "multivariate_detector":"multivariate_component_score",
        "temporal":"temporal_evidence",
        "full_ensemble":"anomaly_score",
    }
    rows=[]
    absfail=pd.to_numeric(d.get("absolute_fail_168h",0),errors="coerce").fillna(0).astype(int).to_numpy()
    escape=(absfail==0)&(y==1)
    for name,col in methods.items():
        s=pd.to_numeric(d.get(col,0),errors="coerce").fillna(0).to_numpy(float)
        t=float(thresholds[name]); m=_metrics(y,s,t); pred=s>=t
        m.update({"method":name,
                  "latent_escape_recall":float((escape&pred).sum()/max(escape.sum(),1)),
                  "latent_escape_fnr":float((escape&~pred).sum()/max(escape.sum(),1))})
        rows.append(m)
    return pd.DataFrame(rows)


def _module_b_table(test_raw, forecast):
    f=forecast.copy()
    keys=[c for c in ["part_id","parameter"] if c in f.columns]
    if not keys or "value_168h" not in test_raw.columns:
        return pd.DataFrame()
    # Forecast uses canonical parameter semantics. Canonicalize the raw target
    # parameter names before aligning 168 h ground truth, avoiding alias mismatch.
    raw_target=test_raw[[c for c in ["part_id","parameter","value_168h"] if c in test_raw.columns]].copy()
    try:
        from src.ingest import ParameterOntology
        from src.utils import load_yaml, PROJECT_ROOT
        ont=ParameterOntology(load_yaml(PROJECT_ROOT/'configs'/'parameters.yaml'))
        raw_target['parameter']=raw_target['parameter'].map(lambda v: ont.resolve(v)[0].name if ont.resolve(v)[0] is not None else str(v))
    except Exception:
        raw_target['parameter']=raw_target['parameter'].astype(str)
    target=raw_target.drop_duplicates(keys).set_index(keys)['value_168h']
    idx=pd.MultiIndex.from_frame(f[keys]); y=target.reindex(idx).to_numpy(float)
    rows=[]
    upper=pd.to_numeric(f.get("prediction_upper",np.nan),errors="coerce").to_numpy(float)
    lower=pd.to_numeric(f.get("prediction_lower",np.nan),errors="coerce").to_numpy(float)
    limit=pd.to_numeric(f.get("absolute_limit_upper",np.nan),errors="coerce").to_numpy(float) if "absolute_limit_upper" in f else None
    defect_map=_part_labels(test_raw).set_index("part_id")["defect_state"] if "defect_state" in test_raw.columns else None
    defect=f["part_id"].astype(str).map(defect_map).astype(str).str.lower().isin({"latent","hard","defective","failed"}).to_numpy() if defect_map is not None else None
    preds={"persistence":_persist(f),"linear":_linear(f),"selected":pd.to_numeric(f["prediction_168h"],errors="coerce").to_numpy(float)}
    for name,p in preds.items():
        ok=np.isfinite(y)&np.isfinite(p)
        row={"model":name,"n":int(ok.sum()),"mae":float(np.mean(np.abs(y[ok]-p[ok]))) if ok.any() else None,
             "rmse":float(np.sqrt(np.mean((y[ok]-p[ok])**2))) if ok.any() else None}
        if name=="selected":
            cov=ok&np.isfinite(lower)&np.isfinite(upper); row["conformal_coverage"]=float(np.mean((y[cov]>=lower[cov])&(y[cov]<=upper[cov]))) if cov.any() else None
        else: row["conformal_coverage"]=None
        if limit is not None:
            ok2=ok&np.isfinite(limit); actual=y>limit; pred_cross=(upper if name=="selected" else p)>limit
            row["limit_crossing_recall"]=float((actual&pred_cross&ok2).sum()/max((actual&ok2).sum(),1)) if ok2.any() else None
        else: row["limit_crossing_recall"]=None
        row["dangerous_case_mae"]=float(np.mean(np.abs(y[ok&(defect==1)]-p[ok&(defect==1)]))) if defect is not None and np.any(ok&(defect==1)) else None
        rows.append(row)
    return pd.DataFrame(rows)


def run(input_path, output_dir, seed=SEED):
    rng=np.random.default_rng(seed); output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    full=pd.read_csv(input_path)
    required={"train","val","test"}-set(full["split"].astype(str).unique()) if "split" in full else {"train","val","test"}
    if required: raise ValueError(f"Missing split labels: {sorted(required)}")
    train=full[full.split.astype(str).eq("train")].copy(); val=full[full.split.astype(str).eq("val")].copy(); test=full[full.split.astype(str).eq("test")].copy()
    if min(len(train),len(val),len(test))==0: raise ValueError("Train, validation and test partitions must all be non-empty.")
    ood_art=output/"ood_train_only.joblib"; fit_ood_profile(train,ood_art)
    artifacts=PipelineArtifacts(Path("models/anomaly/model.joblib"),Path("models/forecast/model.joblib"),ood_art)
    val_run=screen_dataframe(val,output/"validation",artifacts=artifacts,as_of_h=24,auto_train_missing=False)
    test_run=screen_dataframe(test,output/"test",artifacts=artifacts,as_of_h=24,auto_train_missing=False)
    labels=_part_labels(full)
    val_score=_attach_label(val_run.anomaly,labels); test_score=_attach_label(test_run.anomaly,labels)
    method_cols={"absolute_limits":"absolute_violation","robust_PAT":"population_evidence","isolation_forest":"parameter_isolation_evidence","multivariate_detector":"multivariate_component_score","temporal":"temporal_evidence","full_ensemble":"anomaly_score"}
    thresholds=_method_thresholds(val_score,method_cols)
    a_table=_module_a_table(test_score,labels,thresholds)
    f_table=_module_b_table(test, test_run.forecast)
    lead=lead_time_from_screening(test_run.canonical,test_run.screening,first_flag_h=24,horizon_h=168)
    lead_vals=lead.loc[lead.future_defective.eq(1),"lead_time_h"].dropna()
    median_lead=float(lead_vals.median()) if len(lead_vals) else None
    robustness={k:{"rows":len(v),"columns":len(v.columns),"missing_fraction":float(v.isna().mean().mean())} for k,v in robustness_suite(test.head(min(5000,len(test))),seed).items()}
    escape=escape_matrix(test).groupby("group").size().rename("parts").reset_index()
    a_table.to_csv(output/"module_a_comparison.csv",index=False); f_table.to_csv(output/"module_b_comparison.csv",index=False); escape.to_csv(output/"escape_matrix.csv",index=False); pd.DataFrame([{"scenario":k,**v} for k,v in robustness.items()]).to_csv(output/"robustness_summary.csv",index=False)
    md=["# Component Screening Benchmark","","**Protocol:** model selection and threshold calibration use validation lots only; test lots are untouched.","",
        "## Module A — latent-defect screening","",a_table[["method","latent_escape_fnr","latent_escape_recall","recall","pr_auc","false_positive_rate","reject_rate"]].to_markdown(index=False),"",
        "## Module B — 168 h prediction","",f_table[["model","mae","rmse","conformal_coverage","limit_crossing_recall","dangerous_case_mae"]].to_markdown(index=False),"",
        f"**Median lead time for defective parts first flagged at 24 h:** {median_lead if median_lead is not None else 'not measurable'} h","",
        "## Escape matrix","",escape.to_markdown(index=False),"",
        "## Robustness campaign","",pd.DataFrame([{"scenario":k,**v} for k,v in robustness.items()]).to_markdown(index=False),"",
        "## Interpretation","",
        "The benchmark compares the screening evidence channels on the same untouched test lots. A lower latent-escape FNR is better. Lower 168 h MAE/RMSE is better. Robustness results describe controlled perturbations and are not substitutes for real ISRO validation.",""]
    (output.parent/"benchmark_table.md").write_text("\n".join(md),encoding="utf-8")
    result={"seed":seed,"train_lots":sorted(train.lot_id.astype(str).unique()),"validation_lots":sorted(val.lot_id.astype(str).unique()),"test_lots":sorted(test.lot_id.astype(str).unique()),"thresholds_validation":thresholds,"module_a":a_table.to_dict(orient="records"),"module_b":f_table.to_dict(orient="records"),"median_lead_time_h":median_lead,"robustness":robustness}
    (output/"benchmark_results.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output-dir",default="reports/benchmark"); ap.add_argument("--seed",type=int,default=SEED); args=ap.parse_args(); print(json.dumps(run(args.input,args.output_dir,args.seed),indent=2,default=str))
