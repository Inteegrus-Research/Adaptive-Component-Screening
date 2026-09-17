"""Final evaluation and evidence layer for the component screening system.

This module intentionally contains evaluation/validation logic rather than model
training logic. It answers the questions a reliability reviewer will ask:

* Does the system reduce latent-defect escapes relative to absolute limits/PAT?
* What is the 168 h MAE of each forecasting strategy?
* What happens under schema, unit, missing-data, distribution and mechanism shift?
* How stable are thresholds and decisions?
* Can a result be traced back to data/model/configuration fingerprints?
* What is the lot/process state and what additional test would be most useful?

All outputs are dataframe/JSON-friendly so the application layer can render them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)

from src.utils import PROJECT_ROOT, load_yaml, stable_config_hash
from src.pipeline import ACSPipeline, run_screening
from src.ingest import ingest_dataframe

DEFAULT_ORIGINS = (12.0, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce") if isinstance(s, pd.Series) else pd.Series(dtype=float)


def _binary_labels(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
    mapping = {"1":1,"true":1,"yes":1,"bad":1,"fail":1,"failed":1,"defective":1,"latent":1,"hard_failure":1,
               "0":0,"false":0,"no":0,"good":0,"pass":0,"safe":0,"healthy":0}
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)


def _find_col(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for n in names:
        if n in df.columns: return n
        if str(n).lower() in lower: return lower[str(n).lower()]
    return None


def _part_series(df: pd.DataFrame) -> pd.Series:
    c = _find_col(df,["part_id","part","component_id","serial_id"])
    if c is None:
        return pd.Series([f"ROW_{i}" for i in range(len(df))], index=df.index)
    return df[c].astype(str)


def _lot_series(df: pd.DataFrame) -> pd.Series:
    c = _find_col(df,["lot_id","lot","batch_id","batch"])
    if c is None:
        return pd.Series(["LOT_UNKNOWN"] * len(df), index=df.index)
    return df[c].astype(str)


# ---------------------------------------------------------------------------
# Escape matrix + FN-cost optimization
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    threshold: float
    fn: int
    fp: int
    tn: int
    tp: int
    fnr: float
    fpr: float
    precision: float
    recall: float
    review_rate: float
    reject_rate: float
    cost: float


def escape_matrix(input_df: pd.DataFrame, label_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the four-way conventional-screening / future-outcome matrix.

    Expected columns are discovered from common names. `absolute_pass` means
    no known engineering upper/lower violation. `future_defective` is the
    true future label (or the latent/hard label in the reference benchmark).
    """
    d = input_df.copy()
    pid = _part_series(d)
    if label_df is not None:
        ld = label_df.copy(); lc = _find_col(ld,["part_id","part","component_id"])
        if lc is not None:
            d = d.merge(ld, left_on=pid.name if pid.name else "part_id", right_on=lc, how="left", suffixes=("","_label"))
    abs_col = _find_col(d,["absolute_violation","absolute_fail","spec_fail","hard_limit_violation"])
    if abs_col is None:
        abs_pass = pd.Series(True,index=d.index)
    else:
        abs_pass = ~_binary_labels(d[abs_col]).astype(bool)
    fut_col = _find_col(d,["future_defective","defective","latent_defect","hard_failure","target_failure","future_failure"])
    if fut_col is None:
        # If a screening label exists but is named final_status, map it.
        fut_col = _find_col(d,["final_status","status","label"])
    future_bad = _binary_labels(d[fut_col]) if fut_col else pd.Series(0,index=d.index)
    out = pd.DataFrame({"part_id":pid.astype(str),"absolute_pass":abs_pass.astype(bool),"future_defective":future_bad.astype(int)})
    out["latent_escape"] = out["absolute_pass"] & out["future_defective"].eq(1)
    out["group"] = np.select(
        [out["absolute_pass"] & out["future_defective"].eq(0),
         out["absolute_pass"] & out["future_defective"].eq(1),
         ~out["absolute_pass"] & out["future_defective"].eq(0),
         ~out["absolute_pass"] & out["future_defective"].eq(1)],
        ["PASS_SAFE","LATENT_ESCAPE","ABSOLUTE_FAIL_SAFE","ABSOLUTE_FAIL_DEFECT"], default="UNKNOWN")
    return out.drop_duplicates("part_id")


def optimize_threshold(y_true: Sequence[int], score: Sequence[float], *, fn_cost: float = 100.0,
                       fp_cost: float = 1.0, reject_rate_max: float | None = None,
                       fpr_max: float | None = None, grid: int = 201) -> ThresholdResult:
    y=np.asarray(y_true,dtype=int); s=np.asarray(score,dtype=float)
    best=None
    for t in np.linspace(0,1,max(3,grid)):
        pred=(s>=t).astype(int)
        tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
        fpr=fp/max(fp+tn,1); fnr=fn/max(fn+tp,1); reject=float(pred.mean())
        if reject_rate_max is not None and reject>float(reject_rate_max): continue
        if fpr_max is not None and fpr>float(fpr_max): continue
        cost=fn_cost*fn + fp_cost*fp
        r=ThresholdResult(float(t),fn,fp,tn,tp,fnr,fpr,tp/max(tp+fp,1),tp/max(tp+fn,1),reject,reject,cost)
        # lexicographically prioritize FN, then cost, then rejection burden.
        key=(r.fn, r.cost, r.reject_rate)
        if best is None or key<best[0]: best=(key,r)
    if best is None:
        # Feasibility fallback: minimize false negatives then constraints violation.
        vals=[]
        for t in np.linspace(0,1,max(3,grid)):
            pred=(s>=t).astype(int); tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
            vals.append((fn,fp/max(fp+tn,1),float(pred.mean()),t,tp,tn,fp))
        _,_,_,t,tp,tn,fp=min(vals)
        fn=int(((s<t)&(y==1)).sum()); fpr=fp/max(fp+tn,1)
        return ThresholdResult(float(t),fn,int(fp),int(tn),int(tp),fn/max(fn+tp,1),fpr,tp/max(tp+fp,1),tp/max(tp+fn,1),float((s>=t).mean()),float((s>=t).mean()),fn_cost*fn+fp_cost*fp)
    return best[1]


def evaluate_anomaly_scores(y_true: Sequence[int], score: Sequence[float], threshold: float) -> dict[str,float]:
    y=np.asarray(y_true,dtype=int); s=np.asarray(score,dtype=float); p=(s>=threshold).astype(int)
    tp=int(((p==1)&(y==1)).sum()); tn=int(((p==0)&(y==0)).sum()); fp=int(((p==1)&(y==0)).sum()); fn=int(((p==0)&(y==1)).sum())
    out={"threshold":float(threshold),"tp":tp,"tn":tn,"fp":fp,"fn":fn,
         "fnr":fn/max(fn+tp,1),"fpr":fp/max(fp+tn,1),"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"reject_rate":float(p.mean()),
         "escape_fnr":float(fn/max(fn+tp,1))}
    try:
        out["pr_auc"]=float(average_precision_score(y,s)); out["roc_auc"]=float(roc_auc_score(y,s))
    except Exception:
        out["pr_auc"]=float("nan"); out["roc_auc"]=float("nan")
    return out


def threshold_stability(y_true: Sequence[int], score: Sequence[float], *, repeats: int = 50, seed: int = 20260831,
                        fn_cost: float = 100.0, fp_cost: float = 1.0) -> dict[str,float]:
    rng=np.random.default_rng(seed); y=np.asarray(y_true); s=np.asarray(score); ts=[]; fnrs=[]
    n=len(y)
    for _ in range(max(2,repeats)):
        idx=rng.integers(0,n,n); r=optimize_threshold(y[idx],s[idx],fn_cost=fn_cost,fp_cost=fp_cost)
        ts.append(r.threshold); fnrs.append(r.fnr)
    return {"threshold_mean":float(np.mean(ts)),"threshold_std":float(np.std(ts,ddof=1)),"threshold_min":float(np.min(ts)),"threshold_max":float(np.max(ts)),"fnr_mean":float(np.mean(fnrs)),"fnr_std":float(np.std(fnrs,ddof=1))}


# ---------------------------------------------------------------------------
# Forecast benchmark
# ---------------------------------------------------------------------------

def forecast_baselines(df: pd.DataFrame, horizon_h: float = 168.0) -> pd.DataFrame:
    d=df.copy(); v0=pd.to_numeric(d.get("value_0h"),errors="coerce"); v24=pd.to_numeric(d.get("value_24h"),errors="coerce")
    out=pd.DataFrame({"part_id":_part_series(d)})
    out["actual_168h"]=pd.to_numeric(d.get("target_168h",d.get("value_168h")),errors="coerce")
    out["persistence"]=v24
    slope=(v24-v0)/24.0; out["linear"] = v24+slope*(horizon_h-24.0)
    # Robust last-slope clips extreme early slopes by a median/MAD envelope.
    med=float(np.nanmedian(slope)); mad=float(np.nanmedian(np.abs(slope-med))) if np.isfinite(slope).any() else 0.0
    scale=max(1.4826*mad,1e-12); clipped=np.clip(slope,med-4*scale,med+4*scale)
    out["robust_linear"]=v24+clipped*(horizon_h-24.0)
    return out


def benchmark_forecasts(df: pd.DataFrame, predictions: Mapping[str, Sequence[float]] | None = None, horizon_h: float = 168.0) -> pd.DataFrame:
    b=forecast_baselines(df,horizon_h); y=b["actual_168h"]
    rows=[]
    for name in ["persistence","linear","robust_linear"]:
        valid=y.notna() & b[name].notna()
        if valid.any(): rows.append({"model":name,"n":int(valid.sum()),"mae":float(mean_absolute_error(y[valid],b.loc[valid,name])),"rmse":float(mean_squared_error(y[valid],b.loc[valid,name])**0.5)})
    if predictions:
        for name,p in predictions.items():
            p=pd.Series(p,index=b.index,dtype=float); valid=y.notna()&p.notna()
            if valid.any(): rows.append({"model":name,"n":int(valid.sum()),"mae":float(mean_absolute_error(y[valid],p[valid])),"rmse":float(mean_squared_error(y[valid],p[valid])**0.5)})
    return pd.DataFrame(rows).sort_values("mae") if rows else pd.DataFrame(columns=["model","n","mae","rmse"])


def leakage_audit(df: pd.DataFrame, as_of_h: float = 24.0, forbidden: Iterable[str] | None = None) -> dict[str,Any]:
    """Static audit of a forecast feature frame against a declared as-of boundary."""
    forb={str(x).lower() for x in (forbidden or ["value_96h","value_168h","target_168h","future_defective","future_failure","latent_defect","hard_failure"])}
    offenders=[]
    for c in df.columns:
        lc=str(c).lower()
        if lc in forb or any(tok in lc for tok in ["168h","future","target_168","label"]): offenders.append(str(c))
    time_cols=[]
    for c in df.columns:
        m=re.search(r"(?:value|val)[_@]?(\d+(?:\.\d+)?)h$",str(c),re.I)
        if m and float(m.group(1))>float(as_of_h): time_cols.append(str(c))
    offenders=sorted(set(offenders+time_cols))
    return {"as_of_h":float(as_of_h),"leakage_detected":bool(offenders),"forbidden_columns_detected":offenders}


# ---------------------------------------------------------------------------
# Domain/adversarial benchmark helpers
# ---------------------------------------------------------------------------

def mutate_schema_aliases(df: pd.DataFrame, alias_map: Mapping[str,str]) -> pd.DataFrame:
    return df.rename(columns={k:v for k,v in alias_map.items() if k in df.columns}).copy()


def mutate_units(df: pd.DataFrame, column: str, factor: float, new_unit: str | None = None, unit_column: str = "unit") -> pd.DataFrame:
    d=df.copy();
    if column in d.columns: d[column]=pd.to_numeric(d[column],errors="coerce")*float(factor)
    if new_unit is not None and unit_column in d.columns: d[unit_column]=new_unit
    return d


def drop_parameters(df: pd.DataFrame, fraction: float, seed: int = 20260831) -> pd.DataFrame:
    d=df.copy()
    if "parameter" not in d.columns: return d
    params=sorted(d["parameter"].dropna().astype(str).unique().tolist()); rng=np.random.default_rng(seed)
    k=max(1,int(round(len(params)*float(np.clip(fraction,0,1)))))
    remove=set(rng.choice(params,size=min(k,len(params)),replace=False).tolist())
    return d[~d["parameter"].astype(str).isin(remove)].copy()


def add_noise(df: pd.DataFrame, fraction: float, seed: int = 20260831, value_cols: Sequence[str] | None = None) -> pd.DataFrame:
    d=df.copy(); rng=np.random.default_rng(seed)
    cols=list(value_cols or [c for c in d.columns if re.search(r"value|^measurement$|^observed$",str(c),re.I)])
    for c in cols:
        if c in d.columns:
            x=pd.to_numeric(d[c],errors="coerce"); sd=float(np.nanstd(x)) if x.notna().any() else 1.0
            d[c]=x+rng.normal(0,max(sd*float(fraction),1e-12),len(d))
    return d


def shift_distribution(df: pd.DataFrame, mean_factor: float = 1.25, std_factor: float = 1.0, skew_proxy: float = 0.0,
                       column: str = "value_24h", seed: int = 20260831) -> pd.DataFrame:
    d=df.copy(); rng=np.random.default_rng(seed)
    if column in d.columns:
        x=pd.to_numeric(d[column],errors="coerce"); med=float(np.nanmedian(x)); sd=float(np.nanstd(x))
        shifted=med+(x-med)*float(std_factor)+(med*float(mean_factor)-med)
        if skew_proxy:
            shifted=shifted+float(skew_proxy)*rng.normal(0,sd,max(1,len(d)))**2/max(sd,1e-12)
        d[column]=shifted
    return d


def adversarial_suite(df: pd.DataFrame) -> dict[str,pd.DataFrame]:
    """Create a deterministic, label-free set of input stressors.

    These mutations test the architecture's data contract. They intentionally do
    not fabricate future targets.
    """
    out={"baseline":df.copy()}
    aliases={"IDDQ_uA":"Icc_q","value_0h":"v0","value_24h":"v24"}
    out["renamed"] = mutate_schema_aliases(df,aliases)
    if "value_24h" in df.columns: out["unit_scaled"] = mutate_units(df,"value_24h",1e-3,"mA")
    out["missing_20pct"] = drop_parameters(df,0.20)
    out["missing_40pct"] = drop_parameters(df,0.40)
    out["missing_60pct"] = drop_parameters(df,0.60)
    out["shifted_mean"] = shift_distribution(df,mean_factor=1.25)
    out["shifted_variance"] = shift_distribution(df,std_factor=1.75)
    out["noise_5pct"] = add_noise(df,0.05)
    # Informative missingness: preferentially mask higher-value observations.
    inf=df.copy()
    if "value_24h" in inf.columns:
        x=pd.to_numeric(inf["value_24h"],errors="coerce"); threshold=float(x.quantile(0.9)) if x.notna().any() else np.inf; inf.loc[x>=threshold,"value_24h"]=np.nan
    out["informative_missing"] = inf
    if "value_168h" in df.columns: out["irregular_readpoints"] = df.drop(columns=["value_96h","value_168h"],errors="ignore")
    return out


def robustness_summary(run_results: Mapping[str,Mapping[str,float]]) -> pd.DataFrame:
    rows=[]
    for scenario,metrics in run_results.items():
        r={"scenario":scenario}; r.update({k:float(v) if isinstance(v,(int,float,np.number)) else v for k,v in metrics.items()}); rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lot/process intelligence
# ---------------------------------------------------------------------------

def lot_health(input_df: pd.DataFrame, anomaly: pd.DataFrame | None = None, forecast: pd.DataFrame | None = None) -> pd.DataFrame:
    d=input_df.copy(); d["_part"]=_part_series(d); d["_lot"]=_lot_series(d)
    rows=[]
    a = anomaly.copy() if anomaly is not None else pd.DataFrame()
    f = forecast.copy() if forecast is not None else pd.DataFrame()
    a_pid=_find_col(a,["part_id"]) if not a.empty else None; f_pid=_find_col(f,["part_id"]) if not f.empty else None
    for lot,g in d.groupby("_lot",sort=True):
        parts=g["_part"].nunique(); row={"lot_id":lot,"parts":int(parts)}
        if not a.empty and a_pid:
            ag=a[a[a_pid].astype(str).isin(g["_part"].astype(str))]
            ac=_find_col(ag,["anomaly_risk","calibrated_risk_score","anomaly_score"])
            row["mean_anomaly_risk"]=float(pd.to_numeric(ag[ac],errors="coerce").mean()) if ac else 0.0
            row["anomalous_part_fraction"]=float((pd.to_numeric(ag[ac],errors="coerce")>=0.65).mean()) if ac else 0.0
        else:
            row["mean_anomaly_risk"]=0.0; row["anomalous_part_fraction"]=0.0
        if not f.empty and f_pid:
            fg=f[f[f_pid].astype(str).isin(g["_part"].astype(str))]
            ex=_find_col(fg,["limit_exceedance_probability_proxy","failure_risk"])
            row["mean_failure_risk"]=float(pd.to_numeric(fg[ex],errors="coerce").mean()) if ex else 0.0
        else: row["mean_failure_risk"]=0.0
        row["lot_health_score"]=float(np.clip(1.0-(0.65*row["anomalous_part_fraction"]+0.35*row["mean_failure_risk"]),0,1))
        row["process_excursion_suspect"]=bool(row["anomalous_part_fraction"]>=0.20 or row["mean_anomaly_risk"]>=0.65)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Early-detection, next-best-test, screening twin, failure-pattern attribution
# ---------------------------------------------------------------------------

def detection_lead_time(traces: pd.DataFrame, threshold: float = 0.70, traditional_h: float = 168.0) -> pd.DataFrame:
    """Compute first reliable model detection time from long-form score traces.

    Expected columns: part_id, time_h, risk_score or anomaly_risk, future_defective.
    """
    d=traces.copy(); d["part_id"]=_part_series(d); d["time_h"]=pd.to_numeric(d.get("time_h"),errors="coerce")
    score_col=_find_col(d,["risk_score","anomaly_risk","score"])
    ycol=_find_col(d,["future_defective","defective","latent_defect"])
    if score_col is None or ycol is None: raise ValueError("Detection lead time requires score and future-defective columns.")
    rows=[]
    for pid,g in d.groupby("part_id",sort=False):
        if int(_binary_labels(g[ycol]).max())!=1: continue
        gg=g.sort_values("time_h"); hits=gg[pd.to_numeric(gg[score_col],errors="coerce")>=threshold]
        first=float(hits["time_h"].iloc[0]) if not hits.empty else np.nan
        rows.append({"part_id":str(pid),"first_detection_h":first,"lead_time_h":float(traditional_h-first) if np.isfinite(first) else np.nan,"detected":bool(np.isfinite(first))})
    return pd.DataFrame(rows)


def next_best_test(input_df: pd.DataFrame, screening: pd.DataFrame, *, target_parameters: Sequence[str] | None = None) -> pd.DataFrame:
    """Recommend a next measurement using uncertainty × relevance.

    This is diagnostic decision support only; it does not control hardware.
    """
    d=input_df.copy(); s=screening.copy(); pid_s=_find_col(s,["part_id"]); part=_part_series(d)
    candidates=[]
    params=sorted(d["parameter"].dropna().astype(str).unique()) if "parameter" in d.columns else list(target_parameters or [])
    for pid in part.unique():
        sr=s[s[pid_s].astype(str).eq(str(pid))] if pid_s else pd.DataFrame()
        risk=float(pd.to_numeric(sr[_find_col(sr,["risk_score"])],errors="coerce").max()) if not sr.empty and _find_col(sr,["risk_score"]) else 0.5
        unc=float(pd.to_numeric(sr[_find_col(sr,["uncertainty_score"])],errors="coerce").max()) if not sr.empty and _find_col(sr,["uncertainty_score"]) else 0.5
        priority=risk*unc
        for p in params:
            pg=d[(part==str(pid)) & d["parameter"].astype(str).eq(p)] if "parameter" in d.columns else pd.DataFrame()
            if pg.empty: continue
            value=pd.to_numeric(pg.get("value_asof",pg.get("value")),errors="coerce")
            repeat=float(1.0/value.std()) if value.notna().sum()>1 and value.std()>0 else 0.0
            candidates.append({"part_id":str(pid),"parameter":p,"priority":float(priority+0.05*repeat),"reason":"high risk × uncertainty; repeat/measure this parameter for the most information."})
    return pd.DataFrame(candidates).sort_values(["part_id","priority"],ascending=[True,False]) if candidates else pd.DataFrame(columns=["part_id","parameter","priority","reason"])


def what_if_projection(value_0h: float, value_24h: float, horizons: Sequence[float] = (48,72,96,120,144,168)) -> pd.DataFrame:
    slope=(float(value_24h)-float(value_0h))/24.0
    rows=[]
    for h in horizons:
        pred=float(value_24h+slope*(float(h)-24.0)); rows.append({"time_h":float(h),"projected_value":pred,"method":"linear early-drift scenario"})
    return pd.DataFrame(rows)


def failure_pattern_attribution(row: Mapping[str,Any]) -> dict[str,Any]:
    reasons=[]
    pop=float(row.get("population_evidence",0) or 0); temp=float(row.get("temporal_evidence",0) or 0); multi=float(row.get("multivariate_component_score",0) or 0)
    if temp>=0.70: reasons.append("accelerating_or_abnormal_temporal_drift")
    if pop>=0.70: reasons.append("lot_relative_parametric_anomaly")
    if multi>=0.70: reasons.append("cross_parameter_joint_anomaly")
    if row.get("hard_limit_violation"): reasons.append("absolute_specification_violation")
    if row.get("limit_cross"): reasons.append("predicted_future_limit_crossing")
    return {"patterns":reasons[:4],"language":"pattern consistent with observed evidence; not a physical root-cause diagnosis."}


# ---------------------------------------------------------------------------
# Physics/empirical sanity gates and transfer profile
# ---------------------------------------------------------------------------

def physics_sanity(df: pd.DataFrame) -> dict[str,Any]:
    violations=[]
    for c in [c for c in df.columns if re.search(r"temperature",str(c),re.I)]:
        x=pd.to_numeric(df[c],errors="coerce"); bad=x.notna() & ((x<-273.15)|(x>1000))
        if bad.any(): violations.append(f"{c}: physically implausible temperature range")
    for c in [c for c in df.columns if re.search(r"(?:voltage|current|capacitance|leakage|resistance|esr|iddq|idss)",str(c),re.I)]:
        x=pd.to_numeric(df[c],errors="coerce")
        if x.notna().any() and re.search(r"(?:current|capacitance|resistance|esr|iddq|idss|leakage)",str(c),re.I) and (x.dropna()<0).any():
            violations.append(f"{c}: negative value where the modeled quantity is non-negative")
    return {"pass":not violations,"violations":violations}


def transfer_profile(reference: pd.DataFrame, external: pd.DataFrame) -> dict[str,Any]:
    """Compare semantic/shape overlap for an external real dataset.

    This does not claim domain equivalence; it measures whether the universal
    adapter can represent the external dataset without the synthetic schema.
    """
    common=[]
    for c in ["component_family","parameter","semantic_type","physical_quantity","unit"]:
        if c in reference.columns and c in external.columns: common.append(c)
    result={"common_semantic_columns":common,
            "reference_families":sorted(set(reference.get("component_family",pd.Series(dtype=str)).dropna().astype(str))),
            "external_families":sorted(set(external.get("component_family",pd.Series(dtype=str)).dropna().astype(str))),
            "parameter_overlap":0.0,
            "supports_common_times":False}
    rp=set(reference.get("parameter",pd.Series(dtype=str)).dropna().astype(str)); ep=set(external.get("parameter",pd.Series(dtype=str)).dropna().astype(str));
    result["parameter_overlap"]=float(len(rp&ep)/max(len(ep),1))
    rt={float(x) for x in pd.to_numeric(reference.get("time_h",pd.Series(dtype=float)),errors="coerce").dropna().unique()}; et={float(x) for x in pd.to_numeric(external.get("time_h",pd.Series(dtype=float)),errors="coerce").dropna().unique()}
    result["supports_common_times"]=bool(rt & et)
    return result


# ---------------------------------------------------------------------------
# Master evidence report
# ---------------------------------------------------------------------------

def build_evidence_report(*, input_df: pd.DataFrame, screening: pd.DataFrame | None = None,
                          anomaly: pd.DataFrame | None = None, forecast: pd.DataFrame | None = None,
                          y_true: Sequence[int] | None = None, anomaly_score: Sequence[float] | None = None,
                          config: Mapping[str,Any] | None = None) -> dict[str,Any]:
    cfg=config or load_yaml(PROJECT_ROOT/"configs"/"models.yaml")
    rep={"configuration_hash":stable_config_hash(cfg),"rows":int(len(input_df)),"parts":int(_part_series(input_df).nunique()),
         "physics_sanity":physics_sanity(input_df),"leakage_audit":leakage_audit(input_df)}
    if y_true is not None and anomaly_score is not None:
        pol=load_yaml(PROJECT_ROOT/"configs"/"policy.yaml")
        costs=pol.get("costs",{}); tr=optimize_threshold(y_true,anomaly_score,fn_cost=float(costs.get("false_negative",100)),fp_cost=float(costs.get("false_positive",1)))
        rep["fn_optimized_threshold"]=asdict(tr); rep["threshold_stability"]=threshold_stability(y_true,anomaly_score,fn_cost=float(costs.get("false_negative",100)),fp_cost=float(costs.get("false_positive",1)))
        rep["anomaly_metrics"]=evaluate_anomaly_scores(y_true,anomaly_score,tr.threshold)
    if forecast is not None:
        # If forecast has point prediction + hidden target, evaluate it directly.
        pc=_find_col(forecast,["prediction_168h"]); yc=_find_col(forecast,["actual_168h","target_168h"])
        if pc and yc:
            y=pd.to_numeric(forecast[yc],errors="coerce"); p=pd.to_numeric(forecast[pc],errors="coerce"); valid=y.notna()&p.notna()
            rep["forecast_metrics"]={"n":int(valid.sum()),"mae":float(mean_absolute_error(y[valid],p[valid])) if valid.any() else np.nan,"rmse":float(mean_squared_error(y[valid],p[valid])**0.5) if valid.any() else np.nan}
    if screening is not None:
        rep["lot_health"]=lot_health(input_df,anomaly,forecast).to_dict(orient="records")
        rep["decisions"]=screening.get("decision",pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
    return rep


def main() -> int:
    p=argparse.ArgumentParser(description="Final benchmark/evidence utilities")
    sub=p.add_subparsers(dest="cmd",required=True)
    pb=sub.add_parser("anomaly-threshold"); pb.add_argument("labels_csv"); pb.add_argument("score_csv"); pb.add_argument("--label-column",default="future_defective"); pb.add_argument("--score-column",default="score")
    pl=sub.add_parser("leakage-audit"); pl.add_argument("csv"); pl.add_argument("--as-of",type=float,default=24.0)
    ps=sub.add_parser("sanity"); ps.add_argument("csv")
    args=p.parse_args()
    if args.cmd=="anomaly-threshold":
        y=_binary_labels(pd.read_csv(args.labels_csv)[args.label_column]); s=pd.to_numeric(pd.read_csv(args.score_csv)[args.score_column],errors="coerce"); print(json.dumps(asdict(optimize_threshold(y,s)),indent=2)); return 0
    if args.cmd=="leakage-audit": print(json.dumps(leakage_audit(pd.read_csv(args.csv),args.as_of),indent=2)); return 0
    if args.cmd=="sanity": print(json.dumps(physics_sanity(pd.read_csv(args.csv)),indent=2)); return 0
    return 0


# ---------------------------------------------------------------------------
# Comparative screening, dependency, twin and calibration utilities
# ---------------------------------------------------------------------------

def _derive_population_score(df: pd.DataFrame) -> pd.Series:
    if "population_robust_z" in df.columns:
        return np.clip(np.abs(pd.to_numeric(df["population_robust_z"],errors="coerce").fillna(0))/8.0,0,1)
    value_col=_find_col(df,["value_24h","value_asof","value"])
    if value_col is None: return pd.Series(0.0,index=df.index)
    lot=_lot_series(df); x=pd.to_numeric(df[value_col],errors="coerce")
    med=x.groupby(lot).transform("median"); mad=(x-med).abs().groupby(lot).transform("median")
    z=(x-med)/(1.4826*mad.replace(0,np.nan)); return np.clip(z.abs().fillna(0)/8.0,0,1)


def _derive_absolute_violation(df: pd.DataFrame) -> pd.Series:
    c=_find_col(df,["absolute_violation","hard_limit_violation","spec_fail"])
    if c: return _binary_labels(df[c]).astype(bool)
    xcol=_find_col(df,["value_168h","value_24h","value_asof","value"]); lim=_find_col(df,["absolute_limit_upper","engineering_limit_upper","upper_limit","spec_upper"])
    if xcol and lim:
        return (pd.to_numeric(df[xcol],errors="coerce")>pd.to_numeric(df[lim],errors="coerce")).fillna(False)
    return pd.Series(False,index=df.index)


def comparative_anomaly_benchmark(df: pd.DataFrame, *, y_col: str = "future_defective", score_col: str | None = None,
                                  fn_cost: float = 100.0, fp_cost: float = 1.0) -> pd.DataFrame:
    """Compare absolute limits, PAT and AI risk on the same labelled population."""
    d=df.copy(); y=_binary_labels(d[y_col]) if y_col in d.columns else pd.Series(0,index=d.index)
    scores={"absolute_limits":_derive_absolute_violation(d).astype(float),
            "pat_robust":_derive_population_score(d)}
    if "temporal_evidence" in d.columns:
        scores["pat_plus_temporal"]=np.clip(0.55*scores["pat_robust"]+0.45*pd.to_numeric(d["temporal_evidence"],errors="coerce").fillna(0),0,1)
    else: scores["pat_plus_temporal"]=scores["pat_robust"]
    if score_col and score_col in d.columns: scores["ai_full"]=pd.to_numeric(d[score_col],errors="coerce").fillna(0).clip(0,1)
    rows=[]
    for name,s in scores.items():
        if name=="absolute_limits": threshold=0.5
        else: threshold=optimize_threshold(y,s,fn_cost=fn_cost,fp_cost=fp_cost).threshold
        rows.append({"method":name,**evaluate_anomaly_scores(y,s,threshold)})
    return pd.DataFrame(rows)


def cross_parameter_dependency(df: pd.DataFrame, *, min_parts: int = 20) -> pd.DataFrame:
    """Estimate a robust part-by-parameter dependency matrix and joint anomaly score."""
    if "part_id" not in df.columns or "parameter" not in df.columns: return pd.DataFrame()
    value_col=_find_col(df,["value_asof","value_24h","value"])
    if value_col is None: return pd.DataFrame()
    x=df.copy(); x["_v"]=pd.to_numeric(x[value_col],errors="coerce"); x["_p"]=x["part_id"].astype(str); x["_param"]=x["parameter"].astype(str)
    wide=x.pivot_table(index="_p",columns="_param",values="_v",aggfunc="median")
    wide=wide.dropna(axis=1,thresh=min_parts).dropna(axis=0,how="all")
    if wide.shape[1]<2: return pd.DataFrame()
    corr=wide.corr(method="spearman").stack().reset_index(); corr.columns=["parameter_a","parameter_b","spearman_corr"]
    return corr[corr["parameter_a"]<corr["parameter_b"]].sort_values("spearman_corr",ascending=False).reset_index(drop=True)


def screening_twin(value_0h: float, value_24h: float, *, horizons: Sequence[float]=(48,72,96,120,144,168),
                   temperature_C: float | None=None, reference_temperature_C: float | None=None,
                   activation_energy_eV: float | None=None, direction: str="increase") -> pd.DataFrame:
    """Scenario-level screening twin. It is explicitly a what-if tool, not a hardware-validated predictor."""
    import math as _math
    slope=(float(value_24h)-float(value_0h))/24.0; factor=1.0
    if temperature_C is not None and reference_temperature_C is not None and activation_energy_eV is not None:
        k=8.617333262e-5; T1=float(reference_temperature_C)+273.15; T2=float(temperature_C)+273.15
        factor=float(np.clip(np.exp(float(activation_energy_eV)/k*(1/T1-1/T2)),0.01,1e4))
    rows=[]
    for h in horizons:
        pred=float(value_24h+(slope*factor)*(float(h)-24.0)); rows.append({"time_h":float(h),"projected_value":pred,"stress_factor":factor,"method":"linear scenario with optional Arrhenius acceleration","is_validated_physics":False})
    return pd.DataFrame(rows)


def calibrate_cost_policy(y_true: Sequence[int], risk_score: Sequence[float], *, fn_cost: float=100.0, fp_cost: float=1.0,
                          review_rate_max: float=0.25) -> dict[str,Any]:
    r=optimize_threshold(y_true,risk_score,fn_cost=fn_cost,fp_cost=fp_cost,reject_rate_max=review_rate_max)
    return {"threshold":r.threshold,"false_negative_rate":r.fnr,"false_positive_rate":r.fpr,"precision":r.precision,"recall":r.recall,"review_rate":r.reject_rate,"cost":r.cost,
            "objective":"minimize FN-cost subject to a maximum automatic-rejection/review burden; threshold remains a policy parameter."}


def aggregate_decision_trace(screening: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _,r in screening.iterrows():
        trace={}
        raw=r.get("trace_json")
        if isinstance(raw,str):
            try: trace=json.loads(raw)
            except Exception: trace={}
        rows.append({"part_id":str(r.get("part_id")),"decision":r.get("decision"),"risk_score":r.get("risk_score"),
                     "anomaly_risk":r.get("anomaly_risk"),"failure_risk":r.get("failure_risk"),"ood_score":r.get("ood_score"),
                     "uncertainty_score":r.get("uncertainty_score"),"evidence_state":r.get("evidence_state"),
                     "hard_limit_violation":r.get("hard_limit_violation"),"trace":json.dumps(trace,default=str)})
    return pd.DataFrame(rows)


def save_json(data: Mapping[str,Any], path: str | Path) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(data,indent=2,default=str),encoding="utf-8")


def _load_for_benchmark(path: str | Path) -> pd.DataFrame:
    p=Path(path)
    if p.suffix.lower()=='.csv': return pd.read_csv(p)
    if p.suffix.lower() in {'.db','.sqlite','.sqlite3'}:
        import sqlite3
        with sqlite3.connect(p) as con:
            tables=[r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
            for t in ('module_A_dataset','module_B_train','labels','measurements','component_parameter_state','part_parameter_state'):
                if t in tables:
                    try:return pd.read_sql_query(f'SELECT * FROM {t}',con)
                    except Exception: pass
        raise ValueError('No compatible table found in database')
    raise ValueError(f'Unsupported data file: {p}')


def run_master_report(input_path: str | Path, output_dir: str | Path, *, screening_path: str | Path | None=None,
                      anomaly_path: str | Path | None=None, forecast_path: str | Path | None=None) -> dict[str,Any]:
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); d=_load_for_benchmark(input_path)
    screening=pd.read_csv(screening_path) if screening_path else None
    anomaly=pd.read_csv(anomaly_path) if anomaly_path else None
    forecast=pd.read_csv(forecast_path) if forecast_path else None
    rep=build_evidence_report(input_df=d,screening=screening,anomaly=anomaly,forecast=forecast)
    if screening is not None:
        lot=lot_health(d,anomaly,forecast); lot.to_csv(out/'lot_health.csv',index=False)
        nb=next_best_test(d,screening); nb.to_csv(out/'next_best_test.csv',index=False)
    (out/'evidence_report.json').write_text(json.dumps(rep,indent=2,default=str),encoding='utf-8')
    return rep


def master_cli() -> int:
    ap=argparse.ArgumentParser(description='Final evaluation, ablation and robustness campaign')
    sub=ap.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('benchmark'); b.add_argument('csv'); b.add_argument('--label-column',default='future_defective'); b.add_argument('--score-column',default='anomaly_risk'); b.add_argument('--output',default='reports/benchmark.csv')
    e=sub.add_parser('escape'); e.add_argument('csv'); e.add_argument('--label-column',default='future_defective'); e.add_argument('--output',default='reports/escape_matrix.csv')
    a=sub.add_parser('adversarial'); a.add_argument('csv'); a.add_argument('--output-dir',default='reports/adversarial')
    r=sub.add_parser('robustness'); r.add_argument('csv'); r.add_argument('--output',default='reports/robustness_catalog.csv')
    l=sub.add_parser('lot-health'); l.add_argument('csv'); l.add_argument('--anomaly'); l.add_argument('--forecast'); l.add_argument('--output',default='reports/lot_health.csv')
    s=sub.add_parser('sanity'); s.add_argument('csv')
    q=sub.add_parser('leakage-audit'); q.add_argument('csv'); q.add_argument('--as-of',type=float,default=24.0)
    t=sub.add_parser('threshold'); t.add_argument('csv'); t.add_argument('--label-column',default='future_defective'); t.add_argument('--score-column',default='anomaly_risk'); t.add_argument('--output',default='reports/fn_policy.json')
    w=sub.add_parser('twin'); w.add_argument('value_0h',type=float); w.add_argument('value_24h',type=float); w.add_argument('--output',default='reports/screening_twin.csv')
    n=sub.add_parser('next-test'); n.add_argument('input_csv'); n.add_argument('screening_csv'); n.add_argument('--output',default='reports/next_best_test.csv')
    x=sub.add_parser('transfer-profile'); x.add_argument('reference_csv'); x.add_argument('external_csv');
    args=ap.parse_args()
    if args.cmd=='benchmark':
        d=_load_for_benchmark(args.csv); out=comparative_anomaly_benchmark(d,y_col=args.label_column,score_col=args.score_column); Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False); print(out.to_json(orient='records',indent=2)); return 0
    if args.cmd=='escape':
        out=escape_matrix(_load_for_benchmark(args.csv)); Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False); print(out['group'].value_counts().to_json(indent=2)); return 0
    if args.cmd=='adversarial':
        d=_load_for_benchmark(args.csv); outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
        s=adversarial_suite(d)
        for k,v in s.items(): v.to_csv(outdir/f'{k}.csv',index=False)
        print(json.dumps({'scenarios':list(s),'output_dir':str(outdir)},indent=2)); return 0
    if args.cmd=='robustness':
        d=_load_for_benchmark(args.csv); scenarios=adversarial_suite(d); rows=[]
        for k,v in scenarios.items():
            san=physics_sanity(v); rows.append({'scenario':k,'rows':len(v),'parts':int(_part_series(v).nunique()),'physics_pass':san['pass'],'missing_fraction':float(v.isna().mean().mean()),'schema_columns':len(v.columns)})
        out=pd.DataFrame(rows); Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False); print(out.to_json(orient='records',indent=2)); return 0
    if args.cmd=='lot-health':
        d=_load_for_benchmark(args.csv); a=pd.read_csv(args.anomaly) if args.anomaly else None; f=pd.read_csv(args.forecast) if args.forecast else None; out=lot_health(d,a,f); Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False); print(out.to_json(orient='records',indent=2)); return 0
    if args.cmd=='sanity': print(json.dumps(physics_sanity(_load_for_benchmark(args.csv)),indent=2)); return 0
    if args.cmd=='leakage-audit': print(json.dumps(leakage_audit(_load_for_benchmark(args.csv),args.as_of),indent=2)); return 0
    if args.cmd=='threshold':
        d=_load_for_benchmark(args.csv); y=_binary_labels(d[args.label_column]); s=pd.to_numeric(d[args.score_column],errors='coerce').fillna(0); cfg=load_yaml(PROJECT_ROOT/'configs'/'policy.yaml'); costs=cfg.get('costs',{}); out=calibrate_cost_policy(y,s,fn_cost=float(costs.get('false_negative',100)),fp_cost=float(costs.get('false_positive',1))); Path(args.output).parent.mkdir(parents=True,exist_ok=True); save_json(out,args.output); print(json.dumps(out,indent=2)); return 0
    if args.cmd=='twin':
        out=screening_twin(args.value_0h,args.value_24h); Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False); print(out.to_json(orient='records',indent=2)); return 0
    if args.cmd=='next-test':
        d=_load_for_benchmark(args.input_csv); s=pd.read_csv(args.screening_csv); out=next_best_test(d,s); Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False); print(out.head(20).to_json(orient='records',indent=2)); return 0
    if args.cmd=='transfer-profile':
        r=_load_for_benchmark(args.reference_csv); e=_load_for_benchmark(args.external_csv); print(json.dumps(transfer_profile(r,e),indent=2)); return 0
    return 0

# ---------------------------------------------------------------------------
# V4 pipeline integration
# ---------------------------------------------------------------------------

def _drop_evaluation_only_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ground-truth/future-label fields before model execution."""
    result = df.copy()
    patterns = [
        r"^future_defective(?:_\d+h)?$",
        r"^defect_state$",
        r"^latent_defect_label$",
        r"^failure_mode$",
        r"^absolute_fail_\d+h$",
        r"^ground_truth$",
        r"^target_\d+h$",
    ]
    drop = [
        c for c in result.columns
        if any(re.search(p, str(c), flags=re.IGNORECASE) for p in patterns)
    ]
    return result.drop(columns=drop, errors="ignore")


def _canonical_part_column(df: pd.DataFrame) -> str | None:
    return _find_col(df, ["component_id", "part_id", "part", "device_id", "serial_id"])


def _truth_table(df: pd.DataFrame, horizon_h: float = 168.0) -> pd.DataFrame:
    part_col = _canonical_part_column(df)
    if part_col is None:
        return pd.DataFrame(columns=["component_id", "future_defective", "defect_state", "split", "lot_id"])
    d = df.copy()
    d["component_id"] = d[part_col].astype(str)
    future = None
    for candidate in [f"future_defective_{int(horizon_h)}h", "future_defective", "future_failure", "defective"]:
        if candidate in d.columns:
            future = _binary_labels(d[candidate])
            break
    if future is None and "defect_state" in d.columns:
        future = d["defect_state"].astype(str).str.lower().isin({"latent", "hard", "defective", "failed", "failure"}).astype(int)
    if future is None:
        future = pd.Series(0, index=d.index, dtype=int)
    d["future_defective"] = future.astype(int)
    keep = ["component_id", "future_defective"]
    if "defect_state" in d.columns:
        keep.append("defect_state")
    if "split" in d.columns:
        keep.append("split")
    lot = _find_col(d, ["lot_id", "lot", "batch_id", "batch"])
    if lot is not None:
        d["lot_id"] = d[lot].astype(str)
        keep.append("lot_id")
    return d[keep].drop_duplicates("component_id")


def _canonicalize_for_v4(data: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize once for progressive slicing while preserving V4 semantics."""
    canonical, _ = ingest_dataframe(data, parameters_path=PROJECT_ROOT / "configs" / "parameters.yaml")
    return canonical


def progressive_screen_dataframe(
    data: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    origins: Sequence[float] = DEFAULT_ORIGINS,
    target_horizon: float = 168.0,
    parameters_path: str | Path = PROJECT_ROOT / "configs" / "parameters.yaml",
    policy_path: str | Path = PROJECT_ROOT / "configs" / "policy.yaml",
) -> dict[str, Any]:
    """Leak-safe progressive screening using the V4 pipeline.

    Each origin only receives observations with time_h <= origin. The future
    labels remain outside the model input and are used only to calculate
    progressive detection/lead-time metrics.
    """
    canonical, _audit = ingest_dataframe(
        data,
        parameters_path=parameters_path,
    )
    truth = _truth_table(data, target_horizon)
    root = Path(output_dir) if output_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    decisions: list[pd.DataFrame] = []

    for origin in origins:
        early = canonical.loc[
            pd.to_numeric(canonical["time_h"], errors="coerce") <= float(origin)
        ].copy()
        if early.empty:
            continue

        run = run_screening(
            early,
            parameters_path=str(parameters_path),
            policy_path=str(policy_path),
            target_horizon_h=float(target_horizon),
        )
        triage = run.triage.copy()
        triage["origin_h"] = float(origin)
        triage = triage.merge(
            truth[["component_id", "future_defective"]],
            on="component_id",
            how="left",
        )
        decisions.append(triage)

        run_record = {
            "origin_h": float(origin),
            "rows": int(len(early)),
            "components": int(run.triage["component_id"].nunique()) if not run.triage.empty else 0,
            "triage": run.triage,
            "capability_manifest": asdict(run.capability_manifest),
            "ingestion_audit": run.ingestion_audit.to_dict(),
        }
        runs.append(run_record)

        if root is not None:
            od = root / f"origin_{int(origin)}h"
            od.mkdir(parents=True, exist_ok=True)
            early.to_csv(od / "input_canonical.csv", index=False)
            run.triage.to_csv(od / "triage.csv", index=False)
            run.feature_table.to_csv(od / "features.csv", index=False)
            run.canonical_telemetry.to_csv(od / "canonical_telemetry.csv", index=False)
            (od / "capability_manifest.json").write_text(json.dumps(asdict(run.capability_manifest), indent=2, default=str), encoding="utf-8")
            (od / "ingestion_audit.json").write_text(json.dumps(run.ingestion_audit.to_dict(), indent=2, default=str), encoding="utf-8")

    combined = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    lead = _progressive_lead_time(combined, target_horizon=target_horizon)

    if root is not None:
        combined.to_csv(root / "progressive_decisions.csv", index=False)
        lead.to_csv(root / "progressive_lead_time.csv", index=False)

    return {"runs": runs, "decisions": combined, "lead_time": lead}


def _progressive_lead_time(decisions: pd.DataFrame, *, target_horizon: float) -> pd.DataFrame:
    if decisions.empty or "component_id" not in decisions.columns:
        return pd.DataFrame(columns=["component_id", "first_flag_time_h", "lead_time_h", "future_defective"])
    d = decisions.copy()
    d["origin_h"] = pd.to_numeric(d["origin_h"], errors="coerce")
    if "future_defective" in d.columns:
        d["future_defective"] = pd.to_numeric(d["future_defective"], errors="coerce").fillna(0).astype(int)
    else:
        d["future_defective"] = 0
    d["decision"] = d.get("disposition", d.get("decision", "" )).astype(str)
    flags = d[d["decision"].isin({"REVIEW", "REJECT"})].sort_values(["component_id", "origin_h"])
    first = (
        flags.drop_duplicates("component_id", keep="first")[["component_id", "origin_h", "decision", "failure_risk"]]
        .rename(columns={"origin_h": "first_flag_time_h"})
        if not flags.empty
        else pd.DataFrame(columns=["component_id", "first_flag_time_h", "decision", "failure_risk"])
    )
    parts = d[["component_id", "future_defective"]].drop_duplicates("component_id")
    out = parts.merge(first, on="component_id", how="left")
    out["lead_time_h"] = float(target_horizon) - pd.to_numeric(out["first_flag_time_h"], errors="coerce")
    return out


def _get_anomaly_columns(triage: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "absolute_limits": ["absolute_violation", "hard_limit_violation"],
        "robust_PAT": ["robust_pat_score", "population_evidence", "robust_population_score"],
        "isolation_forest": ["isolation_forest_score", "parameter_isolation_evidence"],
        "temporal": ["temporal_score", "temporal_evidence"],
        "multivariate": ["multivariate_score", "multivariate_component_score"],
        "full_ensemble": ["anomaly_score", "calibrated_anomaly_score", "anomaly_risk"],
    }
    resolved: dict[str, str] = {}
    for name, candidates in aliases.items():
        col = _find_col(triage, candidates)
        if col is not None:
            resolved[name] = col
    return resolved


def _absolute_from_triage(triage: pd.DataFrame) -> pd.Series:
    if "absolute_violation" in triage.columns:
        return _binary_labels(triage["absolute_violation"]).astype(bool)
    if "hard_limit_violation" in triage.columns:
        return _binary_labels(triage["hard_limit_violation"]).astype(bool)
    value = pd.to_numeric(triage.get("value"), errors="coerce")
    lower = pd.to_numeric(triage.get("lower_limit"), errors="coerce") if "lower_limit" in triage else pd.Series(np.nan, index=triage.index)
    upper = pd.to_numeric(triage.get("upper_limit"), errors="coerce") if "upper_limit" in triage else pd.Series(np.nan, index=triage.index)
    return ((lower.notna() & (value < lower)) | (upper.notna() & (value > upper))).fillna(False)


def _attach_truth(triage: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    key = _find_col(triage, ["component_id", "part_id"])
    if key is None:
        raise ValueError("V4 triage has no component identifier column.")
    d = triage.copy()
    d["component_id"] = d[key].astype(str)
    return d.merge(labels, on="component_id", how="left", suffixes=("", "_truth"))


def _module_a_table(test_triage: pd.DataFrame, labels: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    d = _attach_truth(test_triage, labels)
    y = pd.to_numeric(d["future_defective"], errors="coerce").fillna(0).astype(int).to_numpy()
    absolute_fail = _absolute_from_triage(d).to_numpy(bool)
    escape = (~absolute_fail) & (y == 1)
    methods = _get_anomaly_columns(d)
    rows: list[dict[str, Any]] = []
    for name, col in methods.items():
        score = pd.to_numeric(d[col], errors="coerce").fillna(0.0).to_numpy(float)
        threshold = float(thresholds.get(name, 0.5 if name == "absolute_limits" else 0.6))
        if name == "absolute_limits":
            score = absolute_fail.astype(float)
        metrics = evaluate_anomaly_scores(y, score, threshold)
        pred = score >= threshold
        metrics.update({
            "method": name,
            "latent_escape_recall": float((escape & pred).sum() / max(int(escape.sum()), 1)),
            "latent_escape_fnr": float((escape & ~pred).sum() / max(int(escape.sum()), 1)),
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def _target_value_table(source: pd.DataFrame, horizon_h: float) -> pd.DataFrame:
    part_col = _canonical_part_column(source)
    if part_col is None or "parameter" not in source.columns or "time_h" not in source.columns or "value" not in source.columns:
        return pd.DataFrame(columns=["component_id", "parameter", "actual_target"])
    d = source.copy()
    d["component_id"] = d[part_col].astype(str)
    d["time_h"] = pd.to_numeric(d["time_h"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["time_h", "value", "parameter"])
    d["distance"] = (d["time_h"] - float(horizon_h)).abs()
    d = d.sort_values(["component_id", "parameter", "distance", "time_h"])
    return d.groupby(["component_id", "parameter"], as_index=False).first()[["component_id", "parameter", "value"]].rename(columns={"value": "actual_target"})


def _prediction_column(triage: pd.DataFrame, horizon_h: float) -> str | None:
    preferred = [
        f"prediction_{int(horizon_h)}h",
        f"prediction_{str(horizon_h).replace('.', '_')}h",
        "prediction_168h",
        "predicted_value_at_horizon",
    ]
    return _find_col(triage, preferred)


def _module_b_table(full_test: pd.DataFrame, early_test: pd.DataFrame, triage: pd.DataFrame, horizon_h: float) -> pd.DataFrame:
    pred_col = _prediction_column(triage, horizon_h)
    if pred_col is None:
        return pd.DataFrame()
    actual = _target_value_table(full_test, horizon_h)
    d = triage.copy()
    key_cols = [c for c in ["component_id", "parameter"] if c in d.columns]
    if len(key_cols) != 2:
        return pd.DataFrame()
    d["component_id"] = d["component_id"].astype(str)
    d["parameter"] = d["parameter"].astype(str)
    m = d.merge(actual, on=key_cols, how="left")
    predictions = pd.to_numeric(m[pred_col], errors="coerce").to_numpy(float)
    lower = pd.to_numeric(m.get("prediction_lower"), errors="coerce").to_numpy(float) if "prediction_lower" in m else np.full(len(m), np.nan)
    upper = pd.to_numeric(m.get("prediction_upper"), errors="coerce").to_numpy(float) if "prediction_upper" in m else np.full(len(m), np.nan)
    y = pd.to_numeric(m["actual_target"], errors="coerce").to_numpy(float)

    # Persistence and local linear baselines use only the early-origin observations.
    early = _drop_evaluation_only_columns(early_test.copy())
    if "component_id" not in early.columns and "part_id" in early.columns:
        early["component_id"] = early["part_id"].astype(str)
    if "component_id" in early.columns:
        early["component_id"] = early["component_id"].astype(str)
    early["time_h"] = pd.to_numeric(early.get("time_h"), errors="coerce")
    early["value"] = pd.to_numeric(early.get("value"), errors="coerce")
    pers = np.full(len(m), np.nan)
    linear = np.full(len(m), np.nan)
    for i, row in m.iterrows():
        g = early.loc[
            early["component_id"].eq(str(row["component_id"]))
            & early.get("parameter", pd.Series(index=early.index, dtype=object)).astype(str).eq(str(row["parameter"]))
        ].dropna(subset=["time_h", "value"]).sort_values("time_h")
        if g.empty:
            continue
        last = float(g.iloc[-1]["value"])
        pers[i] = last
        if len(g) >= 2 and float(g["time_h"].iloc[-1]) != float(g["time_h"].iloc[0]):
            slope = float((g["value"].iloc[-1] - g["value"].iloc[0]) / (g["time_h"].iloc[-1] - g["time_h"].iloc[0]))
            linear[i] = last + slope * (float(horizon_h) - float(g["time_h"].iloc[-1]))
        else:
            linear[i] = last

    rows = []
    for name, pred in [("persistence", pers), ("linear", linear), ("selected", predictions)]:
        ok = np.isfinite(y) & np.isfinite(pred)
        row: dict[str, Any] = {
            "model": name,
            "horizon_h": float(horizon_h),
            "n": int(ok.sum()),
            "mae": float(mean_absolute_error(y[ok], pred[ok])) if ok.any() else None,
            "rmse": float(np.sqrt(mean_squared_error(y[ok], pred[ok]))) if ok.any() else None,
        }
        if name == "selected" and np.isfinite(lower).any() and np.isfinite(upper).any():
            interval_ok = ok & np.isfinite(lower) & np.isfinite(upper)
            row["conformal_coverage"] = float(np.mean((y[interval_ok] >= lower[interval_ok]) & (y[interval_ok] <= upper[interval_ok]))) if interval_ok.any() else None
            row["mean_interval_width"] = float(np.mean(upper[interval_ok] - lower[interval_ok])) if interval_ok.any() else None
        else:
            row["conformal_coverage"] = None
            row["mean_interval_width"] = None

        limit = pd.to_numeric(m.get("upper_limit"), errors="coerce").to_numpy(float) if "upper_limit" in m else np.full(len(m), np.nan)
        limit_ok = ok & np.isfinite(limit)
        actual_cross = y > limit
        forecast_cross = (upper if name == "selected" else pred) > limit
        row["limit_crossing_recall"] = float((actual_cross & forecast_cross & limit_ok).sum() / max(int((actual_cross & limit_ok).sum()), 1)) if limit_ok.any() else None
        rows.append(row)

    return pd.DataFrame(rows)


def calibrate_safety_policy(
    validation_screening: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    fn_cost: float = 100.0,
    fp_cost: float = 1.0,
    max_reject_rate: float = 0.25,
    max_target_fpr: float | None = None,
) -> dict[str, Any]:
    """Validation-only calibration for the V4 failure-risk gate."""
    d = validation_screening.copy()
    y = _binary_labels(d["future_defective"]) if "future_defective" in d.columns else pd.Series(0, index=d.index)
    score_column = _find_col(d, ["failure_risk", "risk_score", "anomaly_score"])
    score = pd.to_numeric(d[score_column], errors="coerce").fillna(0.0).to_numpy(float) if score_column else np.zeros(len(d))
    policy = load_yaml(PROJECT_ROOT / "configs" / "policy.yaml")
    if max_target_fpr is None:
        max_target_fpr = float(policy.get("thresholds", {}).get("max_target_fpr", 0.05))
    result = optimize_threshold(
        y,
        score,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        reject_rate_max=max_reject_rate,
        fpr_max=max_target_fpr,
    )
    payload = {
        "format": "safety_policy_calibration_v4",
        "selection_split": "validation",
        "score_column": score_column or "unavailable",
        "thresholds": {
            "risk_reject_threshold": float(result.threshold),
            "risk_review_threshold": float(min(0.40, result.threshold)),
            "reject_min": float(result.threshold),
            "safe_max": float(min(0.25, result.threshold)),
            "max_target_fpr": float(max_target_fpr),
        },
        "optimization": asdict(result),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def redecide_screening(
    triage: pd.DataFrame,
    calibrated_policy: Mapping[str, Any] | None = None,
    *,
    policy_path: str | Path = PROJECT_ROOT / "configs" / "policy.yaml",
) -> pd.DataFrame:
    """Apply the V4 deterministic policy to an already-computed triage table."""
    d = triage.copy()
    policy = load_yaml(policy_path)
    thresholds = dict(policy.get("thresholds", {}))
    if calibrated_policy:
        thresholds.update(calibrated_policy.get("thresholds", {}))
    risk_reject = float(thresholds.get("risk_reject_threshold", 0.75))
    risk_review = float(thresholds.get("risk_review_threshold", 0.40))
    ood_threshold = float(thresholds.get("ood_novelty_threshold", 0.70))
    uncertainty_threshold = float(thresholds.get("max_uncertainty_threshold", 0.65))
    anomaly_threshold = float(thresholds.get("anomaly_review_threshold", 0.60))
    crossing_confidence = float(thresholds.get("crossing_confidence_threshold", 0.80))

    decisions = []
    reasons = []
    for _, row in d.iterrows():
        value = pd.to_numeric(pd.Series([row.get("value")]), errors="coerce").iloc[0]
        lower = row.get("lower_limit")
        upper = row.get("upper_limit")
        hard = (pd.notna(lower) and pd.notna(value) and float(value) < float(lower)) or (pd.notna(upper) and pd.notna(value) and float(value) > float(upper))
        ood_score = float(pd.to_numeric(pd.Series([row.get("ood_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        ood_status = str(row.get("ood_status", "LOW"))
        risk = float(pd.to_numeric(pd.Series([row.get("failure_risk", row.get("risk_score", 0.0))]), errors="coerce").fillna(0.0).iloc[0])
        anomaly = float(pd.to_numeric(pd.Series([row.get("anomaly_score", row.get("anomaly_risk", 0.0))]), errors="coerce").fillna(0.0).iloc[0])
        confidence = float(pd.to_numeric(pd.Series([row.get("forecast_confidence", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        crossing = bool(row.get("predicted_limit_crossing", False))
        uncertainty = 1.0 - np.clip(confidence, 0.0, 1.0)
        mode = str(row.get("forecast_mode", "INSUFFICIENT"))

        if hard:
            decision, reason = "REJECT", "CURRENT_HARD_LIMIT_EXCEEDED"
        elif ood_status == "SEVERE" or ood_score >= ood_threshold:
            decision, reason = "UNKNOWN", "NOVEL_DOMAIN_OR_FAMILY_INSPECTION"
        elif risk >= risk_reject or (crossing and confidence >= crossing_confidence):
            decision, reason = "REJECT", "CALIBRATED_HIGH_FAILURE_RISK"
        elif anomaly >= anomaly_threshold:
            decision, reason = "REVIEW", "SIGNIFICANT_ANOMALY_OR_RAPID_DEGRADATION"
        elif uncertainty >= uncertainty_threshold and mode in {"COLD_START", "INSUFFICIENT"}:
            decision, reason = ("UNKNOWN", "HIGH_FORECAST_UNCERTAINTY") if mode == "INSUFFICIENT" else ("REVIEW", "HIGH_FORECAST_UNCERTAINTY")
        elif mode == "INSUFFICIENT":
            decision, reason = "UNKNOWN", "INSUFFICIENT_TIME_HISTORY"
        elif risk >= risk_review:
            decision, reason = "REVIEW", "MODERATE_CALIBRATED_FAILURE_RISK"
        else:
            decision, reason = "PASS", "SAFE_IN_DOMAIN_LOW_RISK"
        decisions.append(decision)
        reasons.append(reason)

    d["disposition"] = decisions
    d["decision"] = decisions
    d["policy_reason_code"] = reasons
    d["risk_score"] = pd.to_numeric(d.get("failure_risk", d.get("risk_score", np.nan)), errors="coerce")
    d["anomaly_risk"] = pd.to_numeric(d.get("anomaly_score", d.get("anomaly_risk", np.nan)), errors="coerce")
    pred_col = _prediction_column(d, 168.0)
    if pred_col is not None and "prediction_168h" not in d.columns:
        d["prediction_168h"] = pd.to_numeric(d[pred_col], errors="coerce")
    return d


def run_master_benchmark(
    input_path: str | Path,
    output_dir: str | Path = "reports/benchmark",
    *,
    seed: int = 20260831,
    n_seeds: int = 1,
    target_horizon: float = 168.0,
    target_metric: str = "escape_recall",
    max_reject_rate: float = 0.25,
) -> dict[str, Any]:
    """Leak-safe V4 benchmark over train/validation/test partitions.

    The early screening origin is 24 h so forecast metrics compare predictions
    made from early evidence with the later target measurement.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    full = _load_for_benchmark(input_path)
    if "split" not in full.columns:
        raise ValueError("Benchmark requires a split column containing train, val and test.")
    train = full[full["split"].astype(str).eq("train")].copy()
    validation = full[full["split"].astype(str).eq("val")].copy()
    test = full[full["split"].astype(str).eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation and test partitions must all be non-empty.")

    seeds = [int(seed) + i for i in range(max(1, int(n_seeds)))]
    all_a: list[pd.DataFrame] = []
    all_b: list[pd.DataFrame] = []
    all_progressive: list[pd.DataFrame] = []
    threshold_records: dict[str, Any] = {}

    validation_input = _drop_evaluation_only_columns(validation)
    test_input = _drop_evaluation_only_columns(test)
    labels = _truth_table(full, target_horizon)

    for current_seed in seeds:
        np.random.seed(current_seed)
        seed_dir = outdir / f"seed_{current_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        val_early = validation_input.copy()
        test_early = test_input.copy()
        if "time_h" in val_early.columns:
            val_early = val_early.loc[pd.to_numeric(val_early["time_h"], errors="coerce") <= 24.0].copy()
            test_early = test_early.loc[pd.to_numeric(test_early["time_h"], errors="coerce") <= 24.0].copy()

        val_run = run_screening(
            val_early,
            parameters_path=str(PROJECT_ROOT / "configs" / "parameters.yaml"),
            policy_path=str(PROJECT_ROOT / "configs" / "policy.yaml"),
            target_horizon_h=float(target_horizon),
        )
        test_run = run_screening(
            test_early,
            parameters_path=str(PROJECT_ROOT / "configs" / "parameters.yaml"),
            policy_path=str(PROJECT_ROOT / "configs" / "policy.yaml"),
            target_horizon_h=float(target_horizon),
        )

        val_triage = _attach_truth(val_run.triage, labels)
        test_triage = _attach_truth(test_run.triage, labels)

        thresholds: dict[str, float] = {}
        methods = _get_anomaly_columns(val_triage)
        validation_y = pd.to_numeric(val_triage["future_defective"], errors="coerce").fillna(0).astype(int).to_numpy()
        validation_abs = _absolute_from_triage(val_triage)
        escape_y = ((~validation_abs) & (validation_y == 1)).astype(int)
        calibration_target = escape_y if target_metric == "escape_recall" else validation_y

        for name, column in methods.items():
            scores = validation_abs.astype(float).to_numpy() if name == "absolute_limits" else pd.to_numeric(val_triage[column], errors="coerce").fillna(0.0).to_numpy(float)
            thresholds[name] = 0.5 if name == "absolute_limits" else float(
                optimize_threshold(
                    calibration_target,
                    scores,
                    fn_cost=100.0,
                    fp_cost=1.0,
                    reject_rate_max=max_reject_rate,
                    fpr_max=float(load_yaml(PROJECT_ROOT / "configs" / "policy.yaml").get("thresholds", {}).get("max_target_fpr", 0.05)),
                ).threshold
            )

        threshold_records[str(current_seed)] = thresholds
        a_table = _module_a_table(test_triage, labels, thresholds)
        a_table["seed"] = current_seed
        b_table = _module_b_table(test, test_early, test_run.triage, target_horizon)
        if not b_table.empty:
            b_table["seed"] = current_seed
        all_a.append(a_table)
        all_b.append(b_table)

        progressive = progressive_screen_dataframe(
            test_input,
            seed_dir / "progressive",
            origins=DEFAULT_ORIGINS,
            target_horizon=target_horizon,
            parameters_path=PROJECT_ROOT / "configs" / "parameters.yaml",
            policy_path=PROJECT_ROOT / "configs" / "policy.yaml",
        )
        lead = progressive["lead_time"].copy()
        if not lead.empty:
            lead["seed"] = current_seed
        all_progressive.append(lead)

        test_run.triage.to_csv(seed_dir / "test_triage.csv", index=False)
        val_run.triage.to_csv(seed_dir / "validation_triage.csv", index=False)
        a_table.to_csv(seed_dir / "module_a_comparison.csv", index=False)
        b_table.to_csv(seed_dir / "module_b_comparison.csv", index=False)

    module_a = pd.concat(all_a, ignore_index=True) if all_a else pd.DataFrame()
    module_b = pd.concat(all_b, ignore_index=True) if all_b and any(not x.empty for x in all_b) else pd.DataFrame()
    progressive = pd.concat(all_progressive, ignore_index=True) if all_progressive and any(not x.empty for x in all_progressive) else pd.DataFrame()

    if not module_a.empty:
        module_a.to_csv(outdir / "anomaly_comparison.csv", index=False)
    if not module_b.empty:
        module_b.to_csv(outdir / "forecast_comparison.csv", index=False)
    if not progressive.empty:
        progressive.to_csv(outdir / "progressive_lead_time.csv", index=False)

    summary: dict[str, Any] = {
        "horizon_h": float(target_horizon),
        "seeds": seeds,
        "train_parts": int(_part_series(train).nunique()),
        "validation_parts": int(_part_series(validation).nunique()),
        "test_parts": int(_part_series(test).nunique()),
        "thresholds_validation": threshold_records,
    }

    if not module_a.empty:
        summary["module_a"] = (
            module_a.groupby("method", as_index=False)[
                ["recall", "fnr", "fpr", "pr_auc", "latent_escape_recall", "latent_escape_fnr"]
            ].mean().to_dict("records")
        )

    if not module_b.empty:
        summary["module_b"] = (
            module_b.groupby("model", as_index=False)[
                ["mae", "rmse", "conformal_coverage", "limit_crossing_recall"]
            ].mean(numeric_only=True).to_dict("records")
        )

    if not progressive.empty:
        summary["progressive"] = {
            "rows": int(len(progressive)),
            "defective_parts": int(progressive.get("future_defective", pd.Series(dtype=int)).sum()),
            "median_lead_time_h": float(progressive.loc[progressive["future_defective"].eq(1), "lead_time_h"].median()) if "future_defective" in progressive.columns and progressive.loc[progressive["future_defective"].eq(1), "lead_time_h"].notna().any() else None,
        }

    save_json(summary, outdir / "benchmark_results.json")
    return {
        "summary": summary,
        "module_a": module_a,
        "module_b": module_b,
        "anomaly": module_a,
        "forecast": module_b,
        "progressive": progressive,
        "thresholds_by_seed": threshold_records,
    }



if __name__=="__main__": raise SystemExit(master_cli())
