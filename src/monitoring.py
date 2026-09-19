"""Lightweight production lifecycle monitoring and champion/challenger manifest."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

@dataclass
class MonitorSnapshot:
    feature_drift: dict[str,float]
    prediction_drift: float
    forecast_residual_drift: float
    ood_rate: float
    calibration_drift: float
    false_alarm_rate: float | None
    missed_failure_rate: float | None
    model_role: str = "CHAMPION"

def _ks(a,b):
    aa=pd.to_numeric(pd.Series(a),errors="coerce").dropna().to_numpy(float)
    bb=pd.to_numeric(pd.Series(b),errors="coerce").dropna().to_numpy(float)
    if len(aa)<5 or len(bb)<5:return np.nan
    return float(ks_2samp(aa,bb).statistic)

def snapshot(reference:pd.DataFrame,current:pd.DataFrame,reference_prediction=None,current_prediction=None,ood_rate:float=0.0,y=None,decision=None)->MonitorSnapshot:
    features={c:_ks(reference[c],current[c]) for c in reference.columns if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])}
    pred_drift=float(_ks(reference_prediction,current_prediction)) if reference_prediction is not None and current_prediction is not None else np.nan
    fa=mf=None
    if y is not None and decision is not None:
        yy=np.asarray(y,int); dec=np.asarray(decision).astype(str)
        fa=float(np.mean((yy==0)&np.isin(dec,["REVIEW","REJECT"])))
        mf=float(np.mean((yy==1)&(dec=="SAFE")))
    return MonitorSnapshot(features,pred_drift,np.nan,float(ood_rate),np.nan,fa,mf)

def save_snapshot(s:MonitorSnapshot,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(asdict(s),indent=2,default=str),encoding="utf-8")
