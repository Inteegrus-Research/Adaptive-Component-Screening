"""Module A: calibrated heterogeneous anomaly evidence."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json, math
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.covariance import LedoitWolf
from scipy.special import expit

EPS=1e-9
CHANNELS=["robust_population","isolation_forest","temporal","multivariate"]

def _rank01(x: np.ndarray, cap=0.99) -> np.ndarray:
    a=np.asarray(x,float); out=np.full_like(a,np.nan)
    ok=np.isfinite(a)
    if ok.sum()==0: return np.nan_to_num(out,nan=0.0)
    vals=a[ok]; order=np.argsort(vals,kind="mergesort"); ranks=np.empty(len(vals)); ranks[order]=np.arange(1,len(vals)+1)
    q=(ranks-0.5)/len(vals); q=np.clip(q,0.0,cap); out[ok]=q
    return np.nan_to_num(out,nan=0.0)

def _sigmoid_score(z): return expit(np.clip(np.asarray(z,float),-20,20))

def _mahalanobis(X, center, precision):
    d=X-center
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",d,precision,d),0.0))

@dataclass
class AnomalyArtifact:
    imputer: Any
    iforest: Any
    mv_center: np.ndarray
    mv_precision: np.ndarray
    family_mv_profiles: dict[str,dict[str,Any]]
    feature_columns: list[str]
    family_feature_stats: dict[str,dict[str,tuple[float,float]]]
    fusion_model: Any
    family_fusion: dict[str,Any]
    healthy_q: dict[str,float]
    channel_reference_sorted: dict[str,np.ndarray]
    seed: int

class AnomalyEngine:
    def __init__(self, seed:int=20260831, n_estimators:int=300):
        self.seed=seed; self.n_estimators=n_estimators; self.artifact=None

    def _channel_raw(self, parts:pd.DataFrame) -> pd.DataFrame:
        d=parts.copy()
        d["robust_population"]= _sigmoid_score(0.70*d["mean_robust_z"].fillna(0).to_numpy()+0.30*d["max_robust_z"].fillna(0).to_numpy())
        temporal=(0.45*d["max_acceleration_normalized"].fillna(0).to_numpy()+0.20*d["max_slope_normalized"].fillna(0).to_numpy()+0.20*np.minimum(d["max_change_point_score"].fillna(0).to_numpy()/6.0,1.0)+0.15*np.minimum(d["joint_exceedance_count"].fillna(0).to_numpy()/2.0,1.0))
        d["temporal"]=np.clip(temporal,0,1)
        d["absolute_violation"]=self._absolute_violation(d)
        d["_mv_distance"] = np.zeros(len(d))
        d["multivariate"] = 0.0
        return d

    @staticmethod
    def _absolute_violation(d:pd.DataFrame)->np.ndarray:
        out=[]
        for _,r in d.iterrows():
            bad=False
            for c in d.columns:
                if not c.endswith("__current"): continue
                p=c[:-9]; cur=r.get(c,np.nan)
                if not np.isfinite(cur): continue
                # Limits are not stored in part table; use normalized distance features.
                dn=r.get(f"{p}__distance_nearest",np.nan)
                if np.isfinite(dn) and dn<0: bad=True
            out.append(float(bad))
        return np.asarray(out)

    def fit(self, train_parts:pd.DataFrame, validation_parts:pd.DataFrame|None=None, validation_y:np.ndarray|None=None):
        tr=train_parts.copy(); va=validation_parts.copy() if validation_parts is not None else None
        Xtr, cols = self._matrix(tr)
        imp=SimpleImputer(strategy="median",add_indicator=True)
        Xti=imp.fit_transform(Xtr)
        iforest=IsolationForest(n_estimators=self.n_estimators,contamination=0.05,random_state=self.seed,n_jobs=-1)
        iforest.fit(Xti)
        trc=self._channel_components(tr, Xti, iforest, imp, cols, fit_mv=True)
        mv_center=trc["_mv_center"]; mv_precision=trc["_mv_precision"]
        family_mv_profiles={}
        for fam,g in tr.groupby("component_family"):
            fcols=[c for c in cols if c.endswith("__robust_z") and pd.to_numeric(g[c],errors="coerce").notna().sum()>=max(8,int(0.10*len(g)))]
            if len(fcols)>=2:
                Xi_f=imp.transform(g[cols].to_numpy(float))
                idx=[cols.index(c) for c in fcols]
                Xf=Xi_f[:,idx]
                try:
                    lw=LedoitWolf().fit(Xf)
                    family_mv_profiles[str(fam)]={"columns":fcols,"center":lw.location_,"precision":lw.precision_}
                except Exception:
                    pass
        self._fit_family_mv_profiles=family_mv_profiles
        tr_raw=self._raw_channels(tr,Xti,iforest,mv_center,mv_precision,cols)
        channel_refs={c:np.sort(pd.to_numeric(tr_raw[c],errors="coerce").to_numpy(float)[np.isfinite(pd.to_numeric(tr_raw[c],errors="coerce").to_numpy(float))]) for c in CHANNELS}
        tr_scores=self._normalize_against_reference(tr_raw,channel_refs)
        family_stats={}
        fusion=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=self.seed)
        if validation_parts is not None and validation_y is not None and len(np.unique(validation_y))>1:
            Xv=imp.transform(va[cols].to_numpy(float)); vr=self._raw_channels(va,Xv,iforest,mv_center,mv_precision,cols); vs=self._normalize_against_reference(vr,channel_refs)
            fusion.fit(vs[CHANNELS].to_numpy(float),np.asarray(validation_y,int))
            fam_models={}
            for fam,g in va.assign(_y=np.asarray(validation_y)).groupby("component_family"):
                idx=g.index.to_numpy(); yy=g["_y"].to_numpy(int)
                if len(np.unique(yy))<2: continue
                xx=vs.loc[idx,CHANNELS].to_numpy(float)
                if len(yy)>=8:
                    m=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=self.seed); m.fit(xx,yy); fam_models[str(fam)]=m
            raw=fusion.predict_proba(vs[CHANNELS])[:,1]
            healthy=raw[np.asarray(validation_y)==0]
            q={"q50":float(np.quantile(healthy,0.50)) if len(healthy) else 0.1,"q90":float(np.quantile(healthy,0.90)) if len(healthy) else 0.3,"q99":float(np.quantile(healthy,0.99)) if len(healthy) else 0.7}
        else:
            fam_models={}; q={"q50":0.1,"q90":0.3,"q99":0.7}
            fusion.fit(tr_scores[CHANNELS].to_numpy(float),tr["future_defective"].to_numpy(int)) if "future_defective" in tr and tr.future_defective.nunique()>1 else None
        self.artifact=AnomalyArtifact(imp,iforest,mv_center,mv_precision,family_mv_profiles,cols,family_stats,fusion,fam_models,q,channel_refs,self.seed)
        return self

    def _matrix(self,d):
        cols=[]
        for c in d.columns:
            if c in {"part_id","lot_id","component_family","split","defect_state","future_defective","origin_h"}: continue
            if pd.api.types.is_numeric_dtype(d[c]) and np.isfinite(pd.to_numeric(d[c], errors="coerce").to_numpy(float)).sum() >= 2: cols.append(c)
        cols=sorted(cols)
        return d[cols].to_numpy(float), cols

    def _channel_components(self,d,Xi,iforest,imp,cols,fit_mv=False):
        mvcols=[c for c in cols if c.endswith("__current") or c.endswith("__robust_z")]
        Xm=Xi[:,[cols.index(c) for c in mvcols if c in cols]] if mvcols else Xi
        if fit_mv:
            lw=LedoitWolf().fit(Xm)
            center=lw.location_; precision=lw.precision_
        else: center=lw.location_; precision=lw.precision_
        return {"_mv_center":center,"_mv_precision":precision}

    def _raw_channels(self,d:pd.DataFrame,Xi,iforest,mv_center,mv_precision,cols)->pd.DataFrame:
        out=d[[c for c in ["part_id","lot_id","component_family","future_defective"] if c in d]].copy()
        out["robust_population"]=np.clip(
            np.nan_to_num(d.get("max_robust_z",pd.Series(0,index=d.index)).to_numpy(float),nan=0.0)
            +0.25*np.nan_to_num(d.get("mean_robust_z",pd.Series(0,index=d.index)).to_numpy(float),nan=0.0),0,None)
        out["isolation_forest"]=-iforest.score_samples(Xi)
        out["temporal"]=np.clip(
            0.55*np.nan_to_num(d.get("max_acceleration_normalized",pd.Series(0,index=d.index)).to_numpy(float),nan=0.0)
            +0.25*np.nan_to_num(d.get("max_slope_normalized",pd.Series(0,index=d.index)).to_numpy(float),nan=0.0)
            +0.20*np.minimum(np.nan_to_num(d.get("max_change_point_score",pd.Series(0,index=d.index)).to_numpy(float),nan=0.0)/5.0,1.0),0,1)
        # Multivariate evidence is family-aware: each family is compared only in the
        # shared robust-z feature space of its applicable parameters.
        out["multivariate"]=0.0
        family_profiles=getattr(self,"_fit_family_mv_profiles",{})
        if family_profiles:
            for fam,prof in family_profiles.items():
                mask=d.component_family.astype(str).eq(fam).to_numpy() if "component_family" in d else np.zeros(len(d),dtype=bool)
                if not mask.any(): continue
                fcols=prof["columns"]; idx=[cols.index(c) for c in fcols if c in cols]
                if not idx: continue
                Xm=Xi[mask][:,idx]
                center=prof["center"]; precision=prof["precision"]
                try: out.loc[mask,"multivariate"]=_mahalanobis(Xm,center,precision)
                except Exception: pass
        else:
            mv_cols=[c for c in cols if c.endswith("__robust_z") or c.endswith("__current")]
            Xm=Xi[:,[cols.index(c) for c in mv_cols]] if mv_cols else Xi
            try: out["multivariate"]=_mahalanobis(Xm,mv_center,mv_precision)
            except Exception: pass
        return out

    @staticmethod
    def _normalize_against_reference(raw:pd.DataFrame, refs:dict[str,np.ndarray], cap:float=0.99)->pd.DataFrame:
        out=raw.copy()
        for c in CHANNELS:
            vals=np.asarray(refs.get(c,np.array([])),float)
            x=pd.to_numeric(raw[c],errors="coerce").to_numpy(float)
            if vals.size==0 or not np.isfinite(vals).any():
                out[c]=0.0
                continue
            vals=np.sort(vals[np.isfinite(vals)])
            # Mid-rank-like empirical CDF against the fixed TRAIN reference.
            ranks=np.searchsorted(vals,x,side="right").astype(float)
            score=(ranks-0.5)/max(len(vals),1)
            score=np.clip(score,0.0,cap)
            score[~np.isfinite(x)]=0.0
            out[c]=score
        return out

    def _calibrate(self, raw:np.ndarray)->np.ndarray:
        q=self.artifact.healthy_q
        x=np.asarray(raw,float); xp=np.array([q["q50"],q["q90"],q["q99"]],float)
        if xp[2] <= xp[0]+1e-9: return np.clip(x,0,1)
        xp[1]=max(xp[1],xp[0]+1e-9); xp[2]=max(xp[2],xp[1]+1e-9)
        return np.clip(np.interp(x,xp,[0.0,0.5,1.0]),0,1)

    def score(self,parts:pd.DataFrame)->pd.DataFrame:
        if self.artifact is None: raise RuntimeError("AnomalyEngine not fitted")
        a=self.artifact; X=parts[a.feature_columns].to_numpy(float); Xi=a.imputer.transform(X)
        self._fit_family_mv_profiles=a.family_mv_profiles
        raw_scores=self._raw_channels(parts,Xi,a.iforest,a.mv_center,a.mv_precision,a.feature_columns)
        scores=self._normalize_against_reference(raw_scores,a.channel_reference_sorted)
        chans=scores[CHANNELS].to_numpy(float)
        global_prob=a.fusion_model.predict_proba(chans)[:,1] if hasattr(a.fusion_model,"predict_proba") and getattr(a.fusion_model,"classes_",None) is not None else np.mean(chans,axis=1)
        probs=global_prob.copy()
        for fam,m in a.family_fusion.items():
            mask=parts.component_family.astype(str).eq(fam).to_numpy()
            if mask.any(): probs[mask]=m.predict_proba(chans[mask])[:,1]
        scores["fusion_raw"]=np.clip(probs,0,1); scores["anomaly_score"]=self._calibrate(probs)
        scores["anomaly_support_count"]=(chans>=0.65).sum(axis=1)
        hard=[]
        for _,r in parts.iterrows():
            bad=False
            for c in parts.columns:
                if c.endswith("__distance_lower") or c.endswith("__distance_upper"):
                    if pd.notna(r.get(c)) and float(r.get(c)) < 0: bad=True
            # distance_nearest is normalized absolute distance and therefore is not itself a violation.
            hard.append(float(bad))
        scores["absolute_violation"]=np.asarray(hard,float)
        return pd.concat([parts.reset_index(drop=True),scores.drop(columns=[c for c in ["future_defective","part_id","lot_id","component_family"] if c in scores],errors="ignore").reset_index(drop=True)],axis=1)

    def save(self,path:Path): path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(self.artifact,path)
    @classmethod
    def load(cls,path:Path):
        obj=cls(); obj.artifact=joblib.load(path); return obj
