"""Module B: one model per (family, parameter), 168 h target, finite-sample conformal intervals."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math, joblib, hashlib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

TARGET=168.0

@dataclass
class ForecastModel:
    family: str
    parameter: str
    selected_model: str
    model: Any
    imputer: Any
    feature_columns: list[str]
    residuals: np.ndarray
    conformal_radius: float
    validation_metrics: dict[str,float]

class ForecastEngine:
    def __init__(self, target_horizon:float=168.0, seed:int=20260831):
        self.target_horizon=target_horizon; self.seed=seed; self.models={}; self.family_radii={}

    @staticmethod
    def _part_series(g:pd.DataFrame, origin:float) -> dict[str,float]:
        z=g[g.time_h<=origin+1e-9].sort_values("time_h")
        if z.empty: return {"current":np.nan,"value_0h":np.nan,"value_24h":np.nan,"slope_0_24":np.nan,"slope":np.nan,"recent_slope":np.nan,"n_points":0}
        y=z.value.to_numpy(float); t=z.time_h.to_numpy(float)
        def sl(tt,yy):
            ok=np.isfinite(tt)&np.isfinite(yy)
            if ok.sum()<2: return np.nan
            return float((yy[ok][-1]-yy[ok][0])/max(tt[ok][-1]-tt[ok][0],1e-9))
        v0=float(y[np.isfinite(y)][0]) if np.isfinite(y).any() else np.nan
        v24=float(z.loc[z.time_h.eq(24),"value"].dropna().iloc[-1]) if z.loc[z.time_h.eq(24),"value"].notna().any() else np.nan
        return {"current":float(y[np.isfinite(y)][-1]) if np.isfinite(y).any() else np.nan,
                "value_0h":v0,"value_24h":v24,"slope_0_24":sl(t[t<=24],y[t<=24]),"slope":sl(t,y),
                "recent_slope":sl(t[-3:],y[-3:]),"n_points":int(np.isfinite(y).sum())}

    def _training_table(self, long_df:pd.DataFrame, family:str, parameter:str, max_origin:float) -> pd.DataFrame:
        g=long_df[long_df.component_family.astype(str).eq(family)&long_df.parameter.astype(str).eq(parameter)].copy()
        rows=[]
        origins=[0.0,12.0,24.0,48.0,72.0,96.0,120.0,144.0]
        origins=[o for o in origins if o<=max_origin and o<self.target_horizon]
        for pid,gp in g.groupby("part_id"):
            target=gp.loc[np.isclose(gp.time_h,self.target_horizon),"value"]
            if target.empty or not np.isfinite(target.iloc[-1]): continue
            lot=str(gp.lot_id.iloc[0]); t0=self._part_series(gp,0.0)
            lotgrp=g[g.lot_id.astype(str).eq(lot)]
            lot0=lotgrp[np.isclose(lotgrp.time_h,0.0)].value.dropna().to_numpy(float)
            lot_med=float(np.median(lot0)) if lot0.size else np.nan; lot_mad=float(np.median(np.abs(lot0-lot_med))) if lot0.size else np.nan
            for o in origins:
                s=self._part_series(gp,o)
                if not np.isfinite(s["current"]): continue
                rows.append({"part_id":str(pid),"family":family,"parameter":parameter,"origin_h":o,
                             **s,"lot_median_0h":lot_med,"lot_mad_0h":lot_mad,
                             "current_time":float(o),"target_time":float(self.target_horizon),
                             "family_code":int(hashlib.sha256(str(family).encode()).hexdigest()[:6],16)%997,"parameter_code":int(hashlib.sha256(str(parameter).encode()).hexdigest()[:6],16)%997,
                             "target":float(target.iloc[-1])})
        return pd.DataFrame(rows)

    @staticmethod
    def _features(t:pd.DataFrame)->list[str]:
        return ["value_0h","value_24h","slope_0_24","lot_median_0h","lot_mad_0h","current","slope","recent_slope","current_time","target_time","n_points","family_code","parameter_code"]

    def fit(self, train_df:pd.DataFrame, val_df:pd.DataFrame, as_of_h:float):
        fampar=sorted(set(map(tuple,train_df[["component_family","parameter"]].drop_duplicates().to_numpy())))
        for family,parameter in fampar:
            tt=self._training_table(train_df,family,parameter,as_of_h)
            if len(tt)<24: continue
            vc=self._training_table(val_df,family,parameter,as_of_h)
            if vc.empty: vc=tt.tail(min(32,len(tt))).copy()
            feats=self._features(tt)
            X=tt[feats].to_numpy(float); y=tt.target.to_numpy(float)
            Xi=SimpleImputer(strategy="median"); X=Xi.fit_transform(X)
            candidates={}
            # Persistence baseline handled separately; three ML candidates.
            ridge=Ridge(alpha=1.0).fit(X,y); candidates["ridge"]=ridge
            hgb=HistGradientBoostingRegressor(max_iter=160,learning_rate=0.045,max_leaf_nodes=15,l2_regularization=0.25,random_state=self.seed).fit(X,y)
            candidates["gradient_boosting"]=hgb
            # Validation table may be sparse; score all available.
            vX=Xi.transform(vc[feats].to_numpy(float)); vy=vc.target.to_numpy(float)
            metrics={}
            for name,m in candidates.items():
                pr=m.predict(vX); ok=np.isfinite(vy)&np.isfinite(pr); metrics[name]=float(mean_absolute_error(vy[ok],pr[ok])) if ok.any() else float("inf")
            if len(vy):
                curv=vc.current.to_numpy(float); pers_ok=np.isfinite(vy)&np.isfinite(curv); pers=float(mean_absolute_error(vy[pers_ok],curv[pers_ok])) if pers_ok.any() else float("inf")
                sl=vc.slope.to_numpy(float); linpred=curv+sl*(self.target_horizon-vc.origin_h.to_numpy(float)); lin_ok=np.isfinite(vy)&np.isfinite(linpred); lin=float(mean_absolute_error(vy[lin_ok],linpred[lin_ok])) if lin_ok.any() else float("inf")
                metrics["persistence"]=pers; metrics["linear"]=lin
            selected=min(metrics,key=metrics.get)
            predv=(vc.current.to_numpy(float) if selected=="persistence" else vc.current.to_numpy(float)+vc.slope.to_numpy(float)*(self.target_horizon-vc.origin_h.to_numpy(float)) if selected=="linear" else candidates[selected].predict(vX))
            ok=np.isfinite(vy)&np.isfinite(predv); residuals=np.abs(vy[ok]-predv[ok]) if ok.any() else np.array([])
            if residuals.size:
                alpha=0.05; k=int(np.ceil((len(residuals)+1)*(1-alpha))); k=min(max(k,1),len(residuals)); q=float(np.sort(residuals)[k-1])
            else: q=float(np.nanstd(y)) if len(y) else 1.0
            model = candidates.get(selected)
            if selected in {"persistence","linear"}: model=selected
            self.models[(str(family),str(parameter))]=ForecastModel(str(family),str(parameter),selected,model,Xi,feats,residuals,q,metrics)
        return self

    def predict(self, long_df:pd.DataFrame, origin_h:float)->pd.DataFrame:
        rows=[]
        for (family,parameter),m in self.models.items():
            g=long_df[long_df.component_family.astype(str).eq(family)&long_df.parameter.astype(str).eq(parameter)]
            for pid,gp in g.groupby("part_id"):
                s=self._part_series(gp,origin_h)
                if not np.isfinite(s["current"]): continue
                lot=str(gp.lot_id.iloc[0]); lotgrp=g[g.lot_id.astype(str).eq(lot)]; lot0=lotgrp[np.isclose(lotgrp.time_h,0.0)].value.dropna().to_numpy(float)
                lot_med=float(np.median(lot0)) if lot0.size else np.nan; lot_mad=float(np.median(np.abs(lot0-lot_med))) if lot0.size else np.nan
                x=pd.DataFrame([{**s,"lot_median_0h":lot_med,"lot_mad_0h":lot_mad,"current_time":origin_h,"target_time":self.target_horizon,"family_code":int(hashlib.sha256(str(family).encode()).hexdigest()[:6],16)%997,"parameter_code":int(hashlib.sha256(str(parameter).encode()).hexdigest()[:6],16)%997}])[m.feature_columns]
                if m.selected_model=="persistence": pred=float(s["current"])
                elif m.selected_model=="linear": pred=float(s["current"] + (s["slope"] if np.isfinite(s["slope"]) else 0.0)*(self.target_horizon-origin_h))
                else:
                    pred=float(m.model.predict(m.imputer.transform(x.to_numpy(float)))[0])
                q=float(m.conformal_radius); lo,hi=pred-q,pred+q
                rows.append({"part_id":str(pid),"lot_id":lot,"component_family":family,"parameter":parameter,"origin_h":float(origin_h),"target_h":float(self.target_horizon),
                             "prediction_168h":pred,"prediction_lower":lo,"prediction_upper":hi,"conformal_radius":q,"selected_forecast_model":m.selected_model,
                             "forecast_mode":"ML" if m.selected_model in {"ridge","gradient_boosting"} else "COLD_START_LINEAR" if m.selected_model=="linear" else "PERSISTENCE",
                             "uncertainty_score":float(np.clip(q/max(abs(pred),1e-9),0,1))})
        out=pd.DataFrame(rows)
        if out.empty:return out
        # Component-level aggregation: worst normalized uncertainty and predicted boundary risk.
        return out

    def save(self,path:Path): path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,path)
    @classmethod
    def load(cls,path:Path): return joblib.load(path)

def aggregate_forecast_to_parts(forecasts:pd.DataFrame, long_df:pd.DataFrame)->pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame(columns=["part_id","forecast_risk","forecast_uncertainty","predicted_crossing","crossing_time_h"])
    rows=[]
    for pid,g in forecasts.groupby("part_id"):
        risk=0.0; cross=False; ct=np.nan; unc=float(np.nanmax(g.uncertainty_score))
        current=long_df[long_df.part_id.astype(str).eq(str(pid))]
        for _,r in g.iterrows():
            target=current[current.parameter.astype(str).eq(str(r.parameter))]
            if target.empty: continue
            lo=float(target.absolute_limit_lower.dropna().iloc[-1]) if target.absolute_limit_lower.notna().any() else -np.inf
            hi=float(target.absolute_limit_upper.dropna().iloc[-1]) if target.absolute_limit_upper.notna().any() else np.inf
            pred=float(r.prediction_168h)
            cur=target[target.time_h.le(float(r.origin_h)+1e-9)].sort_values("time_h").value.dropna()
            cv=float(cur.iloc[-1]) if not cur.empty else pred
            span=max(abs(pred-cv),1e-9)
            violates = pred<lo or pred>hi
            risk=max(risk,float(violates))
            if violates:
                cross=True
                limit=lo if pred<lo else hi
                frac=np.clip(abs(limit-cv)/span,0,1)
                t=float(r.origin_h)+(168.0-float(r.origin_h))*float(frac)
                if not np.isfinite(ct) or t<ct: ct=t
        rows.append({"part_id":str(pid),"forecast_risk":float(risk),"forecast_uncertainty":float(unc),"predicted_crossing":bool(cross),"crossing_time_h":ct})
    return pd.DataFrame(rows)
