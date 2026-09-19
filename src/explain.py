"""Auditable engineering explanation packets."""
from __future__ import annotations
import pandas as pd

def explain_frame(scored:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for _,r in scored.iterrows():
        d=str(r.get("decision","UNKNOWN")); reasons=[x for x in str(r.get("reason_codes","INSUFFICIENT_DATA")).split(";") if x]
        if d=="SAFE": txt="Within learned operating envelope; no calibrated safety trigger exceeded."
        else:
            bits=[]
            if r.get("anomaly_score",0)>=0.5: bits.append(f"anomaly={float(r.get('anomaly_score',0)):.3f}")
            bits.append(f"risk={float(r.get('failure_risk',0)):.3f}")
            if pd.notna(r.get("conformal_radius", float("nan"))): bits.append(f"forecast_radius={float(r.get('conformal_radius',0)):.4g}")
            if str(r.get("ood_status","LOW"))!="LOW": bits.append(f"OOD={r.get('ood_status')}")
            txt="; ".join(bits) if bits else "Evidence insufficient for SAFE disposition."
        rows.append({"part_id":r.get("part_id"),"decision":d,"reason_codes":";".join(reasons),"engineering_explanation":txt,"confidence":"LOW" if d=="UNKNOWN" else "HIGH" if d=="SAFE" else "MODERATE"})
    return pd.DataFrame(rows)
