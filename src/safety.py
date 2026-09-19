"""ACS V5 trust calibration and validation-only screening policy.

V5 changes:
- OOD thresholds are calibrated on TRAIN + healthy VALIDATION data, using the
  same Ledoit-Wolf Mahalanobis metric for fitting and scoring.
- Distributional OOD for a known family/parameter is a trust signal, not an
  automatic UNKNOWN disposition. UNKNOWN is reserved for schema novelty and
  severe data insufficiency.
- The operating point is selected on validation using a percentile-rank
  evidence score built from the strongest independent channels. TEST rows are
  never used for threshold selection.
- The policy keeps hard limit / high-evidence REJECT, evidence REVIEW, SAFE,
  and UNKNOWN semantics while avoiding the previous OOD-all-to-UNKNOWN collapse.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json, joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

EARLY_SLOPE_FIELD = "max_slope_normalized"

CHANNEL_WEIGHTS = {
    "isolation_forest": 0.45,
    "temporal": 0.25,
    "anomaly_score": 0.15,
    "multivariate": 0.10,
    "failure_risk": 0.05,
}
CHANNELS = list(CHANNEL_WEIGHTS)


def _num(s: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(default).to_numpy(dtype=float)


def _rank_against_reference(x: np.ndarray, ref: np.ndarray, cap: float = 0.995) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    if ref.size == 0:
        return np.zeros(len(x), dtype=float)
    ranks = np.searchsorted(ref, x, side="right").astype(float)
    out = (ranks - 0.5) / max(len(ref), 1)
    out[~np.isfinite(x)] = 0.0
    return np.clip(out, 0.0, cap)


def _truth_healthy(d: pd.DataFrame) -> np.ndarray:
    if "future_defective" in d.columns:
        return pd.to_numeric(d["future_defective"], errors="coerce").fillna(0).astype(int).to_numpy() == 0
    return np.ones(len(d), dtype=bool)


@dataclass
class OODArtifact:
    family_profiles: dict[str, dict[str, Any]]
    feature_columns: list[str]
    q_moderate: float
    q_severe: float
    unknown_family_fraction: float
    calibration_split: str = "train_plus_healthy_val"


class OODProfile:
    def __init__(self, q_moderate: float = 0.95, q_severe: float = 0.99, unknown_family_fraction: float = 0.05):
        self.q_moderate = q_moderate
        self.q_severe = q_severe
        self.unknown_family_fraction = unknown_family_fraction
        self.artifact: OODArtifact | None = None

    @staticmethod
    def _cols(d: pd.DataFrame) -> list[str]:
        cols: list[str] = []
        for c in d.columns:
            if c.endswith("__robust_z"):
                v = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(v).sum() >= 2:
                    cols.append(c)
        return sorted(cols)

    @staticmethod
    def _fit_family(g: pd.DataFrame, cols: list[str], global_med: np.ndarray) -> dict[str, Any]:
        if not cols:
            return {"center": [], "precision": [], "q95": 1.0, "q99": 1.0, "n": int(len(g)), "distance": "sqrt_mahalanobis"}
        X = g[cols].to_numpy(dtype=float)
        X = np.where(np.isfinite(X), X, global_med)
        if len(X) >= 3:
            lw = LedoitWolf().fit(X)
            center = np.asarray(lw.location_, dtype=float)
            precision = np.asarray(lw.precision_, dtype=float)
        else:
            center = np.nanmedian(X, axis=0)
            center = np.where(np.isfinite(center), center, global_med)
            precision = np.eye(len(cols), dtype=float)
        diff = X - center
        d = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, precision, diff), 0.0))
        return {
            "center": center,
            "precision": precision,
            "q95": float(np.quantile(d, 0.95)) if len(d) else 1.0,
            "q99": float(np.quantile(d, 0.99)) if len(d) else 1.0,
            "n": int(len(g)),
            "distance": "sqrt_mahalanobis",
        }

    @staticmethod
    def _distance(frame: pd.DataFrame, cols: list[str], prof: dict[str, Any]) -> np.ndarray:
        if not cols or prof.get("center") is None or prof.get("precision") is None:
            return np.zeros(len(frame), dtype=float)
        X = frame[cols].to_numpy(dtype=float)
        center = np.asarray(prof["center"], dtype=float)
        precision = np.asarray(prof["precision"], dtype=float)
        X = np.where(np.isfinite(X), X, center)
        diff = X - center
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, precision, diff), 0.0))

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None):
        cols = self._cols(train)
        global_med = np.nanmedian(train[cols].to_numpy(dtype=float), axis=0) if cols else np.array([], dtype=float)
        global_med = np.where(np.isfinite(global_med), global_med, 0.0) if len(global_med) else global_med
        profiles: dict[str, dict[str, Any]] = {}
        for fam, g in train.groupby("component_family", dropna=False):
            profiles[str(fam)] = self._fit_family(g, cols, global_med)

        # Re-calibrate distributional thresholds on healthy validation data.
        # This is allowed because the validation split is the operating-point
        # selection set; test data are never consulted.
        if validation is not None and "future_defective" in validation.columns:
            healthy_mask = _truth_healthy(validation)
            for fam, prof in profiles.items():
                gv = validation[validation.component_family.astype(str).eq(fam) & healthy_mask]
                if len(gv) >= 8:
                    d = self._distance(gv, cols, prof)
                    if len(d) >= 8:
                        prof["q95"] = max(float(np.quantile(d, 0.95)), float(prof["q95"]))
                        prof["q99"] = max(float(np.quantile(d, 0.99)), float(prof["q99"]))
                        prof["threshold_source"] = "healthy_validation_max_train"
                        continue
                prof["threshold_source"] = "train"
        self.artifact = OODArtifact(
            family_profiles=profiles,
            feature_columns=cols,
            q_moderate=self.q_moderate,
            q_severe=self.q_severe,
            unknown_family_fraction=self.unknown_family_fraction,
        )
        return self

    def score(self, d: pd.DataFrame) -> pd.DataFrame:
        if self.artifact is None:
            raise RuntimeError("OODProfile not fitted")
        a = self.artifact
        rows: list[dict[str, Any]] = []
        unknown_frac = float(d["component_family"].astype(str).eq("UNKNOWN").mean()) if "component_family" in d.columns and len(d) else 0.0
        novel_workflow = unknown_frac > a.unknown_family_fraction
        for _, r in d.iterrows():
            fam = str(r.get("component_family", "UNKNOWN"))
            prof = a.family_profiles.get(fam)
            if prof is None:
                rows.append({
                    "part_id": str(r.get("part_id", r.get("component_id", ""))),
                    "ood_distance": 10.0,
                    "ood_score": 10.0,
                    "ood_status": "NOVEL_FAMILY_WORKFLOW" if novel_workflow else "SEVERE",
                    "ood_family_novel": 1.0,
                    "ood_parameter_severe": 0.0,
                })
                continue
            vals = np.asarray([r.get(c, np.nan) for c in a.feature_columns], dtype=float)
            center = np.asarray(prof["center"], dtype=float)
            vals = np.where(np.isfinite(vals), vals, center)
            diff = vals - center
            try:
                dist = float(np.sqrt(max(float(diff @ np.asarray(prof["precision"]) @ diff), 0.0)))
            except Exception:
                dist = float(np.linalg.norm(diff))
            q95 = max(float(prof.get("q95", 1.0)), 1e-9)
            q99 = max(float(prof.get("q99", 1.0)), q95)
            status = "LOW" if dist <= q95 else "MODERATE" if dist <= q99 else "SEVERE"
            rows.append({
                "part_id": str(r.get("part_id", r.get("component_id", ""))),
                "ood_distance": dist,
                "ood_score": float(np.clip(dist / q99, 0.0, 10.0)),
                "ood_status": status,
                "ood_family_novel": 0.0,
                "ood_parameter_severe": 0.0,
            })
        return pd.DataFrame(rows)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.artifact, path)

    @classmethod
    def load(cls, path: Path):
        x = cls()
        x.artifact = joblib.load(path)
        return x


@dataclass
class PolicyArtifact:
    review_threshold: float
    risk_reject_threshold: float
    early_slope_reject_threshold: float
    temporal_review_threshold: float
    ood_unknown_thresholds: dict[str, float]
    calibration: dict[str, Any]
    seed: int


class SafetyPolicy:
    """Validation-calibrated safety policy with a non-saturating decision score.

    V6 keeps the existing evidence channels and OOD semantics, but fixes the
    mismatch in V5 where validation operating-point selection ignored the
    learned failure-risk trigger used at inference. The policy now calibrates
    one conservative decision score on VALIDATION only:

        decision_score = max(evidence_score, failure_risk, temporal_rank)

    where temporal_rank is a percentile score against healthy validation
    acceleration values. Thresholds are selected only on validation; TEST is
    never consulted for policy fitting.
    """

    def __init__(self, seed: int = 20260831):
        self.seed = seed
        self.artifact: PolicyArtifact | None = None

    def _evidence(self, d: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        vals: dict[str, np.ndarray] = {}
        for c in CHANNELS:
            vals[c] = np.clip(_num(d.get(c, pd.Series(0.0, index=d.index))), 0.0, 1.0)
        evidence = np.zeros(len(d), dtype=float)
        for c, w in CHANNEL_WEIGHTS.items():
            evidence += float(w) * vals[c]
        return np.clip(evidence, 0.0, 1.0), vals

    @staticmethod
    def _temporal_raw(d: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(
            _num(d.get("max_acceleration_normalized", pd.Series(0.0, index=d.index))),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

    @staticmethod
    def _healthy_temporal_reference(val: pd.DataFrame) -> np.ndarray:
        y = pd.to_numeric(val.get("future_defective", pd.Series(0, index=val.index)), errors="coerce").fillna(0).astype(int).to_numpy()
        raw = SafetyPolicy._temporal_raw(val)
        ref = raw[y == 0]
        ref = ref[np.isfinite(ref)]
        return np.sort(ref.astype(float))

    def _decision_components(self, d: pd.DataFrame, temporal_reference: np.ndarray | None, slope_reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        evidence, channels = self._evidence(d)
        risk = np.clip(_num(d.get("failure_risk", pd.Series(0.0, index=d.index))), 0.0, 1.0)
        raw_temporal = self._temporal_raw(d)
        ref_t = np.asarray(temporal_reference if temporal_reference is not None else [], dtype=float)
        temporal_rank = _rank_against_reference(raw_temporal, ref_t, cap=0.995) if ref_t.size else np.zeros(len(d), dtype=float)
        raw_slope = np.nan_to_num(_num(d.get(EARLY_SLOPE_FIELD, pd.Series(0.0, index=d.index))), nan=0.0, posinf=0.0, neginf=0.0)
        ref_s = np.asarray(slope_reference if slope_reference is not None else [], dtype=float)
        slope_rank = _rank_against_reference(raw_slope, ref_s, cap=0.995) if ref_s.size else np.zeros(len(d), dtype=float)
        score = np.maximum.reduce([evidence, risk, temporal_rank, slope_rank])
        return np.clip(score, 0.0, 1.0), evidence, risk, temporal_rank, slope_rank, channels

    def calibrate(self, val: pd.DataFrame):
        d = val.copy().reset_index(drop=True)
        if "future_defective" not in d.columns:
            raise ValueError("SafetyPolicy calibration requires future_defective")
        y = pd.to_numeric(d["future_defective"], errors="coerce").fillna(0).astype(int).to_numpy()
        healthy, defective = y == 0, y == 1

        temporal_ref = self._healthy_temporal_reference(d)
        slope_raw = np.nan_to_num(_num(d.get(EARLY_SLOPE_FIELD, pd.Series(0.0, index=d.index))), nan=0.0, posinf=0.0, neginf=0.0)
        slope_ref = np.sort(slope_raw[healthy][np.isfinite(slope_raw[healthy])])
        score, evidence, risk, temporal_rank, slope_rank, _ = self._decision_components(d, temporal_ref, slope_ref)

        hard = d.get("hard_violation", pd.Series(False, index=d.index)).astype(bool).to_numpy()
        unc = np.clip(_num(d.get("forecast_uncertainty", pd.Series(1.0, index=d.index)), 1.0), 0, 1)
        crossing = d.get("predicted_crossing", pd.Series(False, index=d.index)).astype(bool).to_numpy() & (unc < 0.40)
        missing = np.clip(_num(d.get("current_missing_fraction", pd.Series(0.0, index=d.index))), 0, 1)
        family_novel = d.get("component_family", pd.Series("UNKNOWN", index=d.index)).astype(str).eq("UNKNOWN").to_numpy()
        parameter_novel = d.get("parameter", pd.Series("KNOWN", index=d.index)).astype(str).eq("UNKNOWN_PARAMETER").to_numpy()
        unknown_base = family_novel | parameter_novel | (missing > 0.66)

        # First select the dedicated early-degradation gate. This is the critical
        # operating point and is searched directly on validation, not test.
        slope_candidates = np.unique(np.concatenate([
            np.linspace(max(0.0001, float(np.nanmin(slope_raw)) if np.isfinite(slope_raw).any() else 0.0001),
                        max(0.005, float(np.nanmax(slope_raw)) if np.isfinite(slope_raw).any() else 0.005), 121),
            np.quantile(slope_raw, np.linspace(0.50, 0.995, 80)) if len(slope_raw) else np.array([]),
        ]))
        slope_rows=[]
        for early_t in slope_candidates:
            reject = hard | crossing | (slope_raw >= float(early_t))
            unknown = unknown_base & ~reject
            action = reject | unknown
            hrate=float(action[healthy].mean()) if healthy.any() else 1.0
            rec=float(action[defective].mean()) if defective.any() else 0.0
            esc=int((defective & ~action).sum())
            slope_rows.append((rec,hrate,esc,float(early_t)))
        feasible_slope=[r for r in slope_rows if r[0]>=0.90 and r[1]<=0.10 and r[2]==0]
        if feasible_slope:
            feasible_slope.sort(key=lambda z:(-z[0],z[1],z[3]))
            _,_,_,early_t=feasible_slope[0]
            mode_prefix="final_early_slope_constrained_validation"
        else:
            slope_rows.sort(key=lambda z:(max(0.90-z[0],0)+max(z[1]-0.10,0)+(1.0 if z[2] else 0.0),z[1],z[3]))
            _,_,_,early_t=slope_rows[0]
            mode_prefix="final_early_slope_fallback"

        base_reject = hard | crossing | (slope_raw >= float(early_t))

        # Once the safety gate is fixed, search only a small validation grid for
        # the remaining review/reject evidence layers. This is intentionally
        # bounded; it avoids the multi-million-cell sweep used in V6.
        review_candidates=np.unique(np.concatenate([np.linspace(0.50,0.995,80), np.quantile(score,np.linspace(0.60,0.995,50))]))
        reject_candidates=np.unique(np.concatenate([np.linspace(0.70,0.999,60), np.quantile(score,np.linspace(0.70,0.999,40))]))
        feasible=[]; candidates=[]
        for reject_t in reject_candidates:
            for review_t in review_candidates:
                if reject_t <= review_t: continue
                reject=base_reject | (score>=reject_t)
                unknown=unknown_base & ~reject
                review=(~reject) & (~unknown) & (score>=review_t)
                action=reject|unknown|review
                hrate=float(action[healthy].mean()) if healthy.any() else 1.0
                rrate=float(review.mean()) if len(review) else 1.0
                rec=float(action[defective].mean()) if defective.any() else 0.0
                esc=int((defective & ~action).sum())
                violation=max(0.0,0.90-rec)+max(0.0,hrate-0.10)+max(0.0,rrate-0.20)+(1.0 if esc else 0.0)
                row=(violation,esc,hrate,rrate,rec,float(reject_t),float(review_t))
                candidates.append(row)
                if hrate<=0.10 and rrate<=0.20 and rec>=0.90 and esc==0:
                    feasible.append(row)
        if feasible:
            feasible.sort(key=lambda z:(-z[4],z[2],z[3],z[5],z[6])); chosen=feasible[0]; mode=mode_prefix+"_plus_constrained_evidence"
        else:
            candidates.sort(key=lambda z:(z[0],z[1],z[2],z[3],-z[4],z[5],z[6])); chosen=candidates[0]; mode=mode_prefix+"_evidence_fallback"

        _,esc,hrate,rrate,rec,reject_t,review_t=chosen
        temporal_q95=float(np.quantile(temporal_ref,0.95)) if temporal_ref.size else 0.0
        self.artifact=PolicyArtifact(
            review_threshold=float(review_t),
            risk_reject_threshold=float(reject_t),
            early_slope_reject_threshold=float(early_t),
            temporal_review_threshold=float(temporal_q95),
            ood_unknown_thresholds={},
            calibration={
                "split":"val", "seed":self.seed, "calibration_mode":mode,
                "healthy_action_rate":float(hrate), "review_rate":float(rrate), "validation_recall":float(rec), "critical_escapes":int(esc),
                "selected_decision_review_threshold":float(review_t), "selected_decision_reject_threshold":float(reject_t),
                "selected_early_slope_reject_threshold":float(early_t),
                "channel_weights":CHANNEL_WEIGHTS,
                "decision_score_definition":"max(evidence_score,failure_risk,temporal_rank_healthy_validation,early_slope_rank_healthy_validation)",
                "early_slope_field":EARLY_SLOPE_FIELD,
                "early_slope_reference_healthy":slope_ref.tolist(),
                "early_slope_reference_healthy_n":int(len(slope_ref)),
                "temporal_reference_healthy":temporal_ref.tolist(),
                "temporal_reference_healthy_n":int(len(temporal_ref)),
                "selection_constraints":{"min_recall":0.90,"max_healthy_action_rate":0.10,"max_review_rate":0.20,"max_critical_escapes":0},
            },
            seed=self.seed,
        )
        return self

    def apply(self, evidence: pd.DataFrame, ood: pd.DataFrame | None) -> pd.DataFrame:
        if self.artifact is None:
            raise RuntimeError("SafetyPolicy not calibrated")
        d = evidence.copy().reset_index(drop=True)
        o = ood.reset_index(drop=True) if ood is not None else pd.DataFrame(index=d.index)
        temporal_ref = np.asarray(self.artifact.calibration.get("temporal_reference_healthy", []), dtype=float)
        slope_ref = np.asarray(self.artifact.calibration.get("early_slope_reference_healthy", []), dtype=float)
        score, ev, risk_arr, temporal_rank, slope_rank, channels = self._decision_components(d, temporal_ref, slope_ref)
        reject_t=float(self.artifact.calibration.get("selected_decision_reject_threshold", self.artifact.risk_reject_threshold))
        review_t=float(self.artifact.calibration.get("selected_decision_review_threshold", self.artifact.review_threshold))
        early_t=float(self.artifact.calibration.get("selected_early_slope_reject_threshold", self.artifact.early_slope_reject_threshold))
        raw_slope=np.nan_to_num(_num(d.get(EARLY_SLOPE_FIELD,pd.Series(0.0,index=d.index))),nan=0.0,posinf=0.0,neginf=0.0)
        decisions=[]; reasons=[]; action_risk=[]
        for i,r in d.iterrows():
            od=o.iloc[i] if len(o)>i else pd.Series(dtype=object)
            risk=float(np.clip(risk_arr[i],0,1)); hard=bool(r.get("hard_violation",False))
            unc=float(np.clip(pd.to_numeric(pd.Series([r.get("forecast_uncertainty",1.0)]),errors="coerce").fillna(1).iloc[0],0,1))
            missing=float(np.clip(pd.to_numeric(pd.Series([r.get("current_missing_fraction",0.0)]),errors="coerce").fillna(0).iloc[0],0,1))
            family_unknown=str(r.get("component_family","UNKNOWN"))=="UNKNOWN"
            parameter_unknown=str(r.get("parameter","KNOWN"))=="UNKNOWN_PARAMETER"
            crossing=bool(r.get("predicted_crossing",False)) and unc<0.40
            status=str(od.get("ood_status","LOW")); s=float(score[i])
            if hard:
                decisions.append("REJECT"); reasons.append("HARD_LIMIT_VIOLATION"); action_risk.append(1.0); continue
            if crossing:
                decisions.append("REJECT"); reasons.append("PREDICTED_LIMIT_CROSSING"); action_risk.append(risk); continue
            if family_unknown or parameter_unknown or missing>0.66:
                decisions.append("UNKNOWN"); reasons.append("SCHEMA_NOVELTY_OR_INSUFFICIENT_DATA"); action_risk.append(0.0); continue
            if raw_slope[i] >= early_t:
                decisions.append("REJECT"); reasons.append("EARLY_DEGRADATION_SLOPE"); action_risk.append(max(risk, float(slope_rank[i]))); continue
            if s >= reject_t:
                reason="HIGH_DECISION_SCORE"
                if status in {"MODERATE","SEVERE"}: reason += "_WITH_OOD"
                decisions.append("REJECT"); reasons.append(reason); action_risk.append(risk); continue
            if s >= review_t:
                bits=["ELEVATED_DECISION_SCORE"]
                if status in {"MODERATE","SEVERE"}: bits.append("OOD_TRUST_SIGNAL")
                if temporal_rank[i]>=review_t: bits.append("TEMPORAL_EVIDENCE")
                if slope_rank[i]>=review_t: bits.append("EARLY_SLOPE_EVIDENCE")
                if risk>=review_t: bits.append("LEARNED_FAILURE_RISK")
                if unc>=0.65: bits.append("FORECAST_UNCERTAINTY")
                decisions.append("REVIEW"); reasons.append(";".join(bits)); action_risk.append(risk); continue
            decisions.append("SAFE"); reasons.append("NORMAL_EVIDENCE"); action_risk.append(risk)
        d["evidence_score"]=ev; d["decision_score"]=score; d["temporal_rank"]=temporal_rank; d["early_slope_rank"]=slope_rank; d["failure_risk"]=risk_arr
        for c,arr in channels.items(): d[f"{c}_contribution"]=arr
        d["early_slope_reject_threshold"]=float(early_t)
        d["decision"]=decisions; d["reason_codes"]=reasons; d["actionable_failure_risk"]=np.asarray(action_risk,float)
        return d

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.artifact), indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        x = cls(); x.artifact = PolicyArtifact(**json.loads(path.read_text())); return x
