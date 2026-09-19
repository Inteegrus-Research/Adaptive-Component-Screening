from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .ingest import IngestionAudit, ingest_dataframe, load_parameter_config
from .features import FeatureConfig, add_features
from .anomaly import AnomalyConfig, detect_anomalies, load_anomaly_calibration
from .forecast import ForecastConfig, forecast_components, load_forecast_bundle, prepare_forecast_training_cache, fit_forecast_models
from .safety import apply_safety_policy, load_ood_profile, load_policy
from .failure_risk import FailureRiskModel
from .explain import explain_dataframe


@dataclass
class CapabilityManifest:
    ingestion_mode: str = "UNKNOWN"
    module_a_mode: str = "UNKNOWN"
    module_b_mode: str = "UNKNOWN"
    ood_status: str = "UNKNOWN"
    risk_mode: str = "UNKNOWN"
    rows_ingested: int = 0
    components_processed: int = 0
    parameters_processed: int = 0
    unknown_parameters: int = 0
    irregular_timestamp_groups: int = 0
    feature_availability_manifest: dict[str, bool] | None = None


@dataclass
class ScreeningResult:
    canonical_telemetry: pd.DataFrame
    feature_table: pd.DataFrame
    triage: pd.DataFrame
    explanations: list[Any]
    capability_manifest: CapabilityManifest
    ingestion_audit: IngestionAudit

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_telemetry": self.canonical_telemetry.to_dict(orient="records"),
            "features": self.feature_table.to_dict(orient="records"),
            "triage": self.triage.to_dict(orient="records"),
            "explanations": [x.to_dict() if hasattr(x, "to_dict") else x for x in self.explanations],
            "capability_manifest": asdict(self.capability_manifest),
            "ingestion_audit": self.ingestion_audit.to_dict(),
        }


class ACSPipeline:
    def __init__(self, *, policy_path: str | Path = "configs/policy.yaml", parameters_path: str | Path = "configs/parameters.yaml", model_dir: str | Path = "models") -> None:
        self.policy_path = Path(policy_path)
        self.parameters_path = Path(parameters_path)
        self.model_dir = Path(model_dir)
        self.parameter_config = load_parameter_config(self.parameters_path)
        self.policy = load_policy(self.policy_path)

    def _feature_config(self) -> FeatureConfig:
        f = self.policy.get("features", {}) or {}
        return FeatureConfig(
            min_reference_group_size=int(f.get("min_reference_group_size", 8)),
            mad_epsilon=float(f.get("mad_epsilon", 1e-9)),
            ewma_alpha=float(f.get("ewma_alpha", 0.25)),
            cusum_drift=float(f.get("cusum_drift", 0.5)),
            cusum_threshold=float(f.get("cusum_threshold", 2.5)),
            change_point_sigma=float(f.get("change_point_sigma", 3.0)),
        )

    def _anomaly_config(self, seed: int = 20260831) -> AnomalyConfig:
        a = self.policy.get("anomaly", {}) or {}
        w = a.get("weights", {}) or {}
        return AnomalyConfig(
            min_iforest_rows=int(a.get("min_iforest_rows", 12)),
            random_state=seed,
            robust_weight=float(w.get("robust", 0.25)),
            isolation_weight=float(w.get("isolation_forest", 0.15)),
            temporal_weight=float(w.get("temporal", 0.35)),
            multivariate_weight=float(w.get("multivariate", 0.25)),
        )

    def _forecast_config(self, seed: int = 20260831) -> ForecastConfig:
        f = self.policy.get("forecast", {}) or {}
        return ForecastConfig(
            horizon_h=float(f.get("horizon_h", 168.0)),
            target_tolerance_h=float(f.get("target_tolerance_h", 2.0)),
            origins_h=tuple(float(x) for x in f.get("origins_h", [24,48,72,96,120,144])),
            min_component_points=int(f.get("min_component_points", 2)),
            min_training_components=int(f.get("min_training_components", 4)),
            min_training_rows=int(f.get("min_training_rows", 12)),
            conformal_alpha=float(f.get("conformal_alpha", 0.05)),
            min_calibration_residuals=int(f.get("min_calibration_residuals", 8)),
            random_state=seed,
        )

    @staticmethod
    def _latest_readpoint(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        work = frame.copy()
        work["time_h"] = pd.to_numeric(work["time_h"], errors="coerce")
        return work.sort_values(["component_id", "parameter", "time_h"]).groupby(["component_id", "parameter"], as_index=False, sort=False).tail(1).reset_index(drop=True)

    @staticmethod
    def _build_manifest(triage: pd.DataFrame, audit: IngestionAudit) -> CapabilityManifest:
        module_a_mode = "FULL" if not triage.empty and "anomaly_score" in triage else "BATCH_FALLBACK"
        module_b_mode = "FULL" if not triage.empty and triage.get("forecast_mode", pd.Series(dtype=str)).astype(str).eq("ML_FORECAST").any() else "COLD_START"
        ood_status = "LOW"
        if "ood_status" in triage.columns and len(triage):
            vals = set(triage["ood_status"].astype(str))
            if "SEVERE" in vals: ood_status = "SEVERE"
            elif "MODERATE" in vals: ood_status = "MODERATE"
        risk_mode = "FULL" if "failure_risk" in triage.columns else "INSUFFICIENT"
        availability = {c: c in triage.columns for c in ["anomaly_score", "acceleration_normalized", "forecast_confidence", "predicted_limit_crossing", "ood_status", "parameter_ood_status"]}
        return CapabilityManifest(
            ingestion_mode=audit.ingestion_mode,
            module_a_mode=module_a_mode,
            module_b_mode=module_b_mode,
            ood_status=ood_status,
            risk_mode=risk_mode,
            rows_ingested=audit.canonicalized_rows,
            components_processed=int(triage["component_id"].nunique()) if not triage.empty else 0,
            parameters_processed=int(triage["parameter"].nunique()) if not triage.empty else 0,
            unknown_parameters=audit.unknown_parameters_quarantined,
            irregular_timestamp_groups=audit.irregular_timestamps_detected,
            feature_availability_manifest=availability,
        )

    def run(
        self,
        data: pd.DataFrame,
        target_horizon_h: float | None = None,
        *,
        reference_data: pd.DataFrame | None = None,
        anomaly_calibration: Mapping[str, Any] | None = None,
        forecast_bundle: Mapping[str, Any] | None = None,
        forecast_training_data: pd.DataFrame | None = None,
        ood_profile: Mapping[str, Any] | None = None,
        failure_risk_model: FailureRiskModel | None = None,
        calibrated_policy: Mapping[str, Any] | None = None,
        seed: int = 20260831,
        already_canonical: bool = False,
        ingestion_audit: IngestionAudit | None = None,
        apply_policy: bool = True,
    ) -> ScreeningResult:
        if already_canonical:
            canonical = data.copy()
            audit = ingestion_audit or IngestionAudit(input_rows=len(canonical), canonicalized_rows=len(canonical), ingestion_mode="LONG")
        else:
            canonical, audit = ingest_dataframe(data, self.parameter_config)
        print(f"[ingest] {len(canonical)} canonical rows, {canonical['family'].nunique() if not canonical.empty else 0} families")
        ref = reference_data if reference_data is not None else (forecast_training_data if forecast_training_data is not None else canonical)
        features = add_features(canonical, self.policy, reference_frame=ref)
        print(f"[features] {features['component_id'].nunique() if not features.empty else 0} parts, {features.groupby(['component_id','parameter']).ngroups if not features.empty else 0} pairs")
        if anomaly_calibration is None:
            p = self.model_dir / "calibration" / "anomaly_calibration.joblib"
            anomaly_calibration = load_anomaly_calibration(p)
        anomaly = detect_anomalies(features, self._anomaly_config(seed), calibration=anomaly_calibration)
        print(f"[anomaly] {anomaly['component_id'].nunique() if not anomaly.empty else 0} parts scored")
        if forecast_bundle is None:
            p = self.model_dir / "forecast" / "model.joblib"
            forecast_bundle = load_forecast_bundle(p)
        forecast = forecast_components(
            anomaly,
            canonical,
            self.parameter_config,
            self._forecast_config(seed),
            target_horizon_h=target_horizon_h,
            model_bundle=forecast_bundle,
            training_data=forecast_training_data,
        )
        print(f"[forecast] {len((forecast_bundle or {}).get('models', {})) if forecast_bundle else 0} cached models available")
        triage = anomaly.merge(forecast, on=["component_id", "parameter"], how="left", suffixes=("", "_forecast")) if not forecast.empty else anomaly.copy()
        triage = self._latest_readpoint(triage)
        # Failure risk is a learned model output. Never synthesize it from fixed
        # hand weights when a model artifact is supplied.
        if failure_risk_model is not None:
            triage["failure_risk"] = failure_risk_model.predict(triage)
        if ood_profile is None:
            p = self.model_dir / "calibration" / "ood_profile.joblib"
            ood_profile = load_ood_profile(p)
        if apply_policy:
            triage = apply_safety_policy(triage, self.policy, calibrated_policy=calibrated_policy, ood_profile=ood_profile)
        explanations = explain_dataframe(triage)
        manifest = self._build_manifest(triage, audit)
        return ScreeningResult(canonical, features, triage, explanations, manifest, audit)

    def run_csv(self, path: str | Path, target_horizon_h: float | None = None, **kwargs: Any) -> ScreeningResult:
        return self.run(pd.read_csv(path), target_horizon_h=target_horizon_h, **kwargs)


def run_screening(data: pd.DataFrame, *, parameters_path: str | Path = "configs/parameters.yaml", policy_path: str | Path = "configs/policy.yaml", model_dir: str | Path = "models", **kwargs: Any) -> ScreeningResult:
    return ACSPipeline(parameters_path=parameters_path, policy_path=policy_path, model_dir=model_dir).run(data, **kwargs)
