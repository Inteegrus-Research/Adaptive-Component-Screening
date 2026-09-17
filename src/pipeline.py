from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.anomaly import (
    AnomalyConfig,
    detect_anomalies,
)
from src.explain import (
    ExplanationTrace,
    explain_dataframe,
    traces_to_dict,
)
from src.features import (
    FeatureConfig,
    add_features,
)
from src.forecast import (
    ForecastConfig,
    forecast_components,
)
from src.ingest import (
    IngestionAudit,
    ingest_dataframe,
    load_parameter_config,
)
from src.safety import (
    apply_safety_policy,
    load_policy,
)


@dataclass(frozen=True)
class CapabilityManifest:
    ingestion_mode: str

    module_a_mode: str
    module_b_mode: str

    ood_status: str
    risk_mode: str

    rows_ingested: int
    components_processed: int
    parameters_processed: int

    unknown_parameters: int
    irregular_timestamp_groups: int


@dataclass
class ScreeningResult:
    canonical_telemetry: pd.DataFrame
    feature_table: pd.DataFrame
    triage: pd.DataFrame
    explanations: list[ExplanationTrace]
    capability_manifest: CapabilityManifest
    ingestion_audit: IngestionAudit

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonicalized_telemetry":
                self.canonical_telemetry.to_dict(
                    orient="records"
                ),
            "feature_table":
                self.feature_table.to_dict(
                    orient="records"
                ),
            "triage":
                self.triage.to_dict(
                    orient="records"
                ),
            "explanations":
                traces_to_dict(
                    self.explanations
                ),
            "capability_manifest":
                asdict(
                    self.capability_manifest
                ),
            "ingestion_audit":
                self.ingestion_audit.to_dict(),
        }


class ACSPipeline:
    """Pure ACS orchestration.

    The pipeline deliberately contains no scientific decision logic.
    It only wires the six architectural questions together.
    """

    def __init__(
        self,
        parameters_path: str | Path = (
            "configs/parameters.yaml"
        ),
        policy_path: str | Path = (
            "configs/policy.yaml"
        ),
    ) -> None:
        self.parameters_path = Path(
            parameters_path
        )

        self.policy_path = Path(
            policy_path
        )

        self.parameter_config = (
            load_parameter_config(
                self.parameters_path
            )
        )

        self.policy = load_policy(
            self.policy_path
        )

    def run(
        self,
        data: pd.DataFrame,
        target_horizon_h: float | None = None,
    ) -> ScreeningResult:
        canonical, audit = ingest_dataframe(
            data,
            parameters_path=self.parameters_path,
        )

        features = add_features(
            canonical,
            self.parameter_config,
            self._feature_config(),
        )

        anomaly = detect_anomalies(
            features,
            self._anomaly_config(),
        )

        forecast = forecast_components(
            anomaly,
            canonical,
            self.parameter_config,
            self._forecast_config(),
            target_horizon_h=target_horizon_h,
        )

        if forecast.empty:
            triage = anomaly.copy()
        else:
            triage = anomaly.merge(
                forecast,
                on=[
                    "component_id",
                    "parameter",
                ],
                how="left",
                suffixes=(
                    "",
                    "_forecast",
                ),
            )

        triage = (
            self._latest_readpoint(
                triage
            )
        )

        triage = apply_safety_policy(
            triage,
            self.policy,
        )

        explanations = explain_dataframe(
            triage
        )

        manifest = (
            self._build_manifest(
                triage,
                audit,
            )
        )

        return ScreeningResult(
            canonical_telemetry=canonical,
            feature_table=features,
            triage=triage,
            explanations=explanations,
            capability_manifest=manifest,
            ingestion_audit=audit,
        )

    def run_csv(
        self,
        path: str | Path,
        target_horizon_h: float | None = None,
    ) -> ScreeningResult:
        frame = pd.read_csv(path)

        return self.run(
            frame,
            target_horizon_h=target_horizon_h,
        )

    @staticmethod
    def _latest_readpoint(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        result = frame.copy()

        if "time_h" in result.columns:
            result = result.sort_values(
                [
                    "component_id",
                    "parameter",
                    "time_h",
                ],
                kind="stable",
            )

        result = (
            result
            .groupby(
                [
                    "component_id",
                    "parameter",
                ],
                sort=False,
                as_index=False,
            )
            .tail(1)
        )

        return result.reset_index(
            drop=True
        )

    def _feature_config(
        self,
    ) -> FeatureConfig:
        config = self.policy.get(
            "features",
            {},
        )

        return FeatureConfig(
            min_reference_group_size=int(
                config.get(
                    "min_reference_group_size",
                    8,
                )
            ),
            mad_epsilon=float(
                config.get(
                    "mad_epsilon",
                    1e-9,
                )
            ),
            ewma_alpha=float(
                config.get(
                    "ewma_alpha",
                    0.25,
                )
            ),
            cusum_drift=float(
                config.get(
                    "cusum_drift",
                    0.50,
                )
            ),
            cusum_threshold=float(
                config.get(
                    "cusum_threshold",
                    2.50,
                )
            ),
        )

    def _anomaly_config(
        self,
    ) -> AnomalyConfig:
        return AnomalyConfig(
            min_iforest_rows=12,
            random_state=42,
            robust_weight=0.35,
            isolation_weight=0.25,
            temporal_weight=0.20,
            multivariate_weight=0.20,
        )

    def _forecast_config(
        self,
    ) -> ForecastConfig:
        config = self.policy.get(
            "forecast",
            {},
        )

        return ForecastConfig(
            horizon_h=float(
                config.get(
                    "horizon_h",
                    168.0,
                )
            ),
            min_component_points=int(
                config.get(
                    "min_component_points",
                    2,
                )
            ),
            min_training_components=int(
                config.get(
                    "min_training_components",
                    4,
                )
            ),
            min_training_rows=int(
                config.get(
                    "min_training_rows",
                    24,
                )
            ),
            conformal_alpha=float(
                config.get(
                    "conformal_alpha",
                    0.05,
                )
            ),
            min_calibration_residuals=int(
                config.get(
                    "min_calibration_residuals",
                    8,
                )
            ),
            cold_start_min_points_for_trend=int(
                config.get(
                    "cold_start_min_points_for_trend",
                    3,
                )
            ),
            uncertainty_growth_per_sqrt_hour=float(
                config.get(
                    "uncertainty_growth_per_sqrt_hour",
                    0.02,
                )
            ),
        )

    @staticmethod
    def _build_manifest(
        triage: pd.DataFrame,
        audit: IngestionAudit,
    ) -> CapabilityManifest:
        if triage.empty:
            return CapabilityManifest(
                ingestion_mode=audit.ingestion_mode,
                module_a_mode="BATCH_FALLBACK",
                module_b_mode="INSUFFICIENT",
                ood_status="UNKNOWN",
                risk_mode="INSUFFICIENT",
                rows_ingested=audit.canonicalized_rows,
                components_processed=0,
                parameters_processed=0,
                unknown_parameters=(
                    audit.unknown_parameters_quarantined
                ),
                irregular_timestamp_groups=(
                    audit.irregular_timestamps_detected
                ),
            )

        module_a_values = set(
            triage[
                "capability_mode"
            ].astype(str)
        )

        if module_a_values == {"FULL"}:
            module_a_mode = "FULL"
        elif "FULL" in module_a_values or "DEGRADED" in module_a_values:
            module_a_mode = "DEGRADED"
        else:
            module_a_mode = "BATCH_FALLBACK"

        module_b_values = set(
            triage[
                "forecast_mode"
            ].fillna("INSUFFICIENT")
            .astype(str)
        )

        if (
            "ML_FORECAST"
            in module_b_values
        ):
            module_b_mode = "ML_FORECAST"
        elif (
            "COLD_START"
            in module_b_values
        ):
            module_b_mode = "COLD_START"
        else:
            module_b_mode = "INSUFFICIENT"

        ood_values = set(
            triage[
                "ood_status"
            ].fillna("UNKNOWN")
            .astype(str)
        )

        if "SEVERE" in ood_values:
            ood_status = "SEVERE"
        elif "MODERATE" in ood_values:
            ood_status = "MODERATE"
        elif ood_values == {"LOW"}:
            ood_status = "LOW"
        else:
            ood_status = "UNKNOWN"

        risk_mode = (
            "EVIDENCE_FUSION"
            if "risk_mode" in triage.columns
            else "INSUFFICIENT"
        )

        return CapabilityManifest(
            ingestion_mode=audit.ingestion_mode,
            module_a_mode=module_a_mode,
            module_b_mode=module_b_mode,
            ood_status=ood_status,
            risk_mode=risk_mode,
            rows_ingested=audit.canonicalized_rows,
            components_processed=int(
                triage[
                    "component_id"
                ].nunique()
            ),
            parameters_processed=int(
                triage[
                    "parameter"
                ].nunique()
            ),
            unknown_parameters=(
                audit.unknown_parameters_quarantined
            ),
            irregular_timestamp_groups=(
                audit.irregular_timestamps_detected
            ),
        )


def run_screening(
    data: pd.DataFrame,
    parameters_path: str | Path = (
        "configs/parameters.yaml"
    ),
    policy_path: str | Path = (
        "configs/policy.yaml"
    ),
    target_horizon_h: float | None = None,
) -> ScreeningResult:
    pipeline = ACSPipeline(
        parameters_path=parameters_path,
        policy_path=policy_path,
    )

    return pipeline.run(
        data,
        target_horizon_h=target_horizon_h,
    )
