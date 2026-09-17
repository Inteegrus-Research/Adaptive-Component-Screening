from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["SAFE", "REVIEW", "REJECT", "UNKNOWN"]

class APIModel(BaseModel):
    model_config = ConfigDict(extra="allow")

class SystemStatus(APIModel):
    status: str
    pipeline: str = "READY"
    models: str = "AVAILABLE"
    demo_mode: bool = False
    report_source: str | None = None
    report_dir: str | None = None
    integrity: str = "PASS"
    api_version: str = "3.0.0"

class BatchSummary(APIModel):
    total_components: int
    safe: int
    review: int
    reject: int
    unknown: int
    decision_counts: dict[str, int] = Field(default_factory=dict)
    future_defects_detected: int | None = None
    report_source: str

class ComponentSummary(APIModel):
    part_id: str
    lot_id: str | None = None
    component_family: str | None = None
    component_type: str | None = None
    parameter: str | None = None
    unit: str | None = None
    decision: Decision
    risk_score: float | None = None
    confidence: str | None = None
    ood_status: str = "UNKNOWN"
    ood_score: float | None = None
    anomaly_risk: float | None = None
    failure_risk: float | None = None
    uncertainty_score: float | None = None
    burnin_hours: float | None = None
    failure_mode: str | None = None

class EvidenceChannel(APIModel):
    name: str
    score: float | None = None
    level: str | None = None
    detail: str | None = None

class ForecastPayload(APIModel):
    horizon_h: float | None = None
    origin_h: float | None = None
    selected_model: str | None = None
    prediction: float | None = None
    lower: float | None = None
    upper: float | None = None
    interval_width: float | None = None
    conformal_half_width: float | None = None
    predicted_limit_exceedance: bool = False
    limit_exceedance_probability_proxy: float | None = None
    safety_slope_excess: float | None = None
    model_predictions: dict[str, float | None] = Field(default_factory=dict)

class MeasurementPoint(APIModel):
    time_h: float
    value: float | None = None

class ComponentIntelligence(APIModel):
    component: ComponentSummary
    current_measurements: dict[str, Any] = Field(default_factory=dict)
    historical_trajectory: list[MeasurementPoint] = Field(default_factory=list)
    forecast: ForecastPayload
    engineering_limits: dict[str, float | None] = Field(default_factory=dict)
    anomaly_evidence: list[EvidenceChannel] = Field(default_factory=list)
    ood: dict[str, Any] = Field(default_factory=dict)
    decision: dict[str, Any] = Field(default_factory=dict)
    explanation: dict[str, Any] = Field(default_factory=dict)
    audit_metadata: dict[str, Any] = Field(default_factory=dict)

class ScreenResponse(APIModel):
    run_id: str | None = None
    manifest: dict[str, Any]
    summary: BatchSummary
    decisions: list[dict[str, Any]]
    explanations: list[dict[str, Any]]
    components: list[ComponentSummary]
    component_intelligence: list[ComponentIntelligence] = Field(default_factory=list)

class ContractResponse(APIModel):
    api_version: str
    resources: dict[str, str]
