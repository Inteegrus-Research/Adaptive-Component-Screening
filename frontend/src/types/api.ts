export type Decision = 'SAFE' | 'REVIEW' | 'REJECT' | 'UNKNOWN'

export interface SystemStatus {
  status: string
  pipeline: string
  models: string
  demo_mode: boolean
  report_source?: string | null
  report_dir?: string | null
  integrity: string
  api_version: string
}

export interface BatchSummary {
  total_components: number
  safe: number
  review: number
  reject: number
  unknown: number
  decision_counts: Record<string, number>
  future_defects_detected: number | null
  report_source: string
}

export interface ComponentSummary {
  part_id: string
  lot_id?: string | null
  component_family?: string | null
  component_type?: string | null
  parameter?: string | null
  unit?: string | null
  decision: Decision
  risk_score?: number | null
  confidence?: string | null
  ood_status: string
  ood_score?: number | null
  anomaly_risk?: number | null
  failure_risk?: number | null
  uncertainty_score?: number | null
  burnin_hours?: number | null
  failure_mode?: string | null
}

export interface EvidenceChannel {
  name: string
  score: number | null
  level?: string | null
  detail?: string | null
}

export interface MeasurementPoint {
  time_h: number
  value: number | null
}

export interface ForecastPayload {
  horizon_h: number | null
  origin_h: number | null
  selected_model: string | null
  prediction: number | null
  lower: number | null
  upper: number | null
  interval_width: number | null
  conformal_half_width: number | null
  predicted_limit_exceedance: boolean
  limit_exceedance_probability_proxy: number | null
  safety_slope_excess: number | null
  model_predictions: Record<string, number | null>
}

export interface ComponentIntelligence {
  component: ComponentSummary
  current_measurements: Record<string, unknown>
  historical_trajectory: MeasurementPoint[]
  forecast: ForecastPayload
  engineering_limits: Record<string, number | null>
  anomaly_evidence: EvidenceChannel[]
  ood: Record<string, any>
  decision: Record<string, any>
  explanation: Record<string, any>
  audit_metadata: Record<string, any>
}

export interface ScreeningResponse {
  run_id?: string | null
  manifest: Record<string, any>
  summary: BatchSummary
  decisions: any[]
  explanations: any[]
  components: ComponentSummary[]
  component_intelligence: ComponentIntelligence[]
}

export interface BenchmarkResponse {
  available: boolean
  source?: string | null
  demo_mode?: boolean
  files?: Record<string, any[]>
  artifact_count?: number
  message?: string
}

export interface ProgressivePoint {
  origin_h: number
  decision: Decision
  risk_score: number | null
}

export interface ValidationResponse {
  ok: boolean
  source: string
  demo_mode: boolean
  checks: { artifact: string; ok: boolean; rows?: number | null; columns?: number; required?: string[] }[]
}

export interface RunMetadata {
  available: boolean
  demo_mode: boolean
  source: string
  report_dir: string
  manifest: Record<string, any>
}

export interface DecisionTraceStage {
  stage_number: number
  stage_name: string
  status: string
  summary: string
  details?: Record<string, any>
}

export interface SafetyMargin {
  parameter?: string | null
  unit?: string | null
  upper_limit?: number | null
  observed_value?: number | null
  projected_upper_168h?: number | null
  absolute_margin?: number | null
  relative_margin_pct?: number | null
  status: string
  formula: string
  interpretation: string
}

export interface DigitalPassport {
  part_id: string
  lot_id?: string | null
  component_family?: string | null
  component_type?: string | null
  burnin_hours?: number | null
  readpoints_count: number
  parameter?: string | null
  unit?: string | null
  final_disposition: string
  primary_trigger: string
  engineering_recommendation: string
  risk_score: number
  confidence: string
  evidence_state?: string | null
  safety_margin: SafetyMargin | Record<string, any>
  ood_summary: Record<string, any>
  decision_trace: DecisionTraceStage[]
  timestamp?: string | null
}

export interface ArtifactInventory {
  source: string
  demo_mode: boolean
  artifacts: { path: string; bytes: number; kind: string }[]
}

export interface LotEarlyWarningItem {
  lot_id: string
  title: string
  severity: 'critical' | 'warning' | 'moderate' | 'info'
  ood_rate: number
  mean_ood: number
  escalation_burden: number
  message: string
  recommended_action: string
}

export interface LotSummary {
  lot_id: string
  total_components?: number
  total_parts?: number
  disposition_counts?: Record<string, number>
  safe_count?: number
  review_count?: number
  reject_count?: number
  unknown_count?: number
  safe_rate?: number
  safe_pct?: number
  review_rate?: number
  review_pct?: number
  reject_rate?: number
  reject_pct?: number
  unknown_pct?: number
  mean_risk?: number
  mean_risk_score?: number
  max_risk?: number
  max_risk_score?: number
  mean_ood_score?: number
  elevated_ood_count?: number
  ood_rate_pct?: number
  lot_health?: string
  lot_recommendation?: string
  status?: 'HEALTHY' | 'CAUTION' | 'DRIFT_ALERT' | 'ELEVATED_REVIEW' | string
  severity?: 'critical' | 'warning' | 'moderate' | 'info' | string
  diagnostic?: string
}

export interface LotsResponse {
  lots: LotSummary[]
  total_lots: number
  early_warnings: LotEarlyWarningItem[]
  source: string
}

export interface EngineeringMetricsResponse {
  metrics: Record<string, any>
  summary_counts: Record<string, number>
}

export interface ComplianceCertificate {
  certificate_id: string
  audit_hash: string
  issued_utc: string
  standard: string
  system_name: string
  api_version: string
  component: {
    part_id: string
    lot_id: string
    component_family: string
    component_type: string
    parameter: string
    unit: string
    screening_origin_h: number
    target_horizon_h: number
    observed_value_0h: number | null
    observed_value_asof: number | null
  }
  disposition: {
    decision: Decision
    title: string
    stamp_color: string
    risk_score: number | null
    confidence: string
    ood_status: string
    ood_score: number | null
    acceptance_text: string
  }
  forecast: {
    selected_model: string
    prediction_168h: number | null
    prediction_lower: number | null
    prediction_upper: number | null
    conformal_half_width: number | null
    limit_exceedance_projected: boolean
    engineering_limit_upper: number | null
  }
  evidence_scorecard: {
    channel: string
    score: number | null
    level: string
    detail?: string | null
  }[]
  explanation: {
    facts: any[]
    model_findings: any[]
    policy_reasoning: any[]
    counterfactual: any[]
  }
  verification: {
    policy_engine: string
    data_integrity: string
    operator_signoff: string
    status: string
  }
}

