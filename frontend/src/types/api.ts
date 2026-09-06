export type Decision = 'SAFE' | 'REVIEW' | 'REJECT' | 'UNKNOWN'
export interface SystemStatus { status: string; pipeline: string; models: string; demo_mode: boolean }
export interface BatchSummary { total_components:number; safe:number; review:number; reject:number; unknown:number; decision_counts:Record<string,number>; future_defects_detected:number|null; report_source:string }
export interface ComponentSummary { part_id:string; lot_id?:string|null; component_family?:string|null; component_type?:string|null; parameter?:string|null; unit?:string|null; decision:Decision; risk_score?:number|null; confidence?:string|null; ood_status:string; ood_score?:number|null; anomaly_risk?:number|null; failure_risk?:number|null; uncertainty_score?:number|null; burnin_hours?:number|null; failure_mode?:string|null }
export interface EvidenceChannel { name:string; score:number|null; level?:string|null; detail?:string|null }
export interface MeasurementPoint { time_h:number; value:number|null }
export interface ForecastPayload { horizon_h:number|null; origin_h:number|null; selected_model:string|null; prediction:number|null; lower:number|null; upper:number|null; interval_width:number|null; conformal_half_width:number|null; predicted_limit_exceedance:boolean; limit_exceedance_probability_proxy:number|null; safety_slope_excess:number|null; model_predictions:Record<string,number|null> }
export interface ComponentIntelligence { component:ComponentSummary; current_measurements:Record<string,unknown>; historical_trajectory:MeasurementPoint[]; forecast:ForecastPayload; engineering_limits:Record<string,number|null>; anomaly_evidence:EvidenceChannel[]; ood:Record<string,any>; decision:Record<string,any>; explanation:Record<string,any>; audit_metadata:Record<string,any> }
export interface ScreeningResponse { manifest:Record<string,any>; summary:BatchSummary; decisions:any[]; explanations:any[]; components:ComponentSummary[]; component_intelligence:ComponentIntelligence[] }
export interface BenchmarkResponse { available:boolean; source?:string; files?:Record<string,any[]>; message?:string }
export interface ProgressivePoint { origin_h:number; decision:Decision; risk_score:number|null }
