import { useState } from 'react'
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Award,
  CheckCircle2,
  Cpu,
  FileSearch,
  HelpCircle,
  Layers,
  Shield,
  ShieldAlert,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react'
import { useApi } from '../hooks/useApi'
import type { ComponentIntelligence, ProgressivePoint } from '../types/api'
import { TrajectoryChart } from '../charts/TrajectoryChart'
import { ProgressiveTimeline } from '../components/component/ProgressiveTimeline'
import { num, parseList } from '../utils/format'
import { StatusBadge } from '../components/common/StatusBadge'
import { ComplianceCertificateModal } from '../components/common/ComplianceCertificateModal'

export function ComponentPage({
  partId,
  source = 'demo',
  onBack,
  onWhy,
  onOpenEvidence,
}: {
  partId: string
  source?: string
  onBack: () => void
  onWhy: () => void
  onOpenEvidence: () => void
}) {
  const [certOpen, setCertOpen] = useState(false)
  const resolvedSource = source === 'demo' ? 'demo' : source === 'active' ? 'final_submission' : source || 'final_submission'
  const q = `?source=${encodeURIComponent(resolvedSource)}`
  const { data, error, loading } = useApi<ComponentIntelligence>(`/api/components/${encodeURIComponent(partId)}${q}`)
  const { data: prog } = useApi<{ items: ProgressivePoint[]; available?: boolean }>(
    `/api/components/${encodeURIComponent(partId)}/progressive${q}`
  )

  if (loading) {
    return (
      <div className="page">
        <div className="loadingBand">
          <Cpu size={24} className="cyanText" style={{ margin: '0 auto 12px' }} />
          Retrieving Component Digital Passport & Physics Telemetry…
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="page">
        <div className="error">{error}</div>
      </div>
    )
  }

  if (!data) return null

  const c = data.component
  const f = data.forecast
  const exp = data.explanation
  const facts = parseList(exp.facts)
  const policy = parseList(exp.policy_reasoning)
  const cf = parseList(exp.counterfactuals)

  const limitEntry = Object.entries(data.engineering_limits).find(([k]) => /upper|spec_upper|limit_upper/i.test(k))
  const limit = limitEntry?.[1] ?? null
  const dq = String(data.decision.data_quality?.status || 'PASS')
  const asOf = data.current_measurements.value_asof ?? data.current_measurements.value_24h
  const change =
    Number(data.current_measurements.value_0h) && Number(asOf)
      ? Number(asOf) - Number(data.current_measurements.value_0h)
      : null

  const safetyMargin = data.decision.safety_margin || {}
  const oodExp = data.decision.ood_explanation || {}
  const primaryTrigger = String(data.decision.primary_trigger || 'NOMINAL_SCREENING_PASS')
  const engineeringRec = String(data.decision.engineering_recommendation || 'RELEASE_SAFE')

  const upperLimitNum = safetyMargin.upper_limit != null ? Number(safetyMargin.upper_limit) : (limit != null ? Number(limit) : null)
  const currentValNum = asOf != null ? Number(asOf) : null
  const projectedValNum = f.prediction != null ? Number(f.prediction) : currentValNum

  let gaugeFillPct = 50
  if (upperLimitNum && upperLimitNum > 0 && currentValNum != null) {
    gaugeFillPct = Math.min(100, Math.max(0, (currentValNum / upperLimitNum) * 100))
  }

  return (
    <div className="page">
      {certOpen && (
        <ComplianceCertificateModal
          partId={partId}
          source={source}
          onClose={() => setCertOpen(false)}
        />
      )}
      <div className="pageHead componentPageHead">
        <div>
          <button className="btn ghost" onClick={onBack}>
            <ArrowLeft size={13} /> Back to Screening Overview
          </button>
          <div className="eyebrow" style={{ marginTop: 14 }}>
            AEROSPACE COMPONENT DIGITAL PASSPORT · EEE FLIGHT QUALIFICATION
          </div>
          <div className="componentId" style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ color: 'var(--ink)' }}>{c.part_id}</span>
            <StatusBadge decision={c.decision} large />
          </div>
          <div className="metaRow">
            <span className="badge unknown">FAMILY · {c.component_family || 'UNKNOWN'}</span>
            <span className="badge unknown">TYPE · {c.component_type || 'STANDARD_EEE'}</span>
            <span className="badge unknown">LOT · {c.lot_id || 'UNASSIGNED'}</span>
            <span className={`badge ${c.confidence === 'HIGH' ? 'safe' : c.confidence === 'MODERATE' ? 'review' : 'unknown'}`}>
              CONFIDENCE · {c.confidence || 'MODERATE'}
            </span>
            <span className={`badge ${c.ood_status === 'SEVERE' ? 'reject' : c.ood_status === 'MODERATE' ? 'review' : 'safe'}`}>
              OOD NOVELTY · {c.ood_status}
            </span>
          </div>
        </div>

        <div className="componentActionBlock">
          <div className="smallcaps">Engineering Proof & Audit</div>
          <div className="actions">
            <button
              className="btn ghost"
              onClick={() => setCertOpen(true)}
              title="Export MIL-STD-883 Flight Qualification Traveler Certificate"
              style={{ border: '1px solid var(--line2)' }}
            >
              <Award size={13} color="var(--accent)" /> Audit Certificate
            </button>
            <button className="btn" onClick={onOpenEvidence}>
              <FileSearch size={13} /> 9-Stage Decision Trace
            </button>
            <button className="btn primary" onClick={onWhy}>
              Why {c.decision}?
            </button>
          </div>
        </div>
      </div>

      <div className="passportCard">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--line)', paddingBottom: 12 }}>
          <div>
            <div className="smallcaps" style={{ color: 'var(--accent)' }}>DIGITAL PASSPORT · TELEMETRY IDENTITY</div>
            <h2 style={{ margin: '4px 0 0', fontSize: '18px', color: 'var(--ink)' }}>
              Component Flight Readiness & Degradation State
            </h2>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="smallcaps" style={{ color: 'var(--dim)' }}>RECOMMENDATION:</span>
            <span className="badge large review" style={{ color: engineeringRec.includes('RELEASE') ? 'var(--safe)' : 'var(--review)' }}>
              {engineeringRec}
            </span>
          </div>
        </div>

        <div className="passportGrid">
          <div className="fact">
            <span className="smallcaps">Monitored Parameter</span>
            <strong className="mono" style={{ color: 'var(--accent)' }}>
              {c.parameter || 'critical_parameter'} {c.unit ? `(${c.unit})` : ''}
            </strong>
          </div>

          <div className="fact">
            <span className="smallcaps">0h Baseline Value</span>
            <strong className="mono">
              {data.current_measurements.value_0h != null ? `${num(data.current_measurements.value_0h, 3)} ${c.unit || ''}` : '—'}
            </strong>
          </div>

          <div className="fact">
            <span className="smallcaps">As-Of Readpoint Value</span>
            <strong className="mono">
              {asOf != null ? `${num(asOf, 3)} ${c.unit || ''}` : '—'}
              {change != null && (
                <span style={{ fontSize: '9px', marginLeft: 6, color: change >= 0 ? 'var(--review)' : 'var(--safe)' }}>
                  ({change >= 0 ? '+' : ''}{num(change, 3)})
                </span>
              )}
            </strong>
          </div>

          <div className="fact">
            <span className="smallcaps">Primary Policy Trigger</span>
            <strong className="mono" style={{ color: primaryTrigger.includes('HARD') ? 'var(--reject)' : primaryTrigger.includes('OOD') ? 'var(--review)' : 'var(--accent)' }}>
              {primaryTrigger}
            </strong>
          </div>
        </div>

        {/* ============================================================ */}
        {/* QUANTIFIED SAFETY MARGIN (VISUAL GAUGE)                     */}
        {/* ============================================================ */}
        <div className="marginGaugeCard">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <span className="smallcaps" style={{ color: 'var(--accent)' }}>QUANTIFIED ENGINEERING SAFETY MARGIN</span>
              <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: 2 }}>
                Evaluated against authoritative physical design specifications
              </div>
            </div>
            <span className={`badge large ${safetyMargin.status === 'VIOLATED' ? 'reject' : safetyMargin.status === 'NEAR_LIMIT' ? 'review' : 'safe'}`}>
              {safetyMargin.status || 'NOMINAL'}
            </span>
          </div>

          {/* Visual Buffer Bar */}
          <div className="marginBarTrack">
            <div
              className={`marginBarFill ${safetyMargin.status === 'VIOLATED' ? 'violated' : safetyMargin.status === 'NEAR_LIMIT' ? 'near' : ''}`}
              style={{ width: `${Math.min(100, Math.max(5, gaugeFillPct))}%` }}
            />
          </div>

          {/* Buffer Bar Legends */}
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', fontFamily: 'JetBrains Mono', color: 'var(--soft)' }}>
            <span>
              CURRENT: <strong style={{ color: 'var(--ink)' }}>{num(currentValNum, 3)} {c.unit}</strong>
            </span>
            <span>
              SAFETY BUFFER: <strong style={{ color: safetyMargin.status === 'VIOLATED' ? 'var(--reject)' : safetyMargin.status === 'NEAR_LIMIT' ? 'var(--review)' : 'var(--safe)' }}>
                {safetyMargin.relative_margin_pct != null ? `${num(safetyMargin.relative_margin_pct, 1)}%` : '—'}
              </strong>
            </span>
            <span>
              SPEC LIMIT: <strong style={{ color: 'var(--reject)' }}>{num(upperLimitNum, 3)} {c.unit}</strong>
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginTop: 12, borderTop: '1px solid var(--line)', paddingTop: 10 }}>
            <div>
              <span className="smallcaps">Absolute Margin</span>
              <strong className="mono" style={{ display: 'block', marginTop: 4, color: 'var(--ink)' }}>
                {safetyMargin.absolute_margin != null ? `${num(safetyMargin.absolute_margin, 3)} ${c.unit}` : '—'}
              </strong>
            </div>
            <div>
              <span className="smallcaps">168h Projected Peak</span>
              <strong className="mono" style={{ display: 'block', marginTop: 4, color: 'var(--accent)' }}>
                {f.upper != null ? `${num(f.upper, 3)} ${c.unit}` : '—'}
              </strong>
            </div>
            <div>
              <span className="smallcaps">Engineering Evaluation</span>
              <div style={{ fontSize: '10px', color: 'var(--muted)', marginTop: 4 }}>
                {safetyMargin.interpretation || 'Operating within validated aerospace envelope.'}
              </div>
            </div>
          </div>
        </div>

        {/* ============================================================ */}
        {/* OOD DOMAIN NOVELTY DECOMPOSITION                            */}
        {/* ============================================================ */}
        <div style={{ marginTop: 14, borderTop: '1px solid var(--line)', paddingTop: 14 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span className="smallcaps" style={{ color: 'var(--accent2)' }}>
              OOD DOMAIN NOVELTY DECOMPOSITION (OOD ≠ DEFECT)
            </span>
            <span className={`badge ${c.ood_status === 'SEVERE' ? 'reject' : c.ood_status === 'MODERATE' ? 'review' : 'safe'}`}>
              SCORE: {num(c.ood_score, 3)}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
            <div className="fact">
              <span className="smallcaps">Physical / Parameter Novelty</span>
              <strong className="mono">{num(oodExp.physical_novelty, 3)}</strong>
              <div style={{ fontSize: '8px', color: 'var(--dim)', marginTop: 3 }}>Operating parameter deviation</div>
            </div>
            <div className="fact">
              <span className="smallcaps">Contextual / Test Novelty</span>
              <strong className="mono">{num(oodExp.contextual_novelty, 3)}</strong>
              <div style={{ fontSize: '8px', color: 'var(--dim)', marginTop: 3 }}>Temperature & stress conditions</div>
            </div>
            <div className="fact">
              <span className="smallcaps">Population Novelty</span>
              <strong className="mono">{num(oodExp.population_novelty, 3)}</strong>
              <div style={{ fontSize: '8px', color: 'var(--dim)', marginTop: 3 }}>Distribution shift vs historical lots</div>
            </div>
          </div>

          <div style={{ marginTop: 8, fontSize: '9.5px', color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <HelpCircle size={12} color="var(--accent)" />
            <span>
              <b>Core Aerospace Rule:</b> {oodExp.core_rule || 'OOD indicates domain novelty / reference deviation, NOT automatically a defect.'}
            </span>
          </div>
        </div>
      </div>

      {/* ============================================================ */}
      {/* SIGNALS & SENSOR SNAPSHOT                                    */}
      {/* ============================================================ */}
      <div className="signalRibbon">
        <div className="signalStrip">
          <span className="smallcaps">Fused Risk Score</span>
          <strong className="mono" style={{ color: (c.risk_score || 0) >= 0.65 ? 'var(--reject)' : (c.risk_score || 0) >= 0.35 ? 'var(--review)' : 'var(--safe)' }}>
            {num(c.risk_score, 2)}
          </strong>
        </div>

        <div className="signalStrip">
          <span className="smallcaps">Anomaly Risk (Module A)</span>
          <strong className="mono">{num(c.anomaly_risk, 2)}</strong>
        </div>

        <div className="signalStrip">
          <span className="smallcaps">Degradation Risk (Module B)</span>
          <strong className="mono">{num(c.failure_risk, 2)}</strong>
        </div>

        <div className="signalStrip">
          <span className="smallcaps">Conformal Uncertainty</span>
          <strong className="mono">±{num(f.conformal_half_width, 3)} {c.unit}</strong>
        </div>

        <div className="signalStrip">
          <span className="smallcaps">Limit Exceedance Projected</span>
          <strong style={{ color: f.predicted_limit_exceedance ? 'var(--reject)' : 'var(--safe)' }}>
            {f.predicted_limit_exceedance ? 'YES · RISK CROSS' : 'NO · BOUNDARY PRESERVED'}
          </strong>
        </div>
      </div>

      {/* ============================================================ */}
      {/* DEGRADATION TRAJECTORY & FORECAST CHART                      */}
      {/* ============================================================ */}
      <div className="panel chartHero" style={{ marginBottom: 18 }}>
        <div className="panelHead">
          <div>
            <div className="panelTitle">
              <TrendingUp size={14} color="var(--accent)" />
              <span>Physics-of-Failure Degradation Trajectory (0h → 168h)</span>
            </div>
            <div className="smallcaps" style={{ marginTop: 4 }}>
              Observed telemetry readpoints, conformal forecast intervals, and authoritative engineering limits
            </div>
          </div>
          <div className="chartLegend">
            <span><span className="legend" /> Observed Telemetry</span>
            <span><span className="legend forecast" /> Forecast Prediction</span>
            <span><span className="legend bound" /> Conformal Bound</span>
            <span><span className="legend limit" /> Spec Upper Limit</span>
          </div>
        </div>

        <div className="chartPanel">
          <TrajectoryChart data={data} />
        </div>
      </div>

      {/* Progressive Multi-rate Timeline */}
      {prog?.items && prog.items.length > 0 && (
        <div className="panel" style={{ marginBottom: 18 }}>
          <div className="panelHead">
            <div className="panelTitle">
              <Layers size={14} color="var(--accent)" />
              <span>Progressive Readpoint Decision History (Early Exit Tracking)</span>
            </div>
            <span className="smallcaps">{prog.items.length} Multi-Rate Readpoints</span>
          </div>
          <ProgressiveTimeline points={prog.items} />
        </div>
      )}
    </div>
  )
}
