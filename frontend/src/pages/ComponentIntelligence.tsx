import { ArrowLeft, FileSearch, Shield, TriangleAlert, ArrowRight } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import type { ComponentIntelligence, ProgressivePoint } from '../types/api'
import { TrajectoryChart } from '../charts/TrajectoryChart'
import { EvidenceStack } from '../components/component/EvidenceStack'
import { DecisionChain } from '../components/component/DecisionChain'
import { ProgressiveTimeline } from '../components/component/ProgressiveTimeline'
import { num, parseList } from '../utils/format'
import { StatusBadge } from '../components/common/StatusBadge'

export function ComponentPage({ partId, onBack, onWhy, onOpenEvidence }: { partId: string; onBack: () => void; onWhy: () => void; onOpenEvidence: () => void }) {
  const { data, error, loading } = useApi<ComponentIntelligence>(`/api/components/${encodeURIComponent(partId)}`)
  const { data: prog } = useApi<{ items: ProgressivePoint[]; available?: boolean }>(`/api/components/${encodeURIComponent(partId)}/progressive`)

  if (loading) return <div className="page"><div className="loadingBand">Loading component intelligence packet…</div></div>
  if (error) return <div className="page"><div className="error">{error}</div></div>
  if (!data) return null

  const c = data.component
  const f = data.forecast
  const exp = data.explanation
  const facts = parseList(exp.facts)
  const findings = parseList(exp.model_findings)
  const policy = parseList(exp.policy_reasoning)
  const cf = parseList(exp.counterfactuals)
  const limitEntry = Object.entries(data.engineering_limits).find(([k]) => /upper|spec_upper|limit_upper/i.test(k))
  const limit = limitEntry?.[1] ?? null
  const dq = String(data.decision.data_quality?.status || '—')
  const asOf = data.current_measurements.value_asof ?? data.current_measurements.value_24h
  const change = Number(data.current_measurements.value_0h) && Number(asOf) ? Number(asOf) - Number(data.current_measurements.value_0h) : null

  return <div className="page">
    <div className="pageHead componentPageHead">
      <div>
        <button className="btn ghost" onClick={onBack}><ArrowLeft size={13} /> Back to screening</button>
        <div className="eyebrow" style={{ marginTop: 16 }}>SCREEN 02 · COMPONENT INTELLIGENCE</div>
        <div className="componentId" style={{ marginTop: 7 }}>{c.part_id}</div>
        <div className="metaRow"><StatusBadge decision={c.decision} /><span className="badge unknown">CONFIDENCE · {c.confidence || '—'}</span><span className={`badge ${c.ood_status === 'SEVERE' ? 'reject' : c.ood_status === 'MODERATE' ? 'review' : 'unknown'}`}>OOD · {c.ood_status}</span><span className="badge unknown">{c.component_family || 'FAMILY UNKNOWN'}</span></div>
      </div>
      <div className="componentActionBlock"><div className="smallcaps">engineering evidence</div><div className="actions"><button className="btn" onClick={onOpenEvidence}><FileSearch size={13} /> Evidence record</button><button className="btn primary" onClick={onWhy}>Why {c.decision}?</button></div></div>
    </div>

    <div className="heroStateGrid">
      <div className="heroDisposition"><div className="smallcaps">CURRENT DISPOSITION</div><div className="heroDecision"><StatusBadge decision={c.decision} large /></div><div className="heroCaption">Policy decision from the persisted safety layer.</div></div>
      <Metric label="Risk" value={num(c.risk_score, 2)} sub="policy-facing" />
      <Metric label="Trust" value={c.ood_status} sub={`data quality · ${dq}`} />
      <Metric label="Burn-in" value={f.origin_h == null ? '—' : `${num(f.origin_h, 0)} h`} sub="assessment origin" />
      <Metric label="Forecast" value={f.prediction == null ? '—' : `${num(f.prediction, 3)} ${c.unit || ''}`} sub={`${num(f.horizon_h, 0)} h · ${f.selected_model || 'n/a'}`} />
    </div>

    <div className="signalRibbon">
      <SignalItem label="0 h → as-of" value={change == null ? '—' : `${change >= 0 ? '+' : ''}${num(change, 3)} ${c.unit || ''}`} />
      <SignalItem label="Upper prediction" value={f.upper == null ? '—' : `${num(f.upper, 3)} ${c.unit || ''}`} />
      <SignalItem label="Engineering boundary" value={limit == null ? '—' : `${num(limit, 3)} ${c.unit || ''}`} />
      <SignalItem label="Limit crossing" value={f.predicted_limit_exceedance ? 'PROJECTED' : 'NOT PROJECTED'} danger={f.predicted_limit_exceedance} />
      <SignalItem label="Next action" value={exp.recommended_next_test || 'Standard screening'} />
    </div>

    <div className="panel" style={{ marginBottom: 14 }}>
      <div className="panelHead"><div><div className="panelTitle">Measurement Snapshot</div><div className="smallcaps" style={{ marginTop: 4 }}>Authoritative values available at the screening origin</div></div><span className="mono smallMono">{c.parameter || 'parameter unavailable'} · {c.unit || 'unit unavailable'}</span></div>
      <div className="panelBody"><div className="snapshotGrid"><Snap label="0 h" value={data.current_measurements.value_0h} unit={c.unit} /><Snap label="As-of" value={asOf} unit={c.unit} /><Snap label="168 h forecast" value={f.prediction} unit={c.unit} /><Snap label="Upper interval" value={f.upper} unit={c.unit} /><Snap label="Engineering limit" value={limit} unit={c.unit} /></div></div>
    </div>

    <div className="panel chartHero">
      <div className="panelHead"><div><div className="panelTitle">Degradation Trajectory</div><div className="smallcaps" style={{ marginTop: 5 }}>Measured history · projected trajectory · conformal uncertainty · engineering boundary</div></div><div className="chartLegend"><span><i className="legend observed" /> Observed</span><span><i className="legend forecast" /> Forecast</span><span><i className="legend bound" /> Uncertainty</span><span><i className="legend limit" /> Limit</span></div></div>
      <div className="chartPanel"><TrajectoryChart data={data} /></div>
      <div className="forecastRibbon"><div><span className="smallcaps">FORECAST ORIGIN</span><strong>{num(f.origin_h, 0)} h</strong></div><div><span className="smallcaps">PROJECTED HORIZON</span><strong>{num(f.horizon_h, 0)} h</strong></div><div><span className="smallcaps">SELECTED MODEL</span><strong>{f.selected_model || 'not available'}</strong></div><div><span className="smallcaps">LIMIT CROSSING</span><strong className={f.predicted_limit_exceedance ? 'dangerText' : 'safeText'}>{f.predicted_limit_exceedance ? 'PROJECTED' : 'NOT PROJECTED'}</strong></div></div>
    </div>

    <div className="twoCols" style={{ marginTop: 14 }}>
      <div className="panel"><div className="panelHead"><div><div className="panelTitle">Evidence Stack</div><div className="smallcaps" style={{ marginTop: 4 }}>Independent channels ranked for engineering review</div></div></div><div className="panelBody"><EvidenceStack channels={data.anomaly_evidence} /></div></div>
      <div className="panel"><div className="panelHead"><div><div className="panelTitle">Decision Chain</div><div className="smallcaps" style={{ marginTop: 4 }}>How the disposition is constructed</div></div><Shield size={15} color="#c47a4a" /></div><div className="panelBody"><DecisionChain activeDecision={c.decision} /><div className="callout" style={{ marginTop: 15 }}>{limit != null ? <><TriangleAlert size={13} /> Engineering limit exposed: <span className="mono">{num(limit, 3)} {c.unit || ''}</span>.</> : <>No authoritative upper engineering limit is available in the active persisted result.</>}</div></div></div>
    </div>

    <div className="panel" style={{ marginTop: 14 }}><div className="panelHead"><div><div className="panelTitle">Progressive Burn-in</div><div className="smallcaps" style={{ marginTop: 4 }}>Evidence evolution across available screening origins</div></div><ArrowRight size={15} color="#8f7d72" /></div><div className="panelBody"><ProgressiveTimeline points={prog?.items || []} /></div></div>

    <div className="panel" style={{ marginTop: 14 }}><div className="panelHead"><div><div className="panelTitle">Engineering Explanation</div><div className="smallcaps" style={{ marginTop: 4 }}>Observation → Inference → Decision</div></div><button className="btn" onClick={onWhy}>Open evidence record <FileSearch size={13} /></button></div><div className="panelBody"><div className="explainGrid"><ExplainCard n="01" title="Observation" lines={facts} /><ExplainCard n="02" title="Inference" lines={findings} /><ExplainCard n="03" title="Decision" lines={policy} /></div><div className="counterfactual"><div className="smallcaps">Counterfactual sensitivity</div><div className="callout" style={{ marginTop: 7 }}>{cf[0] || exp.specific_counterfactual || 'No single safe counterfactual was established by the backend explanation layer.'}</div></div></div></div>
  </div>
}

function Metric({ label, value, sub }: { label: string; value: string; sub: string }) { return <div className="signal elevated"><div className="signalLabel">{label}</div><div className="signalValue mono">{value}</div><div className="signalSub">{sub}</div></div> }
function SignalItem({ label, value, danger = false }: { label: string; value: string; danger?: boolean }) { return <div className="signalStrip"><span className="smallcaps">{label}</span><strong className={`mono ${danger ? 'dangerText' : ''}`}>{value}</strong></div> }
function Snap({ label, value, unit }: { label: string; value: unknown; unit?: string | null }) { const n = value == null ? null : Number(value); return <div className="snapshot"><div className="label">{label}</div><div className="value mono">{n == null || Number.isNaN(n) ? '—' : `${num(n, 3)} ${unit || ''}`}</div></div> }
function ExplainCard({ n, title, lines }: { n: string; title: string; lines: string[] }) { return <div className="explainCard"><div className="explainNumber">{n}</div><h3>{title}</h3>{lines.length ? <ul>{lines.slice(0, 5).map((x, i) => <li key={i}>{x}</li>)}</ul> : <p>No persisted narrative evidence available.</p>}</div> }
