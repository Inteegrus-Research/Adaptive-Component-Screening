import type { ReactNode } from 'react'
import { ArrowLeft, CheckCircle2, CircleAlert, ClipboardList, FlaskConical, ShieldCheck } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { ComponentIntelligence } from '../types/api'
import { parseList, num } from '../utils/format'
import { StatusBadge } from '../components/common/StatusBadge'

export function EvidencePage({ partId, onBack, onOpenComponent }: { partId: string; onBack: () => void; onOpenComponent: () => void }) {
  const { data, error, loading } = useApi<ComponentIntelligence>(`/api/components/${encodeURIComponent(partId)}`)

  if (loading) return <div className="page"><div className="loadingBand">Loading engineering evidence record…</div></div>
  if (error) return <div className="page"><div className="error">{error}</div></div>
  if (!data) return null

  const c = data.component
  const exp = data.explanation
  const facts = parseList(exp.facts)
  const findings = parseList(exp.model_findings)
  const policy = parseList(exp.policy_reasoning)
  const counterfactuals = parseList(exp.counterfactuals)
  const auditTrace = (exp.audit_trace && typeof exp.audit_trace === 'object') ? exp.audit_trace as Record<string, any> : {}
  const ood = data.ood || {}
  const dq = data.decision.data_quality || {}
  const supportCount = Number(data.decision.supporting_evidence_count || 0)

  return <div className="page">
    <div className="pageHead">
      <div>
        <button className="btn ghost" onClick={onBack}><ArrowLeft size={13} /> Back to component</button>
        <div className="eyebrow" style={{ marginTop: 16 }}>SCREEN 03 · EVIDENCE & EXPLAINABILITY</div>
        <h1>Engineering Evidence Record</h1>
        <p>The interface exposes the persisted analytical trace rather than inventing an AI narrative.</p>
      </div>
      <div className="actions"><StatusBadge decision={c.decision} large /><button className="btn" onClick={onOpenComponent}>Open component intelligence</button></div>
    </div>

    <div className="evidenceHero">
      <div className="evidenceHeroMain"><div className="smallcaps">FINAL DISPOSITION</div><div className="evidenceDecisionLine"><StatusBadge decision={c.decision} large /><span className="mono">{c.part_id}</span></div><p>{exp.summary || 'No persisted narrative summary available.'}</p></div>
      <div className="evidenceHeroMetrics"><Metric label="Risk" value={num(c.risk_score, 3)} /><Metric label="Confidence" value={c.confidence || '—'} /><Metric label="Support channels" value={String(supportCount)} /><Metric label="OOD" value={c.ood_status} /></div>
    </div>

    <div className="sectionTitle"><span>Decision trace</span><span className="smallcaps">MEASUREMENT → EVIDENCE → INFERENCE → FORECAST → POLICY → DISPOSITION</span></div>
    <div className="decisionTrace six"><Trace icon={<FlaskConical size={15} />} title="Measurement" text="What the instrument and canonical representation establish." active /><Trace icon={<ClipboardList size={15} />} title="Evidence" text="Population, temporal, multivariate and absolute-limit channels." active /><Trace icon={<CircleAlert size={15} />} title="Inference" text="Hierarchical analytical findings, ranked rather than opaque." active /><Trace icon={<ActivityGlyph />} title="Forecast" text="Future trajectory with interval-aware uncertainty." active /><Trace icon={<ShieldCheck size={15} />} title="Policy" text="Safety layer determines what may be automated." active /><Trace icon={<CheckCircle2 size={15} />} title="Disposition" text={`Final state: ${c.decision}.`} active last /></div>

    <div className="evidenceThreeCols">
      <EvidenceBlock number="01" title="OBSERVATION" subtitle="What was measured" icon={<FlaskConical size={16} />}>
        <div className="factGrid"><Fact label="Parameter" value={c.parameter || '—'} /><Fact label="Physical quantity" value={String(data.current_measurements.physical_quantity || '—')} /><Fact label="0 h" value={formatWithUnit(data.current_measurements.value_0h, c.unit)} /><Fact label="As-of" value={formatWithUnit(data.current_measurements.value_asof ?? data.current_measurements.value_24h, c.unit)} /></div>
        <ul className="evidenceList">{facts.slice(0, 8).map((x, i) => <li key={i}>{x}</li>)}</ul>
      </EvidenceBlock>

      <EvidenceBlock number="02" title="INFERENCE" subtitle="What the analytical system concluded" icon={<CircleAlert size={16} />}>
        <div className="rankedEvidence">{data.anomaly_evidence.map((item, i) => <div className="rankedRow" key={item.name}><div><strong>{item.name}</strong><span>{item.level || level(item.score)}</span></div><div className="rankTrack"><i style={{ width: `${Math.max(0, Math.min(1, item.score ?? 0)) * 100}%` }} /></div><span className="mono">{item.score == null ? '—' : num(item.score, 2)}</span></div>)}</div>
        <ul className="evidenceList">{findings.slice(0, 8).map((x, i) => <li key={i}>{x}</li>)}</ul>
      </EvidenceBlock>

      <EvidenceBlock number="03" title="DECISION" subtitle="Why policy acted" icon={<ShieldCheck size={16} />}>
        <div className="policyStatement"><StatusBadge decision={c.decision} large /><p>{policy[0] || exp.why_this_decision || 'The persisted policy trace contains no narrative justification.'}</p></div>
        <div className="decisionFacts"><Fact label="Hard limit" value={data.decision.hard_limit_violation ? 'VIOLATED' : 'NOT VIOLATED'} /><Fact label="Near limit" value={data.decision.near_limit ? 'YES' : 'NO'} /><Fact label="Data quality" value={String(dq.status || '—')} /><Fact label="OOD state" value={String(ood.status || c.ood_status || '—')} /></div>
        <div className="auditMini"><div className="smallcaps">Audit trace present</div><span className={auditTrace && Object.keys(auditTrace).length ? 'safeText' : 'muted'}>{auditTrace && Object.keys(auditTrace).length ? 'YES · persisted' : 'NOT AVAILABLE'}</span></div>
      </EvidenceBlock>
    </div>

    <div className="twoCols evidenceBottom">
      <div className="panel"><div className="panelHead"><div><div className="panelTitle">WHY NOT AUTOMATIC REJECT?</div><div className="smallcaps" style={{ marginTop: 4 }}>Safety restraint</div></div><ShieldCheck size={16} color="#c99a55" /></div><div className="panelBody"><WhyNotReject decision={c.decision} hardLimit={!!data.decision.hard_limit_violation} forecastCross={!!data.forecast.predicted_limit_exceedance} uncertainty={data.forecast.conformal_half_width} /><div className="recommendation"><div className="smallcaps">RECOMMENDED NEXT TEST</div><strong>{exp.recommended_next_test || 'Continue standard screening burn-in protocol.'}</strong></div></div></div>
      <div className="panel"><div className="panelHead"><div><div className="panelTitle">WHAT WOULD CHANGE THE DECISION?</div><div className="smallcaps" style={{ marginTop: 4 }}>Counterfactual sensitivity</div></div></div><div className="panelBody"><ul className="evidenceList">{(counterfactuals.length ? counterfactuals : [exp.specific_counterfactual || 'No single evidence change was sufficient to define a safe counterfactual.']).map((x, i) => <li key={i}>{x}</li>)}</ul></div></div>
    </div>

    <div className="footerNote">This record is an engineering interpretation of the backend explanation packet. It is not an LLM-generated causal claim.</div>
  </div>
}

function ActivityGlyph() { return <span className="traceGlyph">↗</span> }
function Metric({ label, value }: { label: string; value: string }) { return <div><span className="smallcaps">{label}</span><strong className="mono">{value}</strong></div> }
function EvidenceBlock({ number, title, subtitle, icon, children }: { number: string; title: string; subtitle: string; icon: ReactNode; children: ReactNode }) { return <div className="evidenceBlock"><div className="evidenceBlockHead"><div className="evidenceNumber">{number}</div><div>{icon}</div><div><h3>{title}</h3><p>{subtitle}</p></div></div>{children}</div> }
function Fact({ label, value }: { label: string; value: string }) { return <div className="fact"><span className="smallcaps">{label}</span><strong>{value}</strong></div> }
function formatWithUnit(v: unknown, unit?: string | null) { const n = v == null ? null : Number(v); return n == null || Number.isNaN(n) ? '—' : `${num(n, 3)} ${unit || ''}` }
function level(score: number | null) { if (score == null) return 'Unavailable'; if (score >= .75) return 'Elevated'; if (score >= .45) return 'Moderate'; return 'Low' }
function WhyNotReject({ decision, hardLimit, forecastCross, uncertainty }: { decision: string; hardLimit: boolean; forecastCross: boolean; uncertainty: number | null }) { const lines = decision === 'REVIEW' ? ['Current evidence does not meet the autonomous rejection boundary.', hardLimit ? 'An authoritative limit violation is present and should be handled as a hard safety signal.' : 'Current engineering limits are not explicitly violated in the persisted result.', forecastCross ? 'The future projection warrants attention but remains subject to forecast uncertainty.' : 'The forecast alone is not treated as an unconditional rejection rule.', uncertainty != null ? `Conformal uncertainty remains material (${num(uncertainty, 3)} in native units).` : 'Prediction uncertainty is available only where persisted by the forecast layer.'] : decision === 'SAFE' ? ['No rejection condition is present in the persisted decision trace.', 'The system does not elevate a SAFE component without independent evidence.'] : decision === 'REJECT' ? ['Autonomous rejection is already justified by the persisted safety decision.', 'Human review remains appropriate for downstream disposition handling.'] : ['Automatic disposition is withheld because the evidence is incomplete or untrusted.']; return <ul className="evidenceList">{lines.map((x, i) => <li key={i}>{x}</li>)}</ul> }


function Trace({ icon, title, text, active, last }: { icon: ReactNode; title: string; text: string; active?: boolean; last?: boolean }) {
  return (
    <div className={`traceItem ${active ? 'active' : ''} ${last ? 'last' : ''}`}>
      <div className="traceIcon">{icon}</div>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  )
}
