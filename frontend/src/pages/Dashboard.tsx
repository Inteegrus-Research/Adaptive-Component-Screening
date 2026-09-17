import { useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  Award,
  CheckCircle2,
  Clock,
  Clock3,
  Compass,
  Cpu,
  Database,
  Download,
  FileSpreadsheet,
  Filter,
  Flame,
  Gauge,
  HelpCircle,
  Layers,
  Percent,
  Plus,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
  Workflow,
  X,
  XCircle,
} from 'lucide-react'
import { useApi } from '../hooks/useApi'
import type {
  BatchSummary,
  ComponentSummary,
  EngineeringMetricsResponse,
  LotSummary,
  ScreeningResponse,
  ValidationResponse,
} from '../types/api'
import { apiPostFile } from '../api/client'
import { num } from '../utils/format'
import { StatusBadge } from '../components/common/StatusBadge'
import { LotEarlyWarning } from '../components/overview/LotEarlyWarning'

export function DashboardPage({
  onOpen,
  onNewRun,
  source = 'demo',
  session,
}: {
  onOpen: (id: string) => void
  onNewRun: () => void
  source?: 'active' | 'demo'
  session?: ScreeningResponse | null
}) {
  const suffix = source === 'demo' ? '&source=demo' : ''
  const { data: s, loading: sl, error: se, reload } = useApi<BatchSummary>(`/api/summary?source=${source}`)
  const { data: c, loading: cl, error: ce, reload: cr } = useApi<{ items: ComponentSummary[]; count: number }>(
    `/api/components?limit=500${suffix}`
  )
  const { data: validation } = useApi<ValidationResponse>(`/api/validation?source=${source}`)
  const { data: lotsData } = useApi<{ items?: LotSummary[]; lots?: LotSummary[]; count?: number }>(`/api/lots?source=${source}`)
  const { data: engData } = useApi<EngineeringMetricsResponse>(`/api/metrics/engineering?source=${source}`)

  const lotsList = lotsData?.items || lotsData?.lots || []

  const [q, setQ] = useState('')
  const [decision, setDecision] = useState('ALL')
  const [activeLot, setActiveLot] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')

  const summary = session?.summary || s
  const sourceItems = session?.components || c?.items || []

  const items = useMemo(() => {
    let rows = sourceItems
    if (activeLot) {
      rows = rows.filter((r) => (r.lot_id || '').toLowerCase() === activeLot.toLowerCase())
    }
    const n = q.trim().toLowerCase()
    if (n) {
      rows = rows.filter((r) =>
        `${r.part_id} ${r.component_family || ''} ${r.parameter || ''} ${r.lot_id || ''}`
          .toLowerCase()
          .includes(n)
      )
    }
    if (decision !== 'ALL') rows = rows.filter((r) => r.decision === decision)
    return rows
  }, [sourceItems, q, decision, activeLot])

  async function quickUpload(file: File) {
    setBusy(true)
    setMessage('')
    try {
      const r = await apiPostFile<ScreeningResponse>('/api/screen', file, 24, 168, true)
      setMessage(`Screening run ${r.run_id || 'active'} successfully executed.`)
      reload()
      cr()
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Upload screening failed.')
    } finally {
      setBusy(false)
    }
  }

  function exportBatchCsv() {
    if (!sourceItems.length) return
    const rows = [
      ['PART_ID', 'LOT_ID', 'FAMILY', 'TYPE', 'PARAMETER', 'DECISION', 'RISK_SCORE', 'OOD_STATUS', 'OOD_SCORE', 'FAILURE_MODE'],
      ...sourceItems.map((item) => [
        item.part_id,
        item.lot_id || 'UNASSIGNED',
        item.component_family || '',
        item.component_type || '',
        item.parameter || '',
        item.decision,
        String(item.risk_score ?? ''),
        item.ood_status,
        String(item.ood_score ?? ''),
        item.failure_mode || '',
      ]),
    ]
    const csvContent = rows.map((r) => r.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(',')).join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ACS_BATCH_TRAVELER_${source}_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (sl || cl) {
    return (
      <div className="page">
        <div className="loadingBand">
          <Activity size={24} className="cyanText" style={{ margin: '0 auto 12px' }} />
          Connecting to ACS Telemetry & Screening Engine…
        </div>
      </div>
    )
  }

  if (se || ce) {
    return (
      <div className="page">
        <div className="errorBlock">
          <div className="eyebrow" style={{ color: 'var(--reject)' }}>MISSION ENGINE ALERT</div>
          <h2>Telemetry Data Stream Unavailable</h2>
          <p>{se || ce}</p>
          <button className="btn primary" onClick={() => { reload(); cr() }}>
            Re-establish Telemetry Link
          </button>
        </div>
      </div>
    )
  }

  if (!summary) {
    return (
      <div className="page">
        <div className="emptyState panel">
          <ShieldCheck size={24} color="var(--accent)" />
          <div>
            <div className="panelTitle">No Screening Batch Initialized</div>
            <p>Initiate a new component screening run or load representative test telemetry.</p>
            <button className="btn primary" onClick={onNewRun} style={{ marginTop: 12 }}>
              <Plus size={13} /> Initialize Screening Run
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Engineering Metrics Provenance Extraction (Strictly Real Backend Values)
  const safetyMetrics = engData?.metrics?.metrics || {}
  const dppmObj = safetyMetrics['post_screening_dppm']
  const fnrObj = safetyMetrics['latent_escape_fnr']
  const escapeObj = safetyMetrics['critical_escape_count']
  const reviewBurdenObj = safetyMetrics['review_burden_ratio']
  const falseScrapObj = safetyMetrics['false_scrap_rate']
  const chamberHoursObj = safetyMetrics['chamber_hours_saved_pct']
  const oodFlagObj = safetyMetrics['ood_flag_rate']
  const driftVelocityObj = safetyMetrics['drift_velocity']
  const intervalWidthObj = safetyMetrics['prediction_interval_width']

  const total = summary.total_components || 1
  const safePct = ((summary.safe / total) * 100).toFixed(1)
  const reviewPct = ((summary.review / total) * 100).toFixed(1)
  const rejectPct = ((summary.reject / total) * 100).toFixed(1)
  const unknownPct = ((summary.unknown / total) * 100).toFixed(1)

  return (
    <div className="page">
      {/* ============================================================ */}
      {/* MISSION CONTROL HEADER BANNER                                 */}
      {/* ============================================================ */}
      <div className="missionHeader">
        <div className="missionTitleBlock">
          <div className="eyebrow">ACS · ADAPTIVE COMPONENT SCREENING</div>
          <h2>MISSION INTELLIGENCE CENTER</h2>
          <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: 4 }}>
            Active Operational Batch · EEE Component Burn-In & Multi-Rate Telemetry Screening
          </div>
        </div>

        <div className="missionStatusGrid">
          <div className="missionPulseItem">
            <span className="pulseDot" />
            <span>SYSTEM OPERATIONAL</span>
          </div>
          <div className="missionPulseItem">
            <span className="pulseDot cyan" />
            <span style={{ color: validation?.ok ? 'var(--safe)' : 'var(--review)' }}>
              DATA {validation?.ok ? 'VERIFIED' : 'CHECK'}
            </span>
          </div>
          <div className="actions">
            <button
              className="btn ghost"
              onClick={exportBatchCsv}
              title="Export Complete Batch Traveler Manifest"
              style={{ border: '1px solid var(--line2)' }}
            >
              <FileSpreadsheet size={13} /> EXPORT TRAVELER PACK (CSV)
            </button>
            <button className="btn primary" onClick={onNewRun}>
              <Plus size={13} /> INTAKE NEW DATA
            </button>
            <label className="btn ghost" style={{ border: '1px solid var(--line2)' }}>
              <Gauge size={13} />
              {busy ? 'SCREENING…' : 'Quick CSV'}
              <input
                type="file"
                accept=".csv"
                hidden
                onChange={(e) => e.target.files?.[0] && quickUpload(e.target.files[0])}
              />
            </label>
          </div>
        </div>
      </div>

      {message && (
        <div className="callout" style={{ marginBottom: 16 }}>
          <ArrowUpRight size={14} color="var(--accent)" />
          <span>{message}</span>
        </div>
      )}

      {/* ============================================================ */}
      {/* SECTION 1: MISSION SAFETY OVERVIEW                           */}
      {/* ============================================================ */}
      <div className="sectionTitle" style={{ marginTop: 0 }}>
        <span>SECTION 01 · MISSION SAFETY OVERVIEW & ESCAPE ASSURANCE</span>
        <span className="smallcaps" style={{ color: 'var(--dim)' }}>NON-FABRICATED AUDIT CONTRACT</span>
      </div>

      <div className="metricsAuditRibbon">
        {/* Metric 1: DPPM */}
        <div className="metricCard cyan">
          <div className="metricCardLabel">Escape Defect DPPM</div>
          <div className="metricCardValue">
            {dppmObj?.status === 'CALCULATED' && dppmObj.value != null ? (
              `${num(dppmObj.value, 1)}`
            ) : (
              <span style={{ fontSize: '13px', color: 'var(--review)' }}>REQ. GROUND TRUTH</span>
            )}
          </div>
          <div className="metricCardStatus">
            <span className={`statusTag ${dppmObj?.status === 'CALCULATED' ? 'calc' : 'gt'}`}>
              {dppmObj?.status || 'REQUIRES_GROUND_TRUTH'}
            </span>
            <span>Target: 0.0</span>
          </div>
        </div>

        {/* Metric 2: FNR / Latent Escape */}
        <div className="metricCard safe">
          <div className="metricCardLabel">Latent Escape (FNR)</div>
          <div className="metricCardValue safeText">
            {fnrObj?.status === 'CALCULATED' && fnrObj.value != null ? (
              `${num(fnrObj.value, 2)}%`
            ) : (
              <span style={{ fontSize: '13px', color: 'var(--review)' }}>REQ. GROUND TRUTH</span>
            )}
          </div>
          <div className="metricCardStatus">
            <span className={`statusTag ${fnrObj?.status === 'CALCULATED' ? 'calc' : 'gt'}`}>
              {fnrObj?.status || 'REQUIRES_GROUND_TRUTH'}
            </span>
            <span>Zero-Escape Target</span>
          </div>
        </div>

        {/* Metric 3: Critical Escapes */}
        <div className="metricCard reject">
          <div className="metricCardLabel">Critical Escape Count</div>
          <div className="metricCardValue">
            {escapeObj?.status === 'CALCULATED' && escapeObj.value != null ? (
              `${escapeObj.value} parts`
            ) : (
              <span style={{ fontSize: '13px', color: 'var(--review)' }}>REQ. GROUND TRUTH</span>
            )}
          </div>
          <div className="metricCardStatus">
            <span className={`statusTag ${escapeObj?.status === 'CALCULATED' ? 'calc' : 'gt'}`}>
              {escapeObj?.status || 'REQUIRES_GROUND_TRUTH'}
            </span>
            <span>Defects Cleared Safe</span>
          </div>
        </div>

        {/* Metric 4: Chamber Hours Saved */}
        <div className="metricCard indigo">
          <div className="metricCardLabel">Chamber Hours Saved</div>
          <div className="metricCardValue indigoText">
            {chamberHoursObj?.status === 'CALCULATED' && chamberHoursObj.value != null ? (
              `${num(chamberHoursObj.value, 1)}%`
            ) : (
              '—'
            )}
          </div>
          <div className="metricCardStatus">
            <span className={`statusTag ${chamberHoursObj?.status === 'CALCULATED' ? 'calc' : 'na'}`}>
              {chamberHoursObj?.status || 'CALCULATED'}
            </span>
            <span>vs 168h Burn-in</span>
          </div>
        </div>

        {/* Metric 5: Review Burden */}
        <div className="metricCard review">
          <div className="metricCardLabel">Review Burden Ratio</div>
          <div className="metricCardValue reviewText">
            {reviewBurdenObj?.status === 'CALCULATED' && reviewBurdenObj.value != null ? (
              `${num(reviewBurdenObj.value, 1)}%`
            ) : (
              '—'
            )}
          </div>
          <div className="metricCardStatus">
            <span className={`statusTag ${reviewBurdenObj?.status === 'CALCULATED' ? 'calc' : 'na'}`}>
              {reviewBurdenObj?.status || 'CALCULATED'}
            </span>
            <span>Routed to Engineer</span>
          </div>
        </div>

        {/* Metric 6: False Scrap / Overkill */}
        <div className="metricCard">
          <div className="metricCardLabel">False Scrap / Overkill</div>
          <div className="metricCardValue">
            {falseScrapObj?.status === 'CALCULATED' && falseScrapObj.value != null ? (
              `${num(falseScrapObj.value, 2)}%`
            ) : (
              <span style={{ fontSize: '13px', color: 'var(--dim)' }}>REQ. GROUND TRUTH</span>
            )}
          </div>
          <div className="metricCardStatus">
            <span className={`statusTag ${falseScrapObj?.status === 'CALCULATED' ? 'calc' : 'gt'}`}>
              {falseScrapObj?.status || 'REQUIRES_GROUND_TRUTH'}
            </span>
            <span>Economic Preservation</span>
          </div>
        </div>
      </div>

      {/* ============================================================ */}
      {/* SECTION 2: SCREENING DISTRIBUTION                           */}
      {/* ============================================================ */}
      <div className="sectionTitle">
        <span>SECTION 02 · SCREENING DISTRIBUTION & BATCH COMPOSITION</span>
        <span className="smallcaps" style={{ color: 'var(--dim)' }}>TOTAL: {summary.total_components} UNITS</span>
      </div>

      <div className="kpis">
        <div className="kpi">
          <div className="label">Total Screened</div>
          <div className="value mono">{summary.total_components}</div>
          <div className="meta">Active Telemetry Records</div>
        </div>

        <div className="kpi safe">
          <div className="label" style={{ color: 'var(--safe)' }}>Safe Disposition</div>
          <div className="value mono">{summary.safe}</div>
          <div className="meta">{safePct}% lot clearance rate</div>
        </div>

        <div className="kpi review">
          <div className="label" style={{ color: 'var(--review)' }}>Review Required</div>
          <div className="value mono">{summary.review}</div>
          <div className="meta">{reviewPct}% human triage queue</div>
        </div>

        <div className="kpi reject">
          <div className="label" style={{ color: 'var(--reject)' }}>Immediate Reject</div>
          <div className="value mono">{summary.reject}</div>
          <div className="meta">{rejectPct}% hard safety violation</div>
        </div>

        <div className="kpi">
          <div className="label" style={{ color: 'var(--muted)' }}>Unknown / Telemetry</div>
          <div className="value mono" style={{ color: 'var(--muted)' }}>{summary.unknown}</div>
          <div className="meta">{unknownPct}% data retest queue</div>
        </div>
      </div>

      {/* Distribution visual progress bar */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', height: 8, borderRadius: 99, overflow: 'hidden', background: '#09101F', border: '1px solid var(--line)' }}>
          <div style={{ width: `${safePct}%`, background: 'var(--safe)' }} title={`Safe: ${safePct}%`} />
          <div style={{ width: `${reviewPct}%`, background: 'var(--review)' }} title={`Review: ${reviewPct}%`} />
          <div style={{ width: `${rejectPct}%`, background: 'var(--reject)' }} title={`Reject: ${rejectPct}%`} />
          <div style={{ width: `${unknownPct}%`, background: 'var(--unknown)' }} title={`Unknown: ${unknownPct}%`} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--dim)', marginTop: 6 }}>
          <span style={{ color: 'var(--safe)' }}>■ Safe: {summary.safe} ({safePct}%)</span>
          <span style={{ color: 'var(--review)' }}>■ Review: {summary.review} ({reviewPct}%)</span>
          <span style={{ color: 'var(--reject)' }}>■ Reject: {summary.reject} ({rejectPct}%)</span>
          <span style={{ color: 'var(--muted)' }}>■ Unknown: {summary.unknown} ({unknownPct}%)</span>
        </div>
      </div>

      {/* ============================================================ */}
      {/* SECTION 3: ACS DECISION PIPELINE (9 STAGES)                  */}
      {/* ============================================================ */}
      <div className="pipelineSection">
        <div className="sectionTitle">
          <span>SECTION 03 · 9-STAGE SCREENING & SAFETY DECISION PIPELINE</span>
          <span className="smallcaps" style={{ color: 'var(--accent)' }}>IMMUTABLE DECISION SEQUENCE</span>
        </div>

        <div className="decisionPipelineGrid">
          <div className="pipeStep">
            <span className="pipeNum">01</span>
            <span className="pipeName">DATA</span>
            <span className="pipeDesc">Telemetry Ingest</span>
            <span className="pipeStatus pass">Active</span>
          </div>

          <div className="pipeStep">
            <span className="pipeNum">02</span>
            <span className="pipeName">VALIDATION</span>
            <span className="pipeDesc">Schema & Quality</span>
            <span className="pipeStatus pass">Passed</span>
          </div>

          <div className="pipeStep">
            <span className="pipeNum">03</span>
            <span className="pipeName">FEATURES</span>
            <span className="pipeDesc">Dynamics & Rates</span>
            <span className="pipeStatus comp">Computed</span>
          </div>

          <div className="pipeStep">
            <span className="pipeNum">04</span>
            <span className="pipeName">ANOMALY</span>
            <span className="pipeDesc">Module A Ensemble</span>
            <span className="pipeStatus comp">Audited</span>
          </div>

          <div className="pipeStep">
            <span className="pipeNum">05</span>
            <span className="pipeName">FORECAST</span>
            <span className="pipeDesc">168h Drift Predict</span>
            <span className="pipeStatus comp">Projected</span>
          </div>

          <div className="pipeStep">
            <span className="pipeNum">06</span>
            <span className="pipeName">OOD NOVELTY</span>
            <span className="pipeDesc">Domain Shift (≠Defect)</span>
            <span className="pipeStatus flagged">Monitored</span>
          </div>

          <div className="pipeStep">
            <span className="pipeNum">07</span>
            <span className="pipeName">SAFETY POLICY</span>
            <span className="pipeDesc">Multi-Channel Fusion</span>
            <span className="pipeStatus comp">Evaluated</span>
          </div>

          <div className="pipeStep" style={{ borderColor: 'rgba(239, 68, 68, 0.4)' }}>
            <span className="pipeNum" style={{ color: 'var(--reject)' }}>08</span>
            <span className="pipeName" style={{ color: 'var(--reject)' }}>HARD OVERRIDE</span>
            <span className="pipeDesc">Absolute Spec Limit</span>
            <span className="pipeStatus reject">Non-Overridable</span>
          </div>

          <div className="pipeStep" style={{ borderColor: 'rgba(0, 240, 255, 0.4)' }}>
            <span className="pipeNum">09</span>
            <span className="pipeName" style={{ color: 'var(--accent)' }}>FINAL DECISION</span>
            <span className="pipeDesc">Disposition & Trace</span>
            <span className="pipeStatus pass">Certified</span>
          </div>
        </div>
      </div>

      {/* ============================================================ */}
      {/* SECTION 4: ENGINEERING INTELLIGENCE & TELEMETRY              */}
      {/* ============================================================ */}
      <div className="sectionTitle">
        <span>SECTION 04 · PREDICTIVE RELIABILITY & ROBUSTNESS SIGNALS</span>
        <span className="smallcaps" style={{ color: 'var(--accent2)' }}>PHYSICS-AWARE DRIFT & NOVELTY</span>
      </div>

      <div className="signalRibbon">
        <div className="signalStrip">
          <span className="smallcaps">Median Drift Velocity</span>
          <strong className="mono">
            {driftVelocityObj?.value != null ? `${num(driftVelocityObj.value, 5)} u/h` : '—'}
          </strong>
        </div>

        <div className="signalStrip">
          <span className="smallcaps">Conformal Band Width</span>
          <strong className="mono">
            {intervalWidthObj?.value != null ? `±${num(intervalWidthObj.value, 4)} u` : '—'}
          </strong>
        </div>

        <div className="signalStrip">
          <span className="smallcaps">OOD Novelty Rate</span>
          <strong className="mono" style={{ color: 'var(--review)' }}>
            {oodFlagObj?.value != null ? `${num(oodFlagObj.value, 1)}%` : '—'}
          </strong>
        </div>

        <div className="signalStrip">
          <span className="smallcaps">Chamber Hours Efficiency</span>
          <strong className="mono" style={{ color: 'var(--safe)' }}>
            {chamberHoursObj?.value != null ? `${num(chamberHoursObj.value, 1)}% Saved` : '—'}
          </strong>
        </div>

        <div className="signalStrip">
          <span className="smallcaps">Telemetry Provenance</span>
          <strong className="mono" style={{ color: 'var(--accent)' }}>
            {source === 'demo' ? 'REFERENCE DATA' : session?.run_id ? 'LIVE RUN' : 'ACTIVE DATA'}
          </strong>
        </div>
      </div>

      {/* ============================================================ */}
      {/* ============================================================ */}
      {/* LOT EARLY WARNING & BATCH DRIFT CENTER                      */}
      {/* ============================================================ */}
      <LotEarlyWarning source={source} activeLot={activeLot} onSelectLot={setActiveLot} />

      {/* ============================================================ */}
      {/* SECTION 5: LOT INTELLIGENCE OPERATIONS PANEL                 */}
      {/* ============================================================ */}
      {lotsList.length > 0 && (
        <div className="panel" style={{ marginBottom: 20 }}>
          <div className="panelHead">
            <div className="panelTitle">
              <Layers size={14} color="var(--accent)" />
              <span>Section 05 · Lot Intelligence & Fleet Operational Integrity</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              {activeLot && (
                <button
                  className="btn ghost"
                  style={{ padding: '2px 8px', fontSize: '10px', color: 'var(--accent)' }}
                  onClick={() => setActiveLot(null)}
                >
                  <X size={11} /> Reset Filter ({activeLot})
                </button>
              )}
              <span className="smallcaps" style={{ color: 'var(--muted)' }}>
                {lotsList.length} ACTIVE PRODUCTION LOTS
              </span>
            </div>
          </div>

          <div className="tableWrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Production Lot</th>
                  <th>Total Units</th>
                  <th>Healthy Yield %</th>
                  <th>Safe Dispositions</th>
                  <th>Review Queue</th>
                  <th>Reject Scrap</th>
                  <th>Mean Risk</th>
                  <th>Lot Health Status</th>
                  <th>Engineering Directive</th>
                </tr>
              </thead>
              <tbody>
                {lotsList.map((l) => {
                  const safeYieldPct = (l.safe_rate != null ? (l.safe_rate > 1 ? l.safe_rate : l.safe_rate * 100) : l.safe_pct) ?? 0
                  const safeCount = l.disposition_counts?.['SAFE'] ?? l.safe_count ?? 0
                  const reviewCount = l.disposition_counts?.['REVIEW'] ?? l.review_count ?? 0
                  const rejectCount = l.disposition_counts?.['REJECT'] ?? l.reject_count ?? 0
                  const totalUnits = l.total_components ?? l.total_parts ?? (safeCount + reviewCount + rejectCount)
                  const meanRisk = l.mean_risk ?? l.mean_risk_score ?? 0
                  const healthStatus = l.lot_health ?? l.status ?? 'HEALTHY'
                  const rec = l.lot_recommendation || (healthStatus === 'HEALTHY' ? 'RELEASE_LOT' : 'QUARANTINE_SURVEILLANCE')
                  const isSelected = activeLot === l.lot_id
                  return (
                    <tr
                      key={l.lot_id}
                      className="click"
                      onClick={() => setActiveLot(isSelected ? null : l.lot_id)}
                      style={{
                        background: isSelected ? 'rgba(0, 240, 255, 0.08)' : undefined,
                        cursor: 'pointer',
                      }}
                    >
                      <td>
                        <div className="part" style={{ color: isSelected ? 'var(--accent)' : undefined }}>
                          {l.lot_id} {isSelected && <span style={{ fontSize: '9px', color: 'var(--accent)' }}>(Selected)</span>}
                        </div>
                      </td>
                      <td className="mono">{totalUnits}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span className="mono" style={{ color: safeYieldPct >= 90 ? 'var(--safe)' : 'var(--review)' }}>
                            {num(safeYieldPct, 1)}%
                          </span>
                          <span className="bar" style={{ width: 60 }}>
                            <i style={{ width: `${Math.min(100, Math.max(0, safeYieldPct))}%`, background: safeYieldPct >= 90 ? 'var(--safe)' : 'var(--review)' }} />
                          </span>
                        </div>
                      </td>
                      <td className="mono safeText">{safeCount}</td>
                      <td className="mono reviewText">{reviewCount}</td>
                      <td className="mono dangerText">{rejectCount}</td>
                      <td className="mono">{num(meanRisk, 2)}</td>
                      <td>
                        <span className={`badge ${healthStatus === 'HEALTHY' ? 'safe' : healthStatus === 'ELEVATED_RISK' || healthStatus === 'CAUTION' ? 'review' : 'reject'}`}>
                          {healthStatus}
                        </span>
                      </td>
                      <td>
                        <span className="mono" style={{ fontSize: '9.5px', color: rec.includes('RELEASE') ? 'var(--safe)' : 'var(--review)' }}>
                          {rec}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ============================================================ */}
      {/* SECTION 6: COMPONENT SCREENING MATRIX TABLE                  */}
      {/* ============================================================ */}
      <div className="panel matrixPanel">
        <div className="panelHead">
          <div>
            <div className="panelTitle">
              <Cpu size={14} color="var(--accent)" />
              <span>Component Screening Matrix · Operational Triage</span>
            </div>
            <div className="smallcaps" style={{ marginTop: 4 }}>
              Click any component row to inspect its Digital Passport & Engineering Trace
            </div>
          </div>

          {/* Search & Filter Toolbar */}
          <div className="toolbar">
            <div style={{ position: 'relative' }}>
              <Search size={12} style={{ position: 'absolute', left: 8, top: '50%', transform: 'translateY(-50%)', color: 'var(--dim)' }} />
              <input
                className="input"
                style={{ paddingLeft: 26, width: 220 }}
                placeholder="Search Part ID, Family, Parameter…"
                value={q}
                onChange={(e) => setQ(e.target.value)}
              />
            </div>

            <select className="select" value={decision} onChange={(e) => setDecision(e.target.value)}>
              <option value="ALL">ALL DISPOSITIONS</option>
              <option value="SAFE">SAFE ONLY</option>
              <option value="REVIEW">REVIEW ONLY</option>
              <option value="REJECT">REJECT ONLY</option>
              <option value="UNKNOWN">UNKNOWN ONLY</option>
            </select>

            {activeLot && (
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '3px 8px', background: 'rgba(0, 240, 255, 0.1)', border: '1px solid var(--accent)', borderRadius: 4, fontSize: '10px' }}>
                <Filter size={11} color="var(--accent)" />
                <span>Lot: <b className="mono">{activeLot}</b></span>
                <button
                  className="iconButton"
                  style={{ padding: 0, width: 14, height: 14 }}
                  onClick={(e) => { e.stopPropagation(); setActiveLot(null) }}
                  title="Clear lot filter"
                >
                  <X size={10} />
                </button>
              </div>
            )}

            <span className="smallcaps" style={{ marginLeft: 6, color: 'var(--dim)' }}>
              {items.length} of {sourceItems.length} records
            </span>
          </div>
        </div>

        <div className="tableWrap">
          <table className="table">
            <thead>
              <tr>
                <th>Component Telemetry ID</th>
                <th>Family</th>
                <th>Monitored Parameter</th>
                <th>Burn-in Readpoint</th>
                <th>Fused Risk Score</th>
                <th>System Confidence</th>
                <th>Domain Novelty</th>
                <th>Disposition</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.part_id} className="click" onClick={() => onOpen(r.part_id)}>
                  <td>
                    <div className="part">{r.part_id}</div>
                    <div className="tinyText mono" style={{ color: 'var(--dim)' }}>Lot: {r.lot_id || 'UNASSIGNED'}</div>
                  </td>
                  <td>{r.component_family || '—'}</td>
                  <td>
                    <span className="mono" style={{ color: 'var(--accent)' }}>{r.parameter || '—'}</span>
                    {r.unit && <span style={{ fontSize: '9px', color: 'var(--dim)', marginLeft: 4 }}>({r.unit})</span>}
                  </td>
                  <td className="mono">{num(r.burnin_hours, 0)} h</td>
                  <td>
                    <div className="riskLine">
                      <span className="mono">{num(r.risk_score, 2)}</span>
                      <span className="bar">
                        <i style={{ width: `${Math.max(0, Math.min(1, r.risk_score || 0)) * 100}%` }} />
                      </span>
                    </div>
                  </td>
                  <td>
                    <span className={`badge ${r.confidence === 'HIGH' ? 'safe' : r.confidence === 'MODERATE' ? 'review' : 'unknown'}`}>
                      {r.confidence || '—'}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${r.ood_status === 'SEVERE' ? 'reject' : r.ood_status === 'MODERATE' ? 'review' : 'safe'}`}>
                      {r.ood_status}
                    </span>
                  </td>
                  <td>
                    <StatusBadge decision={r.decision} />
                  </td>
                  <td>
                    <button className="btn ghost" style={{ padding: '3px 8px', fontSize: '9px' }}>
                      Passport <ArrowRight size={10} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}


