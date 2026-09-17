import { useMemo, useState } from 'react'
import {
  ArrowLeft,
  Check,
  ChevronRight,
  Cpu,
  Database,
  FileSearch,
  FileUp,
  Layers,
  Play,
  ShieldAlert,
  ShieldCheck,
  Table2,
  UploadCloud,
  Workflow,
} from 'lucide-react'
import { apiGet, apiPostFile } from '../api/client'
import type { ScreeningResponse } from '../types/api'

interface ProfileResponse {
  filename: string
  profile: {
    source_format: string
    columns: Record<string, string>
    time_columns: Record<string, number>
    measurement_columns: Array<[string, string | null, number]>
    warnings: string[]
  }
  mapping: Array<{ canonical_field: string; source_column: string | null; status: string }>
  validation: { ok: boolean; errors: string[] }
  canonical_rows: number
  parts: number
  parameters: number
  preview: Array<Record<string, unknown>>
}

const INTAKE_STEPS = [
  { num: '01', title: 'DATA INTAKE' },
  { num: '02', title: 'SCHEMA VALIDATION' },
  { num: '03', title: 'FEATURE EXTRACTION' },
  { num: '04', title: 'AI SCREENING' },
  { num: '05', title: 'SAFETY DECISION' },
  { num: '06', title: 'ENGINEERING REPORT' },
]

export function NewScreeningPage({
  onComplete,
  onCancel,
}: {
  onComplete: (r: ScreeningResponse, source: 'active' | 'demo') => void
  onCancel: () => void
}) {
  const [step, setStep] = useState(1) // 1: Upload, 2: Schema, 3: Processing (steps 3-5), 4: Complete (step 6)
  const [file, setFile] = useState<File | null>(null)
  const [profile, setProfile] = useState<ProfileResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [drag, setDrag] = useState(false)

  const fileLabel = useMemo(
    () => (file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB` : 'No file selected'),
    [file]
  )

  async function inspect(selected: File) {
    setBusy(true)
    setMessage('')
    try {
      const data = await apiPostFile<ProfileResponse>('/api/intake/profile', selected, 24, 168, false)
      setFile(selected)
      setProfile(data)
      setStep(2)
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Schema profiling failed.')
    } finally {
      setBusy(false)
    }
  }

  async function runScreening() {
    if (!file) return
    setBusy(true)
    setMessage('')
    try {
      setStep(3)
      const result = await apiPostFile<ScreeningResponse>('/api/screen', file, 24, 168, true)
      setStep(4)
      setMessage(`Screening run ${result.run_id || 'active'} complete.`)
      onComplete(result, 'active')
    } catch (e) {
      setStep(2)
      setMessage(e instanceof Error ? e.message : 'Screening execution failed.')
    } finally {
      setBusy(false)
    }
  }

  async function demo() {
    setBusy(true)
    setMessage('')
    try {
      const result = await apiGet<ScreeningResponse>('/api/demo')
      setStep(4)
      setMessage('Reference telemetry successfully loaded.')
      onComplete(result, 'demo')
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Reference scenario unavailable.')
    } finally {
      setBusy(false)
    }
  }

  // Active step index for 6-step rail (0-indexed)
  const activeStepIdx = step === 1 ? 0 : step === 2 ? 1 : step === 3 ? 3 : 5

  return (
    <div className="page intakePage">
      {/* ============================================================ */}
      {/* HEADER                                                       */}
      {/* ============================================================ */}
      <div className="pageHead">
        <div>
          <button className="btn ghost" onClick={onCancel}>
            <ArrowLeft size={13} /> Back to Mission Overview
          </button>
          <div className="eyebrow" style={{ marginTop: 14 }}>
            INGESTION WORKFLOW · AEROSPACE DATA INTAKE
          </div>
          <h1>Initialize Component Screening Run</h1>
          <p>
            Ingest heterogeneous electronic component test data, validate schema semantics against space standards,
            and execute the conservative ML screening & safety policy stack.
          </p>
        </div>

        <div className="actions">
          <button className="btn ghost" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>

      {/* ============================================================ */}
      {/* 6-STEP INTAKE WORKFLOW STEP RAIL                             */}
      {/* ============================================================ */}
      <div className="stepRail">
        {INTAKE_STEPS.map((s, idx) => {
          const isDone = activeStepIdx > idx
          const isCurrent = activeStepIdx === idx
          return (
            <div
              key={s.title}
              className={`stepItem ${isDone ? 'done' : ''} ${isCurrent ? 'current' : ''}`}
            >
              <span>{isDone ? <Check size={11} /> : s.num}</span>
              <b>{s.title}</b>
            </div>
          )
        })}
      </div>

      {message && (
        <div className="callout" style={{ marginBottom: 16 }}>
          <ShieldCheck size={14} color="var(--accent)" />
          <span>{message}</span>
        </div>
      )}

      {/* ============================================================ */}
      {/* STEP 1: UPLOAD OR LOAD REFERENCE TELEMETRY                    */}
      {/* ============================================================ */}
      {step === 1 && (
        <section className="intakeGrid">
          <div className="panel">
            <div className="panelHead">
              <div>
                <div className="panelTitle">
                  <UploadCloud size={14} color="var(--accent)" />
                  <span>Upload Component Screening Telemetry (CSV)</span>
                </div>
                <div className="smallcaps" style={{ marginTop: 4 }}>
                  Multi-rate burn-in data · Schema-adaptive parser
                </div>
              </div>
              <FileUp size={16} color="var(--accent)" />
            </div>

            <div
              className={`dropZone ${drag ? 'drag' : ''}`}
              onDragEnter={() => setDrag(true)}
              onDragLeave={() => setDrag(false)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault()
                setDrag(false)
                const f = e.dataTransfer.files[0]
                if (f) inspect(f)
              }}
            >
              <UploadCloud size={36} color="var(--accent)" style={{ marginBottom: 8 }} />
              <h3>Drag & Drop Telemetry CSV Here</h3>
              <p>
                Automatic layout detection resolves wide (value_0h, value_24h, ...) or long formats and standardizes parameter units.
              </p>
              <label className="btn primary" style={{ marginTop: 14 }}>
                Choose Local CSV File
                <input
                  type="file"
                  accept=".csv"
                  hidden
                  onChange={(e) => e.target.files?.[0] && inspect(e.target.files[0])}
                />
              </label>
              {busy && (
                <div style={{ marginTop: 12, fontSize: '10.5px', color: 'var(--accent)' }}>
                  Profiling schema & calculating readpoints…
                </div>
              )}
              <div className="fileName mono" style={{ color: 'var(--soft)', marginTop: 12 }}>
                {fileLabel}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panelHead">
              <div>
                <div className="panelTitle">
                  <Database size={14} color="var(--accent2)" />
                  <span>Reference Flight Telemetry</span>
                </div>
                <div className="smallcaps" style={{ marginTop: 4 }}>
                  Packaged evaluation scenario
                </div>
              </div>
              <span className="badge safe">STANDARDIZED</span>
            </div>

            <div className="panelBody">
              <h3 style={{ fontSize: '15px', color: 'var(--ink)', margin: '0 0 8px' }}>
                Load Pre-Validated Aerospace Test Lot
              </h3>
              <p style={{ fontSize: '11.5px', lineHeight: 1.6, color: 'var(--muted)', margin: '0 0 16px' }}>
                Run the end-to-end mission intelligence stack using our verified EEE component test lot. Inspect
                anomaly detection, 168h drift forecasting, domain novelty, and 9-stage decision traces immediately.
              </p>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: 8,
                  marginBottom: 20,
                  fontSize: '10px',
                  color: 'var(--soft)',
                }}
              >
                <div className="fact">✓ Heterogeneous Families</div>
                <div className="fact">✓ Burn-In Time Series</div>
                <div className="fact">✓ Engineering Limits</div>
                <div className="fact">✓ 9-Stage Decision Trace</div>
                <div className="fact">✓ Quantified Safety Margin</div>
                <div className="fact">✓ OOD Decomposition</div>
              </div>

              <button className="btn primary" onClick={demo} disabled={busy} style={{ width: '100%' }}>
                <Play size={13} /> Load Reference Scenario
              </button>
            </div>
          </div>
        </section>
      )}

      {/* ============================================================ */}
      {/* STEP 2: SCHEMA PROFILING & MAPPING                           */}
      {/* ============================================================ */}
      {step === 2 && profile && (
        <section className="panel">
          <div className="panelHead">
            <div>
              <div className="panelTitle">
                <Table2 size={15} color="var(--accent)" />
                <span>STEP 02 · Schema Mapping & Semantic Conformance</span>
              </div>
              <div className="smallcaps" style={{ marginTop: 4 }}>
                {profile.filename} · {profile.profile.source_format.toUpperCase()} FORMAT
              </div>
            </div>
            <span className={`badge ${profile.validation.ok ? 'safe' : 'review'}`}>
              {profile.validation.ok ? 'SCHEMA VERIFIED' : 'WARNINGS DETECTED'}
            </span>
          </div>

          <div className="panelBody">
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: 10,
                marginBottom: 16,
              }}
            >
              <div className="fact">
                <span className="smallcaps">Canonical Rows</span>
                <strong className="mono">{profile.canonical_rows.toLocaleString()}</strong>
              </div>
              <div className="fact">
                <span className="smallcaps">Component Parts</span>
                <strong className="mono">{profile.parts}</strong>
              </div>
              <div className="fact">
                <span className="smallcaps">Measured Parameters</span>
                <strong className="mono">{profile.parameters}</strong>
              </div>
              <div className="fact">
                <span className="smallcaps">Time Readpoints</span>
                <strong className="mono">{Object.keys(profile.profile.time_columns).length}</strong>
              </div>
            </div>

            <div className="tableWrap" style={{ border: '1px solid var(--line)', borderRadius: 'var(--radius-md)' }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Canonical Telemetry Field</th>
                    <th>Incoming Source Column</th>
                    <th>Conformance Status</th>
                  </tr>
                </thead>
                <tbody>
                  {profile.mapping.map((m, i) => (
                    <tr key={`${m.canonical_field}-${i}`}>
                      <td className="mono" style={{ color: 'var(--accent)' }}>
                        {m.canonical_field}
                      </td>
                      <td>{m.source_column || '—'}</td>
                      <td>
                        <span className={`badge ${m.status.includes('UNRESOLVED') ? 'review' : 'safe'}`}>
                          {m.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {profile.profile.warnings.length > 0 && (
              <div className="callout" style={{ marginTop: 14 }}>
                <ShieldAlert size={14} color="var(--review)" />
                <span>Warnings: {profile.profile.warnings.join(' · ')}</span>
              </div>
            )}

            <div style={{ marginTop: 16 }}>
              <div className="smallcaps" style={{ marginBottom: 6 }}>
                Canonical Preview (Top 3 Records)
              </div>
              <pre
                className="mono"
                style={{
                  background: '#09101F',
                  padding: 14,
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--line)',
                  color: 'var(--soft)',
                  fontSize: '9.5px',
                  maxHeight: 200,
                  overflow: 'auto',
                }}
              >
                {JSON.stringify(profile.preview.slice(0, 3), null, 2)}
              </pre>
            </div>

            <div className="actions" style={{ marginTop: 20, justifyContent: 'flex-end' }}>
              <button
                className="btn ghost"
                onClick={() => {
                  setStep(1)
                  setProfile(null)
                }}
              >
                Choose Another Dataset
              </button>
              <button
                className="btn primary"
                disabled={!profile.validation.ok || busy}
                onClick={runScreening}
              >
                <ShieldCheck size={14} /> Validate & Execute Screening Stack
              </button>
            </div>
          </div>
        </section>
      )}

      {/* ============================================================ */}
      {/* STEP 3: PIPELINE EXECUTION IN PROGRESS                       */}
      {/* ============================================================ */}
      {step === 3 && (
        <section className="panel" style={{ padding: '60px 20px', textAlign: 'center' }}>
          <div
            style={{
              width: 54,
              height: 54,
              borderRadius: '50%',
              border: '2px solid var(--accent)',
              boxShadow: '0 0 20px rgba(0, 240, 255, 0.3)',
              margin: '0 auto 20px',
              display: 'grid',
              placeItems: 'center',
            }}
          >
            <Cpu size={26} color="var(--accent)" />
          </div>

          <div className="eyebrow" style={{ color: 'var(--accent)' }}>
            SCREENING PIPELINE ACTIVE
          </div>
          <h2 style={{ fontSize: '24px', color: 'var(--ink)', margin: '8px 0' }}>
            Processing Component Evidence Stack…
          </h2>
          <p style={{ maxWidth: 600, margin: '0 auto 24px', color: 'var(--muted)', fontSize: '11.5px', lineHeight: 1.6 }}>
            Executing multi-rate dynamics, multivariate anomaly scoring, 168h drift forecasting, domain novelty analysis,
            and the conservative safety policy.
          </p>

          <div
            style={{
              maxWidth: 580,
              margin: '0 auto',
              border: '1px solid var(--line)',
              borderRadius: 'var(--radius-md)',
              overflow: 'hidden',
              textAlign: 'left',
            }}
          >
            {[
              'STEP 01: DATA INGESTION & READPOINT PARSING',
              'STEP 02: CANONICAL SCHEMA VALIDATION',
              'STEP 03: MULTI-RATE TEMPORAL FEATURE EXTRACTION',
              'STEP 04: MODULE A ANOMALY & ENSEMBLE SCORING',
              'STEP 05: MODULE B DEGRADATION DRIFT FORECASTING',
              'STEP 06: OOD DOMAIN NOVELTY ASSESSMENT',
              'STEP 07: CONSERVATIVE SAFETY POLICY EVALUATION',
            ].map((st, i) => (
              <div
                key={st}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  padding: '10px 14px',
                  borderBottom: i < 6 ? '1px solid var(--line)' : 'none',
                  background: i === 0 ? 'var(--cyan-soft)' : '#0A1122',
                  fontSize: '9.5px',
                  fontFamily: 'JetBrains Mono',
                  color: i === 0 ? 'var(--accent)' : 'var(--soft)',
                }}
              >
                <span>{st}</span>
                <span style={{ color: i === 0 ? 'var(--accent)' : 'var(--dim)' }}>
                  {i === 0 ? 'RUNNING' : 'QUEUED'}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ============================================================ */}
      {/* STEP 4: SCREENING COMPLETE                                   */}
      {/* ============================================================ */}
      {step === 4 && (
        <section className="panel" style={{ padding: '80px 20px', textAlign: 'center' }}>
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: '50%',
              border: '2px solid var(--safe)',
              background: 'var(--safe-bg)',
              boxShadow: '0 0 20px rgba(16, 185, 129, 0.3)',
              margin: '0 auto 20px',
              display: 'grid',
              placeItems: 'center',
            }}
          >
            <Check size={26} color="var(--safe)" />
          </div>

          <div className="eyebrow" style={{ color: 'var(--safe)' }}>
            SCREENING PIPELINE CERTIFIED
          </div>
          <h2 style={{ fontSize: '24px', color: 'var(--ink)', margin: '8px 0' }}>
            Mission Screening Evidence Ready
          </h2>
          <p style={{ maxWidth: 540, margin: '0 auto 24px', color: 'var(--muted)', fontSize: '11.5px', lineHeight: 1.6 }}>
            The operational screening batch has been computed and verified. Dispositions, 9-stage traces, and Digital Passports
            are ready for inspection.
          </p>
        </section>
      )}
    </div>
  )
}
