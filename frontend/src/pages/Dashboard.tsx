import { useMemo, useState } from 'react'
import { ArrowUpRight, Gauge, Search, ShieldCheck, Upload } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import { BatchSummary, ComponentSummary, ScreeningResponse } from '../types/api'
import { apiPostFile } from '../api/client'
import { KPIGrid } from '../components/overview/KPIGrid'
import { ScreeningMatrix } from '../components/overview/ScreeningMatrix'

export function DashboardPage({ onOpen, onOpenEvidence }: { onOpen: (id: string) => void; onOpenEvidence?: () => void }) {
  const { data: s, loading: sl, error: se, reload } = useApi<BatchSummary>('/api/summary')
  const { data: c, loading: cl, error: ce, reload: cr } = useApi<{ items: ComponentSummary[]; count: number }>('/api/components?limit=250')
  const [q, setQ] = useState('')
  const [decision, setDecision] = useState('ALL')
  const [busy, setBusy] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')
  const [live, setLive] = useState<ScreeningResponse | null>(null)

  const summary = live?.summary || s
  const sourceItems = live?.components || c?.items || []

  const items = useMemo(() => {
    let rows = sourceItems
    const needle = q.trim().toLowerCase()
    if (needle) rows = rows.filter(r => `${r.part_id} ${r.component_family || ''} ${r.parameter || ''}`.toLowerCase().includes(needle))
    if (decision !== 'ALL') rows = rows.filter(r => r.decision === decision)
    return rows
  }, [sourceItems, q, decision])

  async function upload(file: File) {
    setBusy(true)
    setUploadMessage('')
    try {
      const result = await apiPostFile<ScreeningResponse>('/api/screen', file)
      setLive(result)
      setUploadMessage(`Live screening loaded: ${result.summary.total_components} components from the uploaded batch.`)
    } catch (e) {
      setUploadMessage(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  if (sl || cl) return <div className="page"><div className="loadingBand">Loading operational screening state…</div></div>
  if (se || ce) return <div className="page"><div className="error">{se || ce}</div></div>
  if (!summary) return null

  return <div className="page">
    <div className="pageHead">
      <div>
        <div className="eyebrow">SCREEN 01 · BATCH OVERVIEW</div>
        <h1>Screening Overview</h1>
        <p>Present condition, future-risk signals, data trust and final engineering disposition in one operational view.</p>
      </div>
      <div className="actions">
        <label className="btn primary"><Upload size={14} />{busy ? 'SCREENING…' : 'SCREEN CSV'}<input type="file" accept=".csv" hidden onChange={e => e.target.files?.[0] && upload(e.target.files[0])} /></label>
        <button className="btn ghost" onClick={() => { setLive(null); reload(); cr() }}><Gauge size={14} /> Refresh</button>
        {onOpenEvidence && <button className="btn ghost" onClick={onOpenEvidence}>Evidence</button>}
      </div>
    </div>

    {uploadMessage && <div className="callout" style={{ marginBottom: 14 }}><ArrowUpRight size={13} />{uploadMessage}</div>}

    <KPIGrid s={summary} />

    <div className="overviewMainGrid">
      <ScreeningMatrix items={items} onOpen={onOpen} />
      <FleetSignal summary={summary} />
    </div>

    <div className="panel" style={{ marginTop: 14 }}>
      <div className="panelHead"><div><div className="panelTitle">Filter & triage</div><div className="smallcaps" style={{ marginTop: 5 }}>Find a component, family or parameter</div></div><div className="toolbar"><Search size={14} color="#817167" /><input className="input" placeholder="Search component / family / parameter" value={q} onChange={e => setQ(e.target.value)} /><select className="select" value={decision} onChange={e => setDecision(e.target.value)}><option>ALL</option><option>SAFE</option><option>REVIEW</option><option>REJECT</option><option>UNKNOWN</option></select></div></div>
      <div className="panelBody compact"><div className="smallcaps">Showing {items.length} of {sourceItems.length} available components</div><div className="triageHint">Select a row to open the complete Component Intelligence packet.</div></div>
    </div>

    <div className="footerNote">Source: {summary.report_source}. Future-defect detection is intentionally kept separate from persisted operational screening.</div>
  </div>
}

function FleetSignal({ summary }: { summary: BatchSummary }) {
  const rows = [['SAFE', summary.safe, 'safe'], ['REVIEW', summary.review, 'review'], ['REJECT', summary.reject, 'reject'], ['UNKNOWN', summary.unknown, 'unknown'] as const]
  return <div className="panel fleetPanel"><div className="panelHead"><div><div className="panelTitle">Fleet signal</div><div className="smallcaps" style={{ marginTop: 4 }}>Decision distribution</div></div><ShieldCheck size={16} color="#c47a4a" /></div><div className="panelBody fleetBody">{rows.map(([name, value, kind]) => { const pct = summary.total_components ? Number(value) / summary.total_components * 100 : 0; return <div className="fleetRow" key={name}><div className="fleetMeta"><span>{name}</span><span className="mono">{value}</span></div><div className="fleetTrack"><i className={kind} style={{ width: `${pct}%` }} /></div></div> })}<div className="fleetFooter"><div className="smallcaps">Operating principle</div><p>Risk and data trust remain separate. A REVIEW state routes suspicious evidence to engineering attention rather than silently promoting it to automatic rejection.</p></div></div></div>
}
