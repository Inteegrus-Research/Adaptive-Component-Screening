import type { ReactNode } from 'react'
import { Activity, Database, ShieldCheck, ArrowUpRight, Gauge } from 'lucide-react'
import * as echarts from 'echarts'
import { useEffect, useRef } from 'react'
import { useApi } from '../hooks/useApi'
import type { BenchmarkResponse } from '../types/api'
import { num } from '../utils/format'

export function AuditPage({ onInspect }: { onInspect?: (id: string) => void }) {
  const { data, error, loading } = useApi<BenchmarkResponse>('/api/audit/benchmark')
  if (loading) return <div className="page"><div className="loadingBand">Loading benchmark evidence…</div></div>
  if (error) return <div className="page"><div className="error">{error}</div></div>
  if (!data?.available) {
    return (
      <div className="page">
        <PageHeader />
        <div className="callout"><Database size={14} /> Benchmark artifacts are not available at the active report directory.</div>
      </div>
    )
  }

  const files = data.files || {}
  const system = files['system_metrics_mean_std.csv'] || []
  const lead = files['lead_time_mean_std.csv'] || []
  const escape = files['latent_escape_mean_std.csv'] || []
  const mechanism = files['mechanism_metrics_mean.csv'] || []
  const forecast = files['forecast_metrics_mean.csv'] || []
  const robustness = files['robustness_metrics.csv'] || []
  const progressive = files['progressive_metrics_mean.csv'] || []

  const recall = metricValue(system, 'recall')
  const precision = metricValue(system, 'precision')
  const fpr = metricValue(system, 'fpr')
  const escapeRecall = metricValue(system, 'escape_recall') ?? metricValue(escape, 'escape_recall')
  const leadMedian = metricValue(lead, 'median_lead_time_h')

  return (
    <div className="page">
      <PageHeader source={data.source} />
      {data.demo_mode && (
        <div className="demoBanner">
          <span className="demoTag">DEMO SCENARIO</span>
          <span>Synthetic presentation artifacts only. Replace these benchmark files with the actual benchmark outputs before external submission.</span>
        </div>
      )}

      <div className="kpis auditKpis">
        <AuditK label="Recall / detection" value={recall} fmt="pct" />
        <AuditK label="Latent escape recall" value={escapeRecall} fmt="pct" />
        <AuditK label="Precision" value={precision} fmt="pct" />
        <AuditK label="False positive rate" value={fpr} fmt="pct" />
        <AuditK label="Median lead time" value={leadMedian} suffix="h" />
      </div>

      <div className="auditCallout">
        <ShieldCheck size={16} />
        <div><strong>Proof layer — not a presentation score.</strong><span>These values are read from persisted benchmark artifacts. Operational screening and benchmark ground truth remain separate.</span></div>
      </div>

      <div className="auditHeroGrid">
        <BenchmarkChartPanel title="Progressive detection" subtitle="Performance as burn-in evidence accumulates">
          <BenchmarkLineChart rows={progressive} xKey="origin_h" yKeys={[['recall', 'Recall'], ['fpr', 'False positive rate']]} yPercent />
        </BenchmarkChartPanel>
        <BenchmarkChartPanel title="Forecast model comparison" subtitle="Persisted holdout comparison — model behaviour, not browser inference">
          <ForecastBarChart rows={forecast} />
        </BenchmarkChartPanel>
      </div>

      <div className="auditGrid">
        <AuditPanel title="System Metrics" icon={<ShieldCheck size={14} />} rows={system} />
        <AuditPanel title="Lead Time" icon={<Activity size={14} />} rows={lead} emphasize />
        <AuditPanel title="Latent Escape" icon={<ShieldCheck size={14} />} rows={escape} />
        <AuditPanel title="Mechanism Breakdown" icon={<Activity size={14} />} rows={mechanism} />
        <AuditPanel title="Robustness" icon={<Database size={14} />} rows={robustness} />
      </div>

      {onInspect && (
        <div className="panel auditInspect" style={{ marginTop: 14 }}>
          <div className="panelHead"><div><div className="panelTitle">Close the loop</div><div className="smallcaps" style={{ marginTop: 4 }}>Return from system evidence to a real component</div></div><ArrowUpRight size={15} color="#c47a4a" /></div>
          <div className="panelBody compact"><div className="auditInspectCopy"><div><strong>Need a component-level proof point?</strong><span>Open the highest-risk persisted component and inspect its full intelligence packet, trajectory and evidence record.</span></div><button className="btn primary" onClick={() => onInspect('MMIC_ANALOG_008_00007')}><Gauge size={13} /> Inspect demo component</button></div></div>
        </div>
      )}

      <div className="footerNote">Source: {data.source}. The benchmark page is intentionally transparent about artifact provenance and does not infer missing ground truth.</div>
    </div>
  )
}

function PageHeader({ source }: { source?: string }) {
  return (
    <div className="pageHead">
      <div><div className="eyebrow">SCREEN 04 · AUDIT / BENCHMARK</div><h1>Audit & Benchmark</h1><p>System-level validation for progressive detection, latent escape, lead time, mechanism behaviour and robustness.</p></div>
      {source && <div className="benchmarkSource"><Database size={13} /> SOURCE · {source}</div>}
    </div>
  )
}

function AuditK({ label, value, suffix = '', fmt = 'raw' }: { label: string; value: number | null; suffix?: string; fmt?: 'raw' | 'pct' }) {
  const shown = value == null ? '—' : fmt === 'pct' ? `${num(value * 100, 1)}%` : `${num(value, 2)}${suffix ? ` ${suffix}` : ''}`
  return <div className="kpi"><div className="label">{label}</div><div className="value mono">{shown}</div><div className="meta">persisted benchmark artifact</div></div>
}

function metricValue(rows: any[], metric: string): number | null {
  for (const row of rows) {
    if (String(row.metric || '').toLowerCase() === metric.toLowerCase()) return finite(row.mean ?? row.value)
  }
  for (const row of rows) {
    for (const [key, value] of Object.entries(row)) {
      if (key.toLowerCase().includes(metric.toLowerCase())) {
        const n = finite(value)
        if (n != null) return n
      }
    }
  }
  return null
}

function finite(v: any): number | null { const n = Number(v); return Number.isFinite(n) ? n : null }

function BenchmarkChartPanel({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return <div className="panel chartPanelBox"><div className="panelHead"><div><div className="panelTitle">{title}</div><div className="smallcaps" style={{ marginTop: 4 }}>{subtitle}</div></div></div><div className="chartPanel">{children}</div></div>
}

function BenchmarkLineChart({ rows, xKey, yKeys, yPercent = false }: { rows: any[]; xKey: string; yKeys: [string, string][]; yPercent?: boolean }) {
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (!ref.current || !rows.length) return
    const chart = echarts.init(ref.current)
    const x = rows.map(r => Number(r[xKey]))
    chart.setOption({
      animation: false,
      grid: { left: 52, right: 18, top: 24, bottom: 42 },
      tooltip: { trigger: 'axis', backgroundColor: '#171311', borderColor: '#5b3e2f', textStyle: { color: '#efe2d6', fontSize: 10 } },
      xAxis: { type: 'category', data: x.map(v => `${v}h`), axisLabel: { color: '#8f7d72', fontSize: 9 }, splitLine: { lineStyle: { color: '#2d241f' } } },
      yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#8f7d72', fontSize: 9, formatter: yPercent ? (v: number) => `${Math.round(v * 100)}%` : undefined }, splitLine: { lineStyle: { color: '#2d241f' } } },
      series: yKeys.map(([key, name], i) => ({ name, type: 'line', data: rows.map(r => finite(r[key])), smooth: true, symbol: 'circle', symbolSize: 5, lineStyle: { color: i === 0 ? '#eaded5' : '#c47a4a', width: 2 }, itemStyle: { color: i === 0 ? '#eaded5' : '#c47a4a' } })),
      legend: { bottom: 4, textStyle: { color: '#897970', fontSize: 9 } },
    })
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(ref.current)
    return () => { ro.disconnect(); chart.dispose() }
  }, [rows, xKey, yKeys, yPercent])
  return rows.length ? <div ref={ref} className="chart chartSmall" /> : <div className="empty">Progressive artifact unavailable.</div>
}

function ForecastBarChart({ rows }: { rows: any[] }) {
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (!ref.current || !rows.length) return
    const chart = echarts.init(ref.current)
    chart.setOption({
      animation: false,
      grid: { left: 56, right: 18, top: 24, bottom: 54 },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#171311', borderColor: '#5b3e2f', textStyle: { color: '#efe2d6', fontSize: 10 } },
      xAxis: { type: 'category', data: rows.map(r => String(r.model).replace(/_/g, ' ')), axisLabel: { color: '#8f7d72', fontSize: 8, rotate: 18 } },
      yAxis: { type: 'value', axisLabel: { color: '#8f7d72', fontSize: 9 }, splitLine: { lineStyle: { color: '#2d241f' } } },
      series: [
        { name: 'MAE', type: 'bar', data: rows.map(r => finite(r.mae)), barMaxWidth: 20, itemStyle: { color: '#c47a4a' } },
        { name: 'RMSE', type: 'bar', data: rows.map(r => finite(r.rmse)), barMaxWidth: 20, itemStyle: { color: '#73513e' } },
      ],
      legend: { bottom: 4, textStyle: { color: '#897970', fontSize: 9 } },
    })
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(ref.current)
    return () => { ro.disconnect(); chart.dispose() }
  }, [rows])
  return rows.length ? <div ref={ref} className="chart chartSmall" /> : <div className="empty">Forecast comparison unavailable.</div>
}

function AuditPanel({ title, icon, rows, emphasize = false }: { title: string; icon: ReactNode; rows: any[]; emphasize?: boolean }) {
  const cols = rows.length ? Object.keys(rows[0]).slice(0, 7) : []
  return <div className={`panel auditPanel ${emphasize ? 'emphasize' : ''}`}><div className="panelHead"><div className="auditTitle"><span>{icon}</span><div><div className="panelTitle">{title}</div><div className="smallcaps" style={{ marginTop: 4 }}>persisted artifact</div></div></div></div><div className="panelBody auditBody">{rows.length ? <div className="tableWrap"><table className="miniTable"><thead><tr>{cols.map(k => <th key={k}>{prettyKey(k)}</th>)}</tr></thead><tbody>{rows.slice(0, 12).map((r, i) => <tr key={i}>{cols.map(k => <td key={k} className={typeof r[k] === 'number' ? 'mono' : ''}>{formatValue(r[k])}</td>)}</tr>)}</tbody></table></div> : <div className="empty">No rows available in this artifact.</div>}</div></div>
}
function prettyKey(k: string) { return k.replace(/_/g, ' ').slice(0, 30) }
function formatValue(v: any) { if (typeof v === 'number') return Number.isInteger(v) ? v.toString() : num(v, 4); return String(v ?? '—') }
