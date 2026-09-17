import { AlertTriangle, ShieldCheck, Download, Filter, Layers, CheckCircle2, ChevronRight, X } from 'lucide-react'
import type { LotsResponse, LotSummary } from '../../types/api'
import { useApi } from '../../hooks/useApi'

export function LotEarlyWarning({
  source = 'demo',
  activeLot,
  onSelectLot,
}: {
  source?: 'active' | 'demo'
  activeLot: string | null
  onSelectLot: (lotId: string | null) => void
}) {
  const sourceQ = source === 'demo' ? '?source=demo' : ''
  const { data, loading, error } = useApi<LotsResponse>(`/api/lots${sourceQ}`)

  const lots: LotSummary[] = data?.lots || (data as any)?.items || []
  const earlyWarnings = data?.early_warnings || []

  if (loading) {
    return <div className="panel compact"><div className="loadingBand">Analyzing lot population & drift metrics…</div></div>
  }
  if (error || !data || !lots.length) {
    return null
  }

  const criticalWarning = earlyWarnings.find((w) => w.severity === 'critical') || earlyWarnings[0]

  function downloadLotCsv() {
    if (!data) return
    const rows = [
      [
        'LOT_ID',
        'TOTAL_PARTS',
        'SAFE_COUNT',
        'REVIEW_COUNT',
        'REJECT_COUNT',
        'UNKNOWN_COUNT',
        'SAFE_PCT',
        'REVIEW_PCT',
        'REJECT_PCT',
        'MEAN_RISK_SCORE',
        'MAX_RISK_SCORE',
        'MEAN_OOD_SCORE',
        'OOD_RATE_PCT',
        'STATUS',
        'DIAGNOSTIC',
      ],
      ...lots.map((l) => [
        l.lot_id,
        String(l.total_parts ?? l.total_components ?? 0),
        String(l.safe_count ?? l.disposition_counts?.['SAFE'] ?? 0),
        String(l.review_count ?? l.disposition_counts?.['REVIEW'] ?? 0),
        String(l.reject_count ?? l.disposition_counts?.['REJECT'] ?? 0),
        String(l.unknown_count ?? l.disposition_counts?.['UNKNOWN'] ?? 0),
        `${l.safe_pct ?? (l.safe_rate != null ? l.safe_rate * 100 : 0)}%`,
        `${l.review_pct ?? (l.review_rate != null ? l.review_rate * 100 : 0)}%`,
        `${l.reject_pct ?? (l.reject_rate != null ? l.reject_rate * 100 : 0)}%`,
        String(l.mean_risk_score ?? l.mean_risk ?? ''),
        String(l.max_risk_score ?? l.max_risk ?? ''),
        String(l.mean_ood_score ?? ''),
        `${l.ood_rate_pct ?? 0}%`,
        l.status || 'HEALTHY',
        l.diagnostic ? `"${l.diagnostic.replace(/"/g, '""')}"` : '""',
      ]),
    ]
    const csvContent = rows.map((r) => r.join(',')).join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ACS_LOT_SUMMARY_${data.source.replace(/[^a-zA-Z0-9]/g, '_')}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="lotIntelligenceSection" style={{ marginBottom: 16 }}>
      {/* High-Impact Aerospace Lot Warning Banner */}
      {criticalWarning && (
        <div className={`lotEarlyWarningBanner ${criticalWarning.severity === 'critical' ? 'critical' : 'warning'}`}>
          <div className="lotWarningIcon">
            <AlertTriangle size={20} />
          </div>
          <div className="lotWarningBody">
            <div className="lotWarningTagRow">
              <span className="lotWarningBadge">
                {criticalWarning.severity === 'critical' ? 'CRITICAL WAFER DRIFT EXCURSION' : 'LOT EARLY WARNING'}
              </span>
              <span className="mono lotWarningLid">TARGET: {criticalWarning.lot_id}</span>
              <span className="lotWarningMetric">ESCALATION BURDEN: {criticalWarning.escalation_burden}%</span>
              <span className="lotWarningMetric">OOD NOVELTY: {criticalWarning.ood_rate}%</span>
            </div>
            <div className="lotWarningMessage">{criticalWarning.message}</div>
            <div className="lotWarningAction">
              <strong>RECOMMENDED PROTOCOL:</strong> {criticalWarning.recommended_action}
            </div>
          </div>
          <div className="lotWarningActions">
            <button
              className="btn primary"
              onClick={() => onSelectLot(activeLot === criticalWarning.lot_id ? null : criticalWarning.lot_id)}
            >
              <Filter size={12} />
              {activeLot === criticalWarning.lot_id ? `Clear Filter` : `Inspect ${criticalWarning.lot_id}`}
            </button>
          </div>
        </div>
      )}

      {/* Lot-by-Lot Breakdown Cards */}
      <div className="panel">
        <div className="panelHead">
          <div>
            <div className="panelTitle">Lot-Level Distribution & Drift Intelligence</div>
            <div className="smallcaps" style={{ marginTop: 4 }}>
              Population-level variance, out-of-distribution drift and escalation burden across {data.total_lots || lots.length} lots
            </div>
          </div>
          <div className="actions">
            {activeLot && (
              <button className="btn ghost activeLotTag" onClick={() => onSelectLot(null)}>
                Filter: <strong className="mono">{activeLot}</strong>
                <X size={12} />
              </button>
            )}
            <button className="btn ghost" onClick={downloadLotCsv} title="Export lot breakdown to CSV">
              <Download size={13} /> Export Lot Summary (CSV)
            </button>
          </div>
        </div>
        <div className="panelBody">
          <div className="lotGrid">
            {lots.map((lot) => {
              const isSelected = activeLot === lot.lot_id
              const hasCritical = lot.status === 'DRIFT_ALERT'
              const hasCaution = lot.status === 'CAUTION'
              const safePct = lot.safe_pct ?? ((lot.safe_rate ?? 0) > 1 ? (lot.safe_rate ?? 0) : (lot.safe_rate ?? 0) * 100)
              const reviewPct = lot.review_pct ?? ((lot.review_rate ?? 0) > 1 ? (lot.review_rate ?? 0) : (lot.review_rate ?? 0) * 100)
              const rejectPct = lot.reject_pct ?? ((lot.reject_rate ?? 0) > 1 ? (lot.reject_rate ?? 0) : (lot.reject_rate ?? 0) * 100)

              return (
                <div
                  key={lot.lot_id}
                  className={`lotCard ${isSelected ? 'active' : ''} ${
                    hasCritical ? 'driftAlert' : hasCaution ? 'caution' : 'healthy'
                  }`}
                  onClick={() => onSelectLot(isSelected ? null : lot.lot_id)}
                  title={`Click to filter components by ${lot.lot_id}`}
                >
                  <div className="lotCardHeader">
                    <div className="lotCardName mono">{lot.lot_id}</div>
                    <span
                      className={`badge ${
                        hasCritical ? 'reject' : hasCaution ? 'review' : 'safe'
                      }`}
                    >
                      {(lot.status || 'HEALTHY').replace('_', ' ')}
                    </span>
                  </div>

                  <div className="lotCardPartsRow">
                    <span>{lot.total_parts ?? lot.total_components} parts screened</span>
                    <span className="mono">Mean Risk: {lot.mean_risk_score ?? lot.mean_risk}</span>
                  </div>

                  <div className="lotTrack">
                    <i className="safe" style={{ width: `${safePct}%` }} title={`Safe: ${lot.safe_pct}%`} />
                    <i className="review" style={{ width: `${reviewPct}%` }} title={`Review: ${lot.review_pct}%`} />
                    <i className="reject" style={{ width: `${rejectPct}%` }} title={`Reject: ${lot.reject_pct}%`} />
                    <i className="unknown" style={{ width: `${lot.unknown_pct ?? 0}%` }} title={`Unknown: ${lot.unknown_pct}%`} />
                  </div>

                  <div className="lotCardFooter">
                    <div className="lotCardBreakdown">
                      <span className="safeDot">{lot.safe_count ?? lot.disposition_counts?.['SAFE'] ?? 0}S</span>
                      <span className="reviewDot">{lot.review_count ?? lot.disposition_counts?.['REVIEW'] ?? 0}Rev</span>
                      <span className="rejectDot">{lot.reject_count ?? lot.disposition_counts?.['REJECT'] ?? 0}Rej</span>
                      {(lot.unknown_count ?? 0) > 0 && <span className="unknownDot">{lot.unknown_count}U</span>}
                    </div>
                    <div className="lotCardOod mono" title="Out-of-Distribution Rate">
                      OOD: {lot.ood_rate_pct ?? 0}%
                    </div>
                  </div>

                  {lot.diagnostic && (
                    <div className="lotCardDiag" title={lot.diagnostic}>
                      {lot.diagnostic}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
