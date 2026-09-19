import { useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Printer, Download, ShieldCheck, FileText, Lock, Award } from 'lucide-react'
import { jsPDF } from 'jspdf'
import type { ComplianceCertificate } from '../../types/api'
import { useApi } from '../../hooks/useApi'
import { num } from '../../utils/format'

export function ComplianceCertificateModal({
  partId,
  source = 'active',
  onClose,
}: {
  partId: string
  source?: string
  onClose: () => void
}) {
  const resolvedSource = source === 'demo' ? 'demo' : source === 'active' ? 'final_submission' : source || 'final_submission'
  const sourceQ = `?source=${encodeURIComponent(resolvedSource)}`
  const { data: cert, loading, error } = useApi<ComplianceCertificate>(
    `/api/components/${encodeURIComponent(partId)}/certificate${sourceQ}`
  )

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  function handlePrint() {
    if (!cert) return

    const pdf = new jsPDF({ unit: 'pt', format: 'a4' })
    const pageWidth = pdf.internal.pageSize.getWidth()
    const pageHeight = pdf.internal.pageSize.getHeight()
    let y = 52

    pdf.setFillColor(14, 22, 34)
    pdf.rect(0, 0, pageWidth, 54, 'F')
    pdf.setTextColor(255, 255, 255)
    pdf.setFontSize(18)
    pdf.text('ACS Compliance Report', 40, 32)
    pdf.setFontSize(9)
    pdf.text(`Part ID: ${cert.component.part_id}   Lot: ${cert.component.lot_id}`, 40, 46)

    pdf.setTextColor(17, 24, 39)
    pdf.setFontSize(13)
    y += 24
    pdf.text('Disposition', 40, y)
    y += 18
    pdf.setFontSize(18)
    pdf.text(cert.disposition.title, 40, y)
    y += 18
    pdf.setFontSize(10)
    pdf.setTextColor(71, 85, 105)
    const lines = pdf.splitTextToSize(cert.disposition.acceptance_text || '', 520)
    pdf.text(lines, 40, y)
    y += lines.length * 14 + 20

    const boxes = [
      ['Component family', cert.component.component_family],
      ['Parameter', cert.component.parameter],
      ['Risk score', cert.disposition.risk_score == null ? '—' : String(Number(cert.disposition.risk_score).toFixed(3))],
      ['OOD status', cert.disposition.ood_status],
      ['168h prediction', String(cert.forecast.prediction_168h ?? '—')],
      ['Audit hash', cert.audit_hash],
    ]

    let boxX = 40
    let boxY = y
    const boxW = 230
    const boxH = 52
    for (let i = 0; i < boxes.length; i += 1) {
      const [label, value] = boxes[i]
      pdf.setDrawColor(214, 219, 226)
      pdf.setFillColor(248, 250, 252)
      pdf.roundedRect(boxX, boxY, boxW, boxH, 6, 6, 'FD')
      pdf.setTextColor(100, 116, 139)
      pdf.setFontSize(8)
      pdf.text(label.toUpperCase(), boxX + 10, boxY + 16)
      pdf.setTextColor(17, 24, 39)
      pdf.setFontSize(10)
      pdf.text(String(value), boxX + 10, boxY + 32)
      if (boxX + boxW * 2 > pageWidth - 40) {
        boxX = 40
        boxY += boxH + 10
      } else {
        boxX += boxW + 12
      }
    }
    y = boxY + 72

    pdf.setTextColor(17, 24, 39)
    pdf.setFontSize(12)
    pdf.text('Evidence scorecard', 40, y)
    y += 18

    pdf.setFontSize(9)
    const tableHeaders = ['Channel', 'Level', 'Score', 'Detail']
    const rows = cert.evidence_scorecard.map((ev) => [
      ev.channel,
      ev.level,
      ev.score == null ? '—' : Number(ev.score).toFixed(3),
      ev.detail || 'Normal operation.'
    ])
    const colWidths = [130, 64, 54, 260]
    let rowY = y
    pdf.setDrawColor(214, 219, 226)
    pdf.setFillColor(241, 245, 249)
    pdf.rect(40, rowY - 12, pageWidth - 80, 18, 'F')
    tableHeaders.forEach((header, index) => {
      pdf.text(header, 48 + colWidths.slice(0, index).reduce((a, b) => a + b, 0), rowY)
    })
    rowY += 18

    rows.forEach((row) => {
      if (rowY > pageHeight - 80) {
        pdf.addPage()
        rowY = 52
      }
      pdf.setDrawColor(214, 219, 226)
      pdf.rect(40, rowY, pageWidth - 80, 18)
      row.forEach((cell, index) => {
        pdf.text(String(cell), 48 + colWidths.slice(0, index).reduce((a, b) => a + b, 0), rowY + 12)
      })
      rowY += 18
    })

    pdf.save(`ACS_REPORT_${cert.component.part_id}.pdf`)
  }

  function downloadJson() {
    if (!cert) return
    const blob = new Blob([JSON.stringify(cert, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ACS_CERT_${cert.component.part_id}_${cert.audit_hash.slice(0, 8)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  function downloadCsv() {
    if (!cert) return
    const rows = [
      ['FIELD', 'VALUE'],
      ['CERTIFICATE_ID', cert.certificate_id],
      ['AUDIT_HASH', cert.audit_hash],
      ['ISSUED_UTC', cert.issued_utc],
      ['STANDARD', cert.standard],
      ['PART_ID', cert.component.part_id],
      ['LOT_ID', cert.component.lot_id],
      ['COMPONENT_FAMILY', cert.component.component_family],
      ['PARAMETER', cert.component.parameter],
      ['SCREENING_ORIGIN_H', String(cert.component.screening_origin_h)],
      ['TARGET_HORIZON_H', String(cert.component.target_horizon_h)],
      ['FINAL_DISPOSITION', cert.disposition.decision],
      ['DISPOSITION_TITLE', cert.disposition.title],
      ['RISK_SCORE', String(cert.disposition.risk_score ?? '')],
      ['OOD_STATUS', cert.disposition.ood_status],
      ['168H_PREDICTION', String(cert.forecast.prediction_168h ?? '')],
      ['CONFORMAL_LOWER', String(cert.forecast.prediction_lower ?? '')],
      ['CONFORMAL_UPPER', String(cert.forecast.prediction_upper ?? '')],
      ['ENGINEERING_LIMIT', String(cert.forecast.engineering_limit_upper ?? '')],
      ['LIMIT_EXCEEDANCE_PROJECTED', String(cert.forecast.limit_exceedance_projected)],
    ]
    const csvContent = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ACS_TRAVELER_${cert.component.part_id}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <AnimatePresence>
      <div className="certModalOverlay" onClick={onClose}>
        <motion.div
          className="certModalContainer"
          initial={{ opacity: 0, scale: 0.95, y: 15 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 15 }}
          transition={{ duration: 0.2 }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="certActionBar no-print">
            <div className="certActionTitle">
              <Award size={16} color="var(--accent2)" />
              <span>FLIGHT QUALIFICATION AUDIT TRAVELER</span>
              <span className="mono badge unknown">{partId}</span>
            </div>
            <div className="actions">
              <button className="btn ghost" onClick={downloadCsv} title="Download Disposition CSV">
                <Download size={13} /> Export CSV
              </button>
              <button className="btn ghost" onClick={downloadJson} title="Download Full Machine-Readable JSON Trace">
                <FileText size={13} /> JSON Audit Pack
              </button>
              <button className="btn primary" onClick={handlePrint} title="Save official PDF report">
                <Printer size={13} /> Save PDF
              </button>
              <button className="iconButton" onClick={onClose} aria-label="Close modal">
                <X size={16} />
              </button>
            </div>
          </div>

          <div className="certSheet" id="printable-certificate">
            <div className="certWatermark">ACS FLIGHT QUALIFIED</div>
            <div className="certHeader">
              <div className="certHeaderBrand">
                <div className="certEmblem">ACS</div>
                <div>
                  <div className="certOrgTitle">ADAPTIVE COMPONENT SCREENING INTELLIGENCE</div>
                  <div className="certDocTitle">FLIGHT SCREENING DISPOSITION TRAVELER SHEET</div>
                  <div className="certStandardRef">
                    REF: MIL-STD-883K TM 5004.14 (CLASS B/S) · ESA ESCC 9000 · AEC-Q100 GRADE 1
                  </div>
                </div>
              </div>
              <div className="certSecurityBadge">
                <div className="smallcaps">AUDIT FINGERPRINT</div>
                <div className="mono certHash">{cert ? cert.audit_hash : 'GENERATING…'}</div>
                <div className="certSecurityVerified">
                  <Lock size={10} /> TAMPER-EVIDENT RECORD
                </div>
              </div>
            </div>

            {loading && <div className="loadingBand">Generating certified compliance disposition…</div>}
            {error && <div className="error">{error}</div>}

            {cert && (
              <>
                <div className="certIdGrid">
                  <div className="certIdBox">
                    <span className="certIdLabel">PART SERIAL / ID</span>
                    <strong className="mono certIdValue">{cert.component.part_id}</strong>
                  </div>
                  <div className="certIdBox">
                    <span className="certIdLabel">WAFER / LOT NUMBER</span>
                    <strong className="mono certIdValue">{cert.component.lot_id}</strong>
                  </div>
                  <div className="certIdBox">
                    <span className="certIdLabel">COMPONENT FAMILY</span>
                    <strong className="certIdValue">{cert.component.component_family}</strong>
                  </div>
                  <div className="certIdBox">
                    <span className="certIdLabel">SCREENED PARAMETER</span>
                    <strong className="mono certIdValue">
                      {cert.component.parameter} {cert.component.unit ? `(${cert.component.unit})` : ''}
                    </strong>
                  </div>
                  <div className="certIdBox">
                    <span className="certIdLabel">SCREENING ORIGIN</span>
                    <strong className="mono certIdValue">{num(cert.component.screening_origin_h, 0)} h burn-in</strong>
                  </div>
                  <div className="certIdBox">
                    <span className="certIdLabel">PROJECTED HORIZON</span>
                    <strong className="mono certIdValue">{num(cert.component.target_horizon_h, 0)} h mission</strong>
                  </div>
                </div>

                <div className={`certStampBlock ${cert.disposition.stamp_color}`}>
                  <div className="certStampContent">
                    <div className="certStampTop">
                      <span className="smallcaps">OFFICIAL AUTONOMOUS SAFETY GATE DISPOSITION</span>
                      <span className="mono certDate">{cert.issued_utc}</span>
                    </div>
                    <div className="certStampTitle">{cert.disposition.title}</div>
                    <p className="certStampDesc">{cert.disposition.acceptance_text}</p>
                  </div>
                  <div className="certStampMetrics">
                    <div>
                      <span className="smallcaps">DISPOSITION RISK</span>
                      <strong className="mono">
                        {cert.disposition.risk_score == null ? '—' : Number(cert.disposition.risk_score).toFixed(3)}
                      </strong>
                    </div>
                    <div>
                      <span className="smallcaps">DOMAIN TRUST</span>
                      <strong className="mono">{cert.disposition.ood_status}</strong>
                    </div>
                    <div>
                      <span className="smallcaps">CONFIDENCE</span>
                      <strong>{cert.disposition.confidence}</strong>
                    </div>
                  </div>
                </div>

                <div className="certSectionTitle">
                  <span>01 · INDEPENDENT EVIDENCE SCORECARD</span>
                  <span className="smallcaps">LEAKAGE-SAFE ISOLATION ENFORCED</span>
                </div>
                <table className="certTable">
                  <thead>
                    <tr>
                      <th>Evidence Channel</th>
                      <th>Methodology</th>
                      <th>Raw Score</th>
                      <th>Operational Level</th>
                      <th>Channel Diagnostic</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cert.evidence_scorecard.map((ev, i) => (
                      <tr key={i}>
                        <td className="boldText">{ev.channel}</td>
                        <td className="smallText">
                          {ev.channel.includes('Population')
                            ? 'Part Average Testing (PAT) Robust Dispersion'
                            : ev.channel.includes('Temporal')
                            ? 'Horizon-Isolated dV/dt Acceleration'
                            : ev.channel.includes('Multivariate')
                            ? 'Cross-Parameter Manifold Distance'
                            : ev.channel.includes('Absolute')
                            ? 'Non-Overridable Engineering Bounds'
                            : 'OOD Reference Domain Profile'}
                        </td>
                        <td className="mono boldText">
                          {ev.score == null ? '—' : Number(ev.score).toFixed(3)}
                        </td>
                        <td>
                          <span
                            className={`badge ${
                              ev.level === 'ELEVATED'
                                ? 'review'
                                : ev.level === 'CRITICAL'
                                ? 'reject'
                                : 'safe'
                            }`}
                          >
                            {ev.level}
                          </span>
                        </td>
                        <td className="smallText">{ev.detail || 'Normal operation.'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* Conformal Drift Forecast & Boundary Margin */}
                <div className="certSectionTitle" style={{ marginTop: 14 }}>
                  <span>02 · HORIZON DRIFT FORECAST & CONFORMAL UNCERTAINTY</span>
                  <span className="smallcaps">TARGET HORIZON: 168 HOURS</span>
                </div>
                <div className="certForecastGrid">
                  <div className="certForecastBox">
                    <span className="smallcaps">0h Baseline</span>
                    <strong className="mono">
                      {cert.component.observed_value_0h == null
                        ? '—'
                        : `${num(cert.component.observed_value_0h, 3)} ${cert.component.unit}`}
                    </strong>
                  </div>
                  <div className="certForecastBox">
                    <span className="smallcaps">24h Screening As-Of</span>
                    <strong className="mono">
                      {cert.component.observed_value_asof == null
                        ? '—'
                        : `${num(cert.component.observed_value_asof, 3)} ${cert.component.unit}`}
                    </strong>
                  </div>
                  <div className="certForecastBox highlight">
                    <span className="smallcaps">168h Forecast Prediction</span>
                    <strong className="mono">
                      {cert.forecast.prediction_168h == null
                        ? '—'
                        : `${num(cert.forecast.prediction_168h, 3)} ${cert.component.unit}`}
                    </strong>
                    <span className="certModelTag">Model: {cert.forecast.selected_model}</span>
                  </div>
                  <div className="certForecastBox">
                    <span className="smallcaps">Conformal 95% Interval</span>
                    <strong className="mono">
                      [{num(cert.forecast.prediction_lower, 3)}, {num(cert.forecast.prediction_upper, 3)}]
                    </strong>
                    <span className="smallText">Half-width: ±{num(cert.forecast.conformal_half_width, 3)}</span>
                  </div>
                  <div className="certForecastBox">
                    <span className="smallcaps">Authoritative Limit</span>
                    <strong className="mono">
                      {cert.forecast.engineering_limit_upper == null
                        ? 'N/A'
                        : `${num(cert.forecast.engineering_limit_upper, 3)} ${cert.component.unit}`}
                    </strong>
                    <span
                      className={`badge ${
                        cert.forecast.limit_exceedance_projected ? 'reject' : 'safe'
                      }`}
                    >
                      {cert.forecast.limit_exceedance_projected ? 'EXCEEDANCE PROJECTED' : 'WITHIN SPEC'}
                    </span>
                  </div>
                </div>

                {/* Audit Reasoning & Counterfactual Sensitivity */}
                <div className="certSectionTitle" style={{ marginTop: 14 }}>
                  <span>03 · AUDITABLE REASONING & COUNTERFACTUAL BOUNDARIES</span>
                </div>
                <div className="certReasoningGrid">
                  <div className="certReasoningBox">
                    <h4>Primary Observations & Findings</h4>
                    <ul>
                      {cert.explanation.facts?.slice(0, 3).map((f, idx) => (
                        <li key={idx}>{String(f)}</li>
                      ))}
                      {cert.explanation.model_findings?.slice(0, 2).map((mf, idx) => (
                        <li key={idx}>{String(mf)}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="certReasoningBox">
                    <h4>Counterfactual Sensitivity</h4>
                    <div className="certCounterfactualCallout">
                      <div className="smallcaps">CONDITION REQUIRED TO OVERTURN DISPOSITION:</div>
                      <p>
                        {Array.isArray(cert.explanation.counterfactual)
                          ? cert.explanation.counterfactual[0] ||
                            'No safe counterfactual path available within nominal parameter bounds.'
                          : String(cert.explanation.counterfactual)}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Sign-off & Authority Block */}
                <div className="certSignBlock">
                  <div className="certSignCol">
                    <div className="smallcaps">SAFETY GATE ENGINE</div>
                    <strong>{cert.verification.policy_engine}</strong>
                    <div className="certSealMark">
                      <ShieldCheck size={14} color="var(--safe)" />
                      <span>DATA INTEGRITY VERIFIED</span>
                    </div>
                  </div>
                  <div className="certSignCol">
                    <div className="smallcaps">STATUS & AUTHENTICITY</div>
                    <strong className="mono">{cert.verification.status} · AUDITED</strong>
                    <div className="smallText">Non-repudiation SHA-256 hash sealed in local audit ledger.</div>
                  </div>
                  <div className="certSignCol">
                    <div className="smallcaps">QA / CHIEF SCREENING INSPECTOR</div>
                    <div className="certSigLine">ELECTRONICALLY VALIDATED VIA ACS GATE</div>
                    <div className="mono smallText">TS: {cert.issued_utc}</div>
                  </div>
                </div>
              </>
            )}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  )
}
