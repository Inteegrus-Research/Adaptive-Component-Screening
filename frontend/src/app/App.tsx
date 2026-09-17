import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  Award,
  ChevronRight,
  ClipboardCheck,
  Cpu,
  Database,
  FileSearch,
  Layers3,
  Plus,
  PlusCircle,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  UploadCloud,
  X,
} from 'lucide-react'
import { DashboardPage } from '../pages/Dashboard'
import { ComponentPage } from '../pages/ComponentIntelligence'
import { EvidencePage } from '../pages/EvidenceExplainability'
import { AuditPage } from '../pages/AuditBenchmark'
import { NewScreeningPage } from '../pages/NewScreening'
import { useApi } from '../hooks/useApi'
import type { ComponentIntelligence, ComponentSummary, ScreeningResponse, SystemStatus } from '../types/api'
import { StatusBadge } from '../components/common/StatusBadge'
import { parseList } from '../utils/format'
import { ComplianceCertificateModal } from '../components/common/ComplianceCertificateModal'


type View = 'overview' | 'component' | 'evidence' | 'audit' | 'intake'
type SourceMode = 'active' | 'demo'

const NAV: { id: Exclude<View, 'intake'>; label: string; compact: string; icon: ReactNode }[] = [
  { id: 'overview', label: 'Overview', compact: 'Overview', icon: <Layers3 size={19} /> },
  { id: 'component', label: 'Component Passport', compact: 'Component Passport', icon: <Cpu size={19} /> },
  { id: 'evidence', label: 'Evidence', compact: 'Evidence', icon: <FileSearch size={19} /> },
  { id: 'audit', label: 'Audit', compact: 'Audit', icon: <Award size={19} /> },
]

export function App() {
  const [view, setView] = useState<View>('overview')
  const [source, setSource] = useState<SourceMode>('demo')
  const [selected, setSelected] = useState<string | null>(null)
  const [session, setSession] = useState<ScreeningResponse | null>(null)
  const [whyOpen, setWhyOpen] = useState(false)

  const sourceQ = source === 'demo' ? '?source=demo' : ''
  const components = useApi<{ items: ComponentSummary[]; count: number }>(
    `/api/components?limit=500${source === 'demo' ? '&source=demo' : ''}`
  )
  const health = useApi<SystemStatus>(`/api/health${sourceQ}`)
  const validation = useApi<any>(`/api/validation${sourceQ}`)
  const packet = useApi<ComponentIntelligence>(
    selected ? `/api/components/${encodeURIComponent(selected)}${source === 'demo' ? '?source=demo' : ''}` : null
  )

  const preferred = useMemo(
    () =>
      [...(components.data?.items ?? [])].sort(
        (a, b) => (b.risk_score ?? -1) - (a.risk_score ?? -1)
      )[0]?.part_id ?? null,
    [components.data]
  )

  useEffect(() => {
    if (selected && components.data?.items?.every((c) => c.part_id !== selected)) {
      setSelected(null)
    }
  }, [components.data, selected])

  useEffect(() => {
    if (!selected && preferred) setSelected(preferred)
  }, [preferred, selected])

  const openComponent = (id: string) => {
    setSelected(id)
    setView('component')
  }

  const navigate = (next: View) => {
    if ((next === 'component' || next === 'evidence') && !selected && preferred) {
      setSelected(preferred)
    }
    setView(next)
  }

  const selectedSummary = components.data?.items.find((x) => x.part_id === selected)
  const activeLabel =
    view === 'intake' ? 'New Screening' : NAV.find((x) => x.id === view)?.compact ?? 'Overview'

  const reloadAll = () => {
    components.reload()
    health.reload()
    validation.reload()
  }

  const completeRun = (result: ScreeningResponse, mode: SourceMode) => {
    setSession(result)
    setSource(mode)
    setSelected(result.components[0]?.part_id ?? null)
    setView('overview')
    components.reload()
    health.reload()
    validation.reload()
  }

  const dashboardSession =
    source === 'demo' && session?.run_id === 'demo_scenario'
      ? session
      : session?.run_id?.startsWith('live_')
      ? session
      : null

  return (
    <div className="app">
      {/* Aerospace Mission Control Rail Navigation */}
      <aside className="rail" aria-label="Aerospace mission navigation">
        <div className="brandMark" title="Adaptive Component Screening — Mission Control">
          ACS
          <span>AERO</span>
        </div>
        <div className="railCaption">MISSION</div>
        <div className="railNav">
          {NAV.map((item) => (
            <NavButton
              key={item.id}
              active={view === item.id}
              title={item.label}
              onClick={() => navigate(item.id)}
            >
              {item.icon}
            </NavButton>
          ))}
          <NavButton
            active={view === 'intake'}
            title="New Screening"
            onClick={() => navigate('intake')}
          >
            <PlusCircle size={19} />
          </NavButton>
        </div>
        <div className="railFooter">ACS</div>
      </aside>

      {/* Main Mission View Area */}
      <main className="main">
        <header className="topbar">
          <div className="topIdentity">
            <div className="eyebrow" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>ACS</span> · <span>ADAPTIVE COMPONENT SCREENING</span>
            </div>
            <div className="title">SCREENING OVERVIEW</div>
          </div>

          <div className="topContext">
            <div className="contextPath">
              <span>ACS</span>
              <ChevronRight size={11} />
              <span style={{ color: 'var(--accent)', fontWeight: 600 }}>{activeLabel}</span>
              {selectedSummary && view !== 'overview' && view !== 'intake' && (
                <>
                  <ChevronRight size={11} />
                  <span className="mono">{selectedSummary.part_id}</span>
                </>
              )}
            </div>

            <div className="topstate">
              <button className="btn topNew" onClick={() => setView('intake')}>
                <UploadCloud size={13} /> NEW SCREENING
              </button>

              <div className="statusChip">
                <span className={`dot ${health.data?.status === 'ok' ? '' : 'alert'}`} />
                <span style={{ color: 'var(--ink)' }}>READY</span>
              </div>

              <div className="statusChip">
                <ShieldCheck size={12} color="var(--safe)" />
                <span style={{ color: validation.data?.ok ? 'var(--safe)' : 'var(--review)' }}>
                  {validation.data?.ok ? 'VALIDATED' : 'CHECK'}
                </span>
              </div>

              <button className="iconButton" title="Refresh telemetry" onClick={reloadAll}>
                <RefreshCw size={13} />
              </button>
            </div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {view === 'overview' && (
            <ViewFrame keyValue={`overview-${source}`}>
              <DashboardPage
                onOpen={openComponent}
                onNewRun={() => setView('intake')}
                source={source}
                session={dashboardSession}
              />
            </ViewFrame>
          )}

          {view === 'component' && selected && (
            <ViewFrame keyValue={`component-${source}-${selected}`}>
              <ComponentPage
                partId={selected}
                source={source}
                onBack={() => navigate('overview')}
                onWhy={() => setWhyOpen(true)}
                onOpenEvidence={() => navigate('evidence')}
              />
            </ViewFrame>
          )}

          {view === 'evidence' && selected && (
            <ViewFrame keyValue={`evidence-${source}-${selected}`}>
              <EvidencePage
                partId={selected}
                source={source}
                onBack={() => navigate('component')}
                onOpenComponent={() => navigate('component')}
              />
            </ViewFrame>
          )}

          {view === 'audit' && (
            <ViewFrame keyValue={`audit-${source}`}>
              <AuditPage source={source} onInspect={openComponent} />
            </ViewFrame>
          )}

          {view === 'intake' && (
            <ViewFrame keyValue="intake">
              <NewScreeningPage onComplete={completeRun} onCancel={() => navigate('overview')} />
            </ViewFrame>
          )}

          {(view === 'component' || view === 'evidence') && !selected && (
            <ViewFrame keyValue="empty">
              <div className="page">
                <div className="emptyState panel">
                  <ShieldAlert size={20} color="var(--accent)" />
                  <div>
                    <div className="panelTitle">No Component Selected</div>
                    <p>Select a telemetry record from the Screening Matrix to inspect Digital Passport.</p>
                  </div>
                </div>
              </div>
            </ViewFrame>
          )}
        </AnimatePresence>

        {whyOpen && packet.data && <WhyDrawer data={packet.data} onClose={() => setWhyOpen(false)} source={source} />}
      </main>
    </div>
  )
}

function ViewFrame({ keyValue, children }: { keyValue: string; children: ReactNode }) {
  return (
    <motion.div
      key={keyValue}
      className="viewTransition"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      transition={{ duration: 0.16 }}
    >
      {children}
    </motion.div>
  )
}

function NavButton({
  active,
  title,
  onClick,
  children,
}: {
  active: boolean
  title: string
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      className={`railButton ${active ? 'active' : ''}`}
      title={title}
      aria-label={title}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

function WhyDrawer({ data, onClose, source = 'demo' }: { data: ComponentIntelligence; onClose: () => void; source?: 'active' | 'demo' }) {
  const [certModalOpen, setCertModalOpen] = useState(false)
  const c = data.component
  const e = data.explanation || {}
  const facts = parseList(e.facts)
  const findings = parseList(e.model_findings)
  const policy = parseList(e.policy_reasoning)
  const cf = parseList(e.counterfactuals)

  return (
    <>
      {certModalOpen && <ComplianceCertificateModal partId={c.part_id} source={source} onClose={() => setCertModalOpen(false)} />}
      <motion.div
        className="drawerOverlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      />
      <motion.aside
        className="drawer"
        initial={{ x: 600 }}
        animate={{ x: 0 }}
        exit={{ x: 600 }}
        transition={{ type: 'spring', stiffness: 350, damping: 32 }}
      >
        <div className="drawerHead">
          <div>
            <div className="eyebrow">ENGINEERING EVIDENCE RECORD</div>
            <div className="drawerTitle" style={{ color: 'var(--ink)' }}>
              WHY THIS DECISION?
            </div>
            <div className="mono drawerPart" style={{ color: 'var(--accent)' }}>
              {c.part_id}
            </div>
          </div>
          <button className="close" onClick={onClose}>
            <X size={17} />
          </button>
        </div>
        <div className="drawerDecision">
          <StatusBadge decision={c.decision} large />
          <span className="smallcaps">
            RISK <b style={{ color: 'var(--ink)' }}>{c.risk_score == null ? '—' : Number(c.risk_score).toFixed(2)}</b>
          </span>
          <span className="smallcaps">
            OOD <b style={{ color: 'var(--ink)' }}>{c.ood_status}</b>
          </span>
        </div>
        <div style={{ margin: '14px 0', padding: '12px', background: 'rgba(0, 240, 255, 0.08)', border: '1px solid var(--line)', borderRadius: 8 }}>
          <button className="btn primary" style={{ width: '100%', justifyContent: 'center' }} onClick={() => setCertModalOpen(true)}>
            <Award size={14} /> OFFICIAL FLIGHT TRAVELER CERTIFICATE
          </button>
        </div>
        <DrawerSection n="01" title="OBSERVATION" lines={facts.slice(0, 6)} />
        <DrawerSection n="02" title="INFERENCE" lines={findings.slice(0, 6)} />
        <DrawerSection n="03" title="DECISION POLICY" lines={policy.slice(0, 5)} />
        <DrawerSection
          n="04"
          title="WHY NOT AUTOMATIC REJECT?"
          lines={
            c.decision === 'REVIEW'
              ? [
                  'Current evidence is routed to engineering review rather than unconditional rejection.',
                  'Forecast evidence is interval-aware and remains subject to epistemic uncertainty.',
                ]
              : [
                  'The persisted safety disposition does not require an automatic reject narrative beyond its recorded policy state.',
                ]
          }
        />
        <DrawerSection n="05" title="WHAT WOULD CHANGE THE DECISION?" lines={cf.slice(0, 4)} />
      </motion.aside>
    </>
  )
}

function DrawerSection({ n, title, lines }: { n: string; title: string; lines: string[] }) {
  return (
    <section className="whySection">
      <div className="whyNumber" style={{ color: 'var(--accent)' }}>
        {n}
      </div>
      <div>
        <h4 style={{ color: 'var(--soft)' }}>{title}</h4>
        <ul>
          {(lines.length ? lines : ['No persisted narrative evidence available.']).map((x, i) => (
            <li key={i}>{x}</li>
          ))}
        </ul>
      </div>
    </section>
  )
}
