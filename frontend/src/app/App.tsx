import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  ClipboardCheck,
  Database,
  FileSearch,
  Layers3,
  ShieldCheck,
  X,
  Command,
  ChevronRight,
} from 'lucide-react'
import { DashboardPage } from '../pages/Dashboard'
import { ComponentPage } from '../pages/ComponentIntelligence'
import { EvidencePage } from '../pages/EvidenceExplainability'
import { AuditPage } from '../pages/AuditBenchmark'
import { useApi } from '../hooks/useApi'
import type { ComponentSummary } from '../types/api'
import { StatusBadge } from '../components/common/StatusBadge'
import { parseList, num } from '../utils/format'

type View = 'overview' | 'component' | 'evidence' | 'audit'

const NAV: { id: View; label: string; compact: string; icon: ReactNode }[] = [
  { id: 'overview', label: 'Screening Overview', compact: 'Overview', icon: <Layers3 size={18} /> },
  { id: 'component', label: 'Component Intelligence', compact: 'Component', icon: <Activity size={18} /> },
  { id: 'evidence', label: 'Evidence & Explainability', compact: 'Evidence', icon: <FileSearch size={18} /> },
  { id: 'audit', label: 'Audit / Benchmark', compact: 'Audit', icon: <ClipboardCheck size={18} /> },
]

export function App() {
  const [view, setView] = useState<View>('overview')
  const [selected, setSelected] = useState<string | null>(null)
  const [whyOpen, setWhyOpen] = useState(false)
  const components = useApi<{ items: ComponentSummary[]; count: number }>('/api/components?limit=250')
  const health = useApi<{ status: string; pipeline: string; models: string; demo_mode: boolean }>('/api/health')
  const componentPacket = useApi<any>(selected ? `/api/components/${encodeURIComponent(selected)}` : null)

  const preferred = useMemo(() => {
    const rows = components.data?.items ?? []
    return [...rows]
      .sort((a, b) => (b.risk_score ?? -1) - (a.risk_score ?? -1))[0]?.part_id ?? null
  }, [components.data])

  useEffect(() => {
    if (!selected && preferred) setSelected(preferred)
  }, [preferred, selected])

  const openComponent = (id: string) => {
    setSelected(id)
    setView('component')
  }

  const openWhy = (id: string) => {
    setSelected(id)
    setWhyOpen(true)
  }

  const navigate = (next: View) => {
    if ((next === 'component' || next === 'evidence') && !selected && preferred) {
      setSelected(preferred)
    }
    setView(next)
  }

  const selectedSummary = components.data?.items.find(x => x.part_id === selected)
  const activeLabel = NAV.find(x => x.id === view)?.compact ?? 'Overview'

  return (
    <div className="app">
      <aside className="rail" aria-label="Primary navigation">
        <div className="brandMark">ACS</div>
        <div className="railCaption">BURN-IN / EEE</div>
        <div className="railNav">
          {NAV.map(item => (
            <NavButton
              key={item.id}
              active={view === item.id}
              title={item.label}
              onClick={() => navigate(item.id)}
            >
              {item.icon}
              <span className="railLabel">{item.compact}</span>
            </NavButton>
          ))}
        </div>
        <div className="railFooter">ENGINEERING INTELLIGENCE</div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topIdentity">
            <div className="eyebrow">ADAPTIVE COMPONENT SCREENING</div>
            <div className="title">EEE Burn-In Intelligence Console</div>
          </div>
          <div className="topContext">
            <div className="contextPath"><span>ACS</span><ChevronRight size={11} /><span>{activeLabel}</span>{selectedSummary && view !== 'overview' && <><ChevronRight size={11} /><span className="mono">{selectedSummary.part_id}</span></>}</div>
            <div className="topstate">
              <div className="statusChip"><span className="dot" /> PIPELINE READY</div>
              {health.data?.demo_mode && <div className="statusChip demoChip"><span className="demoPulse" /> DEMO SCENARIO</div>}
              <div className="statusChip"><Database size={12} /> MODEL ARTIFACTS</div>
            </div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {view === 'overview' && (
            <ViewFrame keyValue="overview">
              <DashboardPage onOpen={openComponent} onOpenEvidence={() => navigate('evidence')} />
            </ViewFrame>
          )}
          {view === 'component' && selected && (
            <ViewFrame keyValue="component">
              <ComponentPage partId={selected} onBack={() => navigate('overview')} onWhy={() => openWhy(selected)} onOpenEvidence={() => navigate('evidence')} />
            </ViewFrame>
          )}
          {view === 'evidence' && selected && (
            <ViewFrame keyValue="evidence">
              <EvidencePage partId={selected} onBack={() => navigate('component')} onOpenComponent={() => navigate('component')} />
            </ViewFrame>
          )}
          {view === 'audit' && (
            <ViewFrame keyValue="audit">
              <AuditPage onInspect={openComponent} />
            </ViewFrame>
          )}
          {((view === 'component' || view === 'evidence') && !selected) && (
            <ViewFrame keyValue="select"><EmptySelection onBack={() => navigate('overview')} /></ViewFrame>
          )}
        </AnimatePresence>
      </main>

      <AnimatePresence>
        {whyOpen && componentPacket.data && (
          <WhyDrawer data={componentPacket.data} onClose={() => setWhyOpen(false)} />
        )}
      </AnimatePresence>
    </div>
  )
}

function ViewFrame({ keyValue, children }: { keyValue: string; children: ReactNode }) {
  return (
    <motion.div
      key={keyValue}
      className="viewTransition"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.18 }}
    >
      {children}
    </motion.div>
  )
}

function NavButton({ active, title, onClick, children }: { active: boolean; title: string; onClick: () => void; children: ReactNode }) {
  return (
    <button className={`railButton ${active ? 'active' : ''}`} title={title} aria-label={title} onClick={onClick}>
      {children}
    </button>
  )
}

function EmptySelection({ onBack }: { onBack: () => void }) {
  return (
    <div className="page">
      <div className="emptyState panel">
        <ShieldCheck size={18} />
        <div>
          <div className="panelTitle">No component selected</div>
          <p>Select a component from Screening Overview to inspect its intelligence packet.</p>
        </div>
        <button className="btn primary" onClick={onBack}>Back to screening</button>
      </div>
    </div>
  )
}

function WhyDrawer({ data, onClose }: { data: any; onClose: () => void }) {
  const c = data.component
  const exp = data.explanation || {}
  const findings = parseList(exp.model_findings)
  const policy = parseList(exp.policy_reasoning)
  const facts = parseList(exp.facts)
  const cf = parseList(exp.counterfactuals)
  const evidence = Array.isArray(data.anomaly_evidence) ? data.anomaly_evidence : []
  const strongest = [...evidence].sort((a: any, b: any) => (b.score ?? -1) - (a.score ?? -1))[0]

  return (
    <>
      <motion.div className="drawerOverlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
      <motion.aside
        className="drawer"
        initial={{ x: 590 }} animate={{ x: 0 }} exit={{ x: 590 }}
        transition={{ type: 'spring', stiffness: 340, damping: 32 }}
      >
        <div className="drawerHead">
          <div>
            <div className="eyebrow">ENGINEERING EVIDENCE RECORD</div>
            <div className="drawerTitle">WHY THIS DECISION?</div>
            <div className="mono drawerPart">{c.part_id}</div>
          </div>
          <button className="close" onClick={onClose} aria-label="Close evidence drawer"><X size={17} /></button>
        </div>

        <div className="drawerDecision">
          <StatusBadge decision={c.decision} large />
          <span className="smallcaps">Risk <b className="mono">{c.risk_score == null ? '—' : Number(c.risk_score).toFixed(2)}</b></span>
          <span className="smallcaps">OOD <b>{c.ood_status}</b></span>
          {strongest && <span className="smallcaps">Primary <b>{strongest.name}</b></span>}
        </div>

        <WhySection n="01" title="OBSERVATION" lines={facts.length ? facts.slice(0, 6) : ['No persisted observation narrative is available.']} />
        <WhySection n="02" title="INFERENCE" lines={findings.length ? findings.slice(0, 6) : ['No persisted model findings are available.']} />
        <WhySection n="03" title="DECISION" lines={policy.length ? policy.slice(0, 5) : [exp.why_this_decision || 'No persisted policy narrative is available.']} />
        <WhySection
          n="04"
          title="WHY NOT AUTOMATIC REJECT?"
          lines={
            c.decision === 'REVIEW'
              ? [
                  data.decision?.hard_limit_violation ? 'A hard-limit condition is present; this is a safety-critical signal.' : 'Current engineering limits do not establish an unconditional hard reject.',
                  data.forecast?.predicted_limit_exceedance ? 'Future-limit evidence is interval-aware and is not treated as a causal failure claim.' : 'The forecast alone is not treated as an unconditional rejection rule.',
                ]
              : c.decision === 'REJECT'
                ? ['The persisted safety policy already supports REJECT for this component.', 'Human disposition remains part of downstream engineering handling.']
                : ['No autonomous rejection condition is present in the persisted decision trace.']
          }
        />
        <WhySection n="05" title="WHAT WOULD CHANGE THE DECISION?" lines={cf.length ? cf.slice(0, 3) : [exp.specific_counterfactual || 'No single evidence change was sufficient to define a safe counterfactual.']} />

        <div className="drawerNote">
          <Command size={13} />
          Persisted analytical evidence only. The interface does not invent an LLM-generated causal opinion.
        </div>
      </motion.aside>
    </>
  )
}

function WhySection({ n, title, lines }: { n: string; title: string; lines: string[] }) {
  return (
    <section className="whySection">
      <div className="whyNumber">{n}</div>
      <div>
        <h4>{title}</h4>
        <ul>{lines.map((x, i) => <li key={i}>{x}</li>)}</ul>
      </div>
    </section>
  )
}
