import { Decision } from '../../types/api'
export function StatusBadge({ decision, large = false }: { decision: Decision; large?: boolean }) { const cls = decision.toLowerCase(); return <span className={`badge ${cls} ${large ? 'large' : ''}`}>{decision}</span> }
