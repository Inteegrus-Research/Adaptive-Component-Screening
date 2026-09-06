import { EvidenceChannel } from '../../types/api'
import { num } from '../../utils/format'
export function EvidenceStack({ channels }: { channels: EvidenceChannel[] }) { return <div className="evidence">{channels.map(c => <div className="evidenceRow" key={c.name}><div><div className="evidenceLabel">{c.name}</div>{c.level && <div className="evidenceLevel">{c.level}</div>}</div><div className="evidenceBar"><i style={{ width: `${Math.max(0, Math.min(1, c.score ?? 0)) * 100}%` }} /></div><div className="evidenceScore">{c.score == null ? '—' : num(c.score, 2)}</div></div>)}</div> }
