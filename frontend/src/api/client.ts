const base = ''

async function request<T>(path: string, init?: RequestInit, timeoutMs = 30000): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${base}${path}`, { ...init, signal: controller.signal })
    const text = await response.text()
    let body: any = {}
    try { body = text ? JSON.parse(text) : {} } catch { body = { detail: text } }
    if (!response.ok) throw new Error(body?.detail || `Request failed (${response.status})`)
    return body as T
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw new Error('Request timed out. Check the screening service and try again.')
    if (e instanceof TypeError) throw new Error('Screening service is unreachable. Start FastAPI and try again.')
    throw e
  } finally { window.clearTimeout(timer) }
}

export const apiGet = <T,>(path: string) => request<T>(path)
export const apiPostFile = <T,>(path: string, file: File, asOf = 24, horizon = 168, persist = true) => {
  const fd = new FormData(); fd.append('file', file)
  return request<T>(`${path}?as_of_h=${encodeURIComponent(asOf)}&target_horizon=${encodeURIComponent(horizon)}&persist=${persist ? 'true' : 'false'}`, { method: 'POST', body: fd }, 120000)
}
