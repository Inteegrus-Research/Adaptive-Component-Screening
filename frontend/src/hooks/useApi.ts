import { useCallback, useEffect, useState } from 'react'
import { apiGet } from '../api/client'

export function useApi<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(Boolean(path))
  const [error, setError] = useState<string | null>(null)
  const reload = useCallback(async () => {
    if (!path) return
    setLoading(true); setError(null)
    try { setData(await apiGet<T>(path)) }
    catch (e) { setError(e instanceof Error ? e.message : 'Request failed') }
    finally { setLoading(false) }
  }, [path])
  useEffect(() => { reload() }, [reload])
  return { data, loading, error, reload }
}
