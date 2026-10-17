import { useEffect, useState } from "react"

export interface AsyncState<T> {
  data: T | null
  error: Error | null
  loading: boolean
}

const cache = new Map<string, Promise<unknown>>()

/** Load data once per key; repeated calls with the same key share one request. */
export function useAsync<T>(key: string | null, load: () => Promise<T>): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ data: null, error: null, loading: !!key })

  useEffect(() => {
    if (!key) return
    let active = true
    let pending = cache.get(key) as Promise<T> | undefined
    if (!pending) {
      pending = load()
      cache.set(key, pending)
      pending.catch(() => cache.delete(key))
    }
    pending.then(
      (data) => active && setState({ data, error: null, loading: false }),
      (error: Error) => active && setState({ data: null, error, loading: false }),
    )
    return () => {
      active = false
    }
    // `load` is keyed by `key`; a new key means a new request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return state
}
