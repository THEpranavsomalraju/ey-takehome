import { api, type Table } from "@/lib/api"
import { useAsync } from "@/lib/use-async"

type Row = Record<string, string | number | boolean | null>

/** Load one CSV report (as rows) through the API, cached for the session. */
export function useTable<R = Row>(file: string) {
  return useAsync<Table<R>>(`report:${file}`, () => api.report<Table<R>>(file))
}

/** Load one JSON report through the API, cached for the session. */
export function useJson<T>(file: string) {
  return useAsync<T>(`report:${file}`, () => api.report<T>(file))
}

export const asNumber = (value: unknown) => (typeof value === "number" ? value : Number(value))
