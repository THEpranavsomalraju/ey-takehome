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

/** Name of the final model, read from the selection report (no hard-coded copy). */
export function useFinalModel() {
  const { data } = useJson<{ champion: string; fallback: string }>("selection/champion_decision.json")
  const name = data?.champion ?? null
  // The benchmark leaderboard spells powers without the dot: logodds_blend_p0.25 -> p025.
  return { name, leaderboardKey: name ? name.replace(".", "") : null }
}
