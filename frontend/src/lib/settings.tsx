import { createContext, useContext, useMemo, useState, type ReactNode } from "react"

interface AgentSettings {
  apiKey: string
  setApiKey: (key: string) => void
  useLlm: boolean
  setUseLlm: (value: boolean) => void
}

const SettingsContext = createContext<AgentSettings | null>(null)

/** Agent settings live in memory only: the key is never written to storage. */
export function SettingsProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKey] = useState("")
  const [useLlm, setUseLlm] = useState(true)
  const value = useMemo(() => ({ apiKey, setApiKey, useLlm, setUseLlm }), [apiKey, useLlm])
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
}

export function useSettings() {
  const context = useContext(SettingsContext)
  if (!context) throw new Error("useSettings must be used inside SettingsProvider")
  return context
}
