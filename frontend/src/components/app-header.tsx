import { useState } from "react"
import { CircleCheckIcon, CircleXIcon, CpuIcon, KeyRoundIcon, SettingsIcon, TriangleAlertIcon } from "lucide-react"

import { LoadError } from "@/components/blocks"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Separator } from "@/components/ui/separator"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Spinner } from "@/components/ui/spinner"
import { Switch } from "@/components/ui/switch"
import { api, type AgentStatus, type HealthCheck, type ModelInfo } from "@/lib/api"
import { flagLabel, modelName, num, pct } from "@/lib/domain"
import { useSettings } from "@/lib/settings"
import { useAsync } from "@/lib/use-async"
import { cn } from "@/lib/utils"

const ROUTES: Record<AgentStatus["route"], string> = {
  direct: "Claude, your API key",
  relay: "Claude via project relay",
  offline: "Offline explanations",
}

export function AgentBadge() {
  const { data } = useAsync("agent-status", api.agentStatus)
  const { apiKey, useLlm } = useSettings()
  const route = !useLlm ? "offline" : apiKey ? "direct" : (data?.route ?? "offline")
  return (
    <Badge variant="outline" className="hidden font-normal text-muted-foreground md:inline-flex">
      Agent: {ROUTES[route]}
    </Badge>
  )
}

export function SettingsPopover() {
  const { apiKey, setApiKey, useLlm, setUseLlm } = useSettings()
  const [draft, setDraft] = useState(apiKey)
  return (
    <Popover onOpenChange={(open) => open && setDraft(apiKey)}>
      <PopoverTrigger asChild>
        <Button size="icon-sm" variant="ghost" aria-label="Agent settings">
          <SettingsIcon />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 space-y-4">
        <div className="space-y-1">
          <p className="text-sm font-medium">Explanation agent</p>
          <p className="text-xs text-muted-foreground">
            Works without setup through the project relay. Optionally use your own Anthropic key; it is
            kept in this browser tab's memory only and never stored.
          </p>
        </div>
        <form
          className="space-y-2"
          onSubmit={(e) => {
            e.preventDefault()
            setApiKey(draft.trim())
          }}
        >
          <Label htmlFor="api-key" className="text-xs">Anthropic API key (optional)</Label>
          <div className="flex gap-2">
            <Input
              id="api-key"
              type="password"
              autoComplete="off"
              placeholder="sk-ant-..."
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              className="h-8 text-xs"
            />
            <Button type="submit" size="sm" variant="secondary">
              <KeyRoundIcon />
              Use
            </Button>
          </div>
          {apiKey && (
            <button type="button" className="text-xs text-muted-foreground underline underline-offset-4" onClick={() => (setApiKey(""), setDraft(""))}>
              Remove key (back to the relay)
            </button>
          )}
        </form>
        <Separator />
        <div className="flex items-center justify-between gap-3">
          <div>
            <Label htmlFor="use-llm" className="text-xs">Use Claude</Label>
            <p className="text-[11px] text-muted-foreground">Off: deterministic offline explanations only.</p>
          </div>
          <Switch id="use-llm" checked={useLlm} onCheckedChange={setUseLlm} />
        </div>
      </PopoverContent>
    </Popover>
  )
}

export function ModelSheet() {
  const model = useAsync<ModelInfo>("model-info", api.model)
  const [health, setHealth] = useState<HealthCheck[] | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [checking, setChecking] = useState(false)
  const info = model.data
  const active = info ? (info.fallback_active ? info.fallback : info.champion) : null
  const scores = (active?.scores ?? {}) as Record<string, number>
  const runHealth = () => {
    setChecking(true)
    setHealthError(null)
    api
      .health()
      .then((h) => setHealth(h.checks))
      .catch((e: Error) => setHealthError(e.message))
      .finally(() => setChecking(false))
  }
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button size="sm" variant="outline" className="gap-2" aria-label="Model details">
          <CpuIcon />
          <span className="hidden sm:inline">{info ? modelName(info.model_used) : "Model"}</span>
          {info?.fallback_active && <TriangleAlertIcon className="text-warning" />}
        </Button>
      </SheetTrigger>
      <SheetContent className="w-full gap-0 p-0 data-[side=right]:w-full data-[side=right]:sm:max-w-md">
        <SheetHeader className="border-b px-6 py-4">
          <SheetTitle>Model details</SheetTitle>
          <SheetDescription>What is running, how it was trained, and its review policy.</SheetDescription>
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
          <div className="space-y-6 px-6 py-5 text-sm">
            {model.error ? (
              <LoadError error={model.error} />
            ) : !info || !active ? (
              <Spinner />
            ) : (
              <>
                <section className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{modelName(info.model_used)}</span>
                    <Badge variant="outline" className={info.fallback_active ? "text-warning" : "text-good"}>
                      {info.fallback_active ? "Fallback active" : "Primary model"}
                    </Badge>
                  </div>
                  {info.fallback_reason && <p className="text-xs text-warning">{info.fallback_reason}</p>}
                  <dl className="grid grid-cols-[9rem_1fr] gap-y-1.5 text-xs">
                    <dt className="text-muted-foreground">Trained on</dt>
                    <dd>{num(info.training_rows)} rows (holdout excluded)</dd>
                    <dt className="text-muted-foreground">CV accuracy</dt>
                    <dd className="tabular">{pct(Number(scores.cv_accuracy))}</dd>
                    <dt className="text-muted-foreground">CV macro F1</dt>
                    <dd className="tabular">{Number(scores.macro_f1_5).toFixed(3)}</dd>
                    <dt className="text-muted-foreground">Calibration</dt>
                    <dd className="tabular">temperature {Number(active.temperature).toFixed(3)}</dd>
                    <dt className="text-muted-foreground">Trained</dt>
                    <dd>{new Date(info.created_utc).toLocaleString()}</dd>
                  </dl>
                </section>
                <Separator />
                <section className="space-y-2">
                  <p className="font-medium">Review policy</p>
                  <p className="text-xs text-muted-foreground">
                    A row goes to review below {pct(info.review_policy.threshold, 0)} confidence or with any critical warning.
                  </p>
                  {(["critical", "warning", "info"] as const).map((level) => {
                    const codes = Object.entries(info.review_policy.severities)
                      .filter(([, severity]) => severity === level)
                      .map(([code]) => flagLabel(code))
                      .sort()
                    if (!codes.length) return null
                    return (
                      <div key={level} className="space-y-1.5">
                        <p className="text-xs text-muted-foreground">
                          {level === "critical" ? "Sends the row to review" : level === "warning" ? "Shown as a warning only" : "Shown as a note"}
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {codes.map((label) => (
                            <Badge
                              key={label}
                              variant="outline"
                              className={cn("font-normal", level === "critical" ? "border-warning/40 text-warning" : "text-muted-foreground")}
                            >
                              {label}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </section>
                <Separator />
                <section className="space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="font-medium">Environment check</p>
                    <Button size="sm" variant="outline" onClick={runHealth} disabled={checking}>
                      {checking && <Spinner />}
                      Run check
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">Same checks as python -m ledgerclf doctor (takes a few seconds).</p>
                  {healthError && <p className="text-xs text-destructive">{healthError}</p>}
                  {health && (
                    <ul className="space-y-1 text-xs">
                      {health.map((c) => (
                        <li key={c.name} className="flex gap-2">
                          {c.status === "PASS" ? (
                            <CircleCheckIcon className="mt-0.5 size-3.5 shrink-0 text-good" />
                          ) : c.status === "FAIL" ? (
                            <CircleXIcon className="mt-0.5 size-3.5 shrink-0 text-critical" />
                          ) : (
                            <TriangleAlertIcon className="mt-0.5 size-3.5 shrink-0 text-warning" />
                          )}
                          <span>
                            <span className="text-foreground">{c.name}</span>{" "}
                            <span className="text-muted-foreground">{c.detail}</span>
                            {c.fix && <span className="block text-warning">{c.fix}</span>}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
                <Separator />
                <section className="space-y-2">
                  <p className="font-medium">Library versions</p>
                  <dl className="grid grid-cols-[9rem_1fr] gap-y-1 text-xs">
                    {Object.entries(info.library_versions).map(([k, v]) => (
                      <div key={k} className="contents">
                        <dt className="text-muted-foreground">{k}</dt>
                        <dd className="tabular">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              </>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
