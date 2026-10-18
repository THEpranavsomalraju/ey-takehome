import { useState, type ReactNode } from "react"
import { SparklesIcon } from "lucide-react"

import { Markdown } from "@/components/markdown"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Skeleton } from "@/components/ui/skeleton"
import { api, type ChartAnswer } from "@/lib/api"
import { useSettings } from "@/lib/settings"
import { cn } from "@/lib/utils"

export interface ExplainTarget {
  /** Chart id known to the server (ledgerclf/chart_explainer.py). */
  chart: string
  /** What the reader currently sees, e.g. the selected toggle. */
  view?: string
  /** The upload, for charts about an uploaded file. */
  uploadId?: string
}

/**
 * "Explain" button for one chart. On open it asks the server for an AI
 * explanation built from the chart's own data; without Claude it shows the
 * chart's built-in summary instead.
 */
export function ExplainChart({ target, fallback }: { target: ExplainTarget; fallback?: ReactNode }) {
  const { apiKey, useLlm } = useSettings()
  const key = `${target.chart}|${target.view ?? ""}|${target.uploadId ?? ""}|${useLlm}`
  const [state, setState] = useState<{ key: string; answer?: ChartAnswer; error?: string }>({ key: "" })
  const current = state.key === key ? state : null

  const load = () => {
    if (current) return
    setState({ key })
    api
      .explainChart(target.chart, target.view ?? "", target.uploadId ?? null, useLlm, apiKey || null)
      .then((answer) => setState({ key, answer }))
      .catch((e: Error) => setState({ key, error: e.message }))
  }

  const answer = current?.answer
  return (
    <Popover onOpenChange={(open) => open && load()}>
      <PopoverTrigger asChild>
        <Button size="sm" variant="ghost" className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground">
          <SparklesIcon className="size-3.5" />
          Explain
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" collisionPadding={{ top: 72, bottom: 16 }} className="w-[min(30rem,calc(100vw-2rem))] space-y-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-sm font-medium">About this chart</p>
          {answer && (
            <Badge variant="outline" className={cn("font-normal", answer.engine === "claude" && "border-primary/50 text-primary")}>
              {answer.engine === "claude" ? `Claude${answer.cached ? ", saved answer" : ""}` : "Built-in summary"}
            </Badge>
          )}
        </div>
        {current?.error && <p className="text-sm text-destructive">{current.error}</p>}
        {!answer && !current?.error && (
          <div className="space-y-2">
            <Skeleton className="h-4 w-11/12" />
            <Skeleton className="h-4 w-10/12" />
            <Skeleton className="h-4 w-9/12" />
            <Skeleton className="h-4 w-7/12" />
          </div>
        )}
        {answer?.text && <Markdown text={answer.text} />}
        {answer && !answer.text && (
          <div className="space-y-2 text-sm">
            {fallback}
            {answer.fallback_reason && (
              <p className="text-xs text-muted-foreground">AI explanation not used: {answer.fallback_reason}</p>
            )}
          </div>
        )}
        {answer?.text && (
          <p className="text-[11px] text-muted-foreground">
            Written by Claude from this chart's data.
          </p>
        )}
      </PopoverContent>
    </Popover>
  )
}
