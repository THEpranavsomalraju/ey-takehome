import type { ReactNode } from "react"

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

export interface DotItem {
  key: string
  label: ReactNode
  value: number
  low?: number
  high?: number
  highlight?: boolean
  detail?: ReactNode
}

/**
 * Horizontal dot plot on a zoomed axis. Bars would exaggerate differences on a
 * truncated axis, so position (and an optional +-1 std range) carries the value.
 */
export function DotPlot({
  items,
  domain,
  format,
  tickFormat = format,
  ticks: tickValues,
  reference,
}: {
  items: DotItem[]
  domain: [number, number]
  format: (v: number) => string
  tickFormat?: (v: number) => string
  ticks?: number[]
  reference?: { value: number; label: string }
}) {
  const [min, max] = domain
  const position = (v: number) => `${((Math.min(max, Math.max(min, v)) - min) / (max - min)) * 100}%`
  const ticks = tickValues ?? Array.from({ length: 5 }, (_, i) => min + ((max - min) * i) / 4)

  return (
    <div className="space-y-1">
      {items.map((item) => (
        <Tooltip key={item.key}>
          <TooltipTrigger asChild>
            <div className="grid grid-cols-[minmax(7rem,11rem)_1fr_4rem] items-center gap-3 rounded-md px-1 py-1 hover:bg-muted/50">
              <div
                className={cn(
                  "truncate text-xs",
                  item.highlight ? "font-medium text-foreground" : "text-muted-foreground",
                )}
              >
                {item.label}
              </div>
              <div className="relative h-5">
                <div className="absolute inset-x-0 top-1/2 h-px bg-border" />
                {reference && (
                  <div
                    className="absolute top-0 h-full w-px bg-muted-foreground/50"
                    style={{ left: position(reference.value) }}
                  />
                )}
                {item.low != null && item.high != null && (
                  <div
                    className={cn(
                      "absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full",
                      item.highlight ? "bg-primary/60" : "bg-muted-foreground/50",
                    )}
                    style={{
                      left: position(item.low),
                      width: `calc(${position(item.high)} - ${position(item.low)})`,
                    }}
                  />
                )}
                <div
                  className={cn(
                    "absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-card",
                    item.highlight ? "bg-primary" : "bg-foreground/70",
                  )}
                  style={{ left: position(item.value) }}
                />
              </div>
              <div className="tabular text-right text-xs">{format(item.value)}</div>
            </div>
          </TooltipTrigger>
          {item.detail && <TooltipContent className="max-w-64 text-xs">{item.detail}</TooltipContent>}
        </Tooltip>
      ))}
      <div className="grid grid-cols-[minmax(7rem,11rem)_1fr_4rem] gap-3 px-1">
        <div />
        <div className="relative h-4 text-[10px] text-muted-foreground">
          {ticks.map((t) => (
            <span key={t} className="tabular absolute -translate-x-1/2" style={{ left: position(t) }}>
              {tickFormat(t)}
            </span>
          ))}
        </div>
        <div />
      </div>
      {reference && (
        <p className="px-1 text-[11px] text-muted-foreground">
          Vertical line: {reference.label} ({format(reference.value)})
        </p>
      )}
    </div>
  )
}
