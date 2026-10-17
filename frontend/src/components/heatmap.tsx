import type { ReactNode } from "react"

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

/**
 * Matrix heatmap with one sequential hue (blue): the cell share sets how much
 * blue is mixed into the card surface. Every cell shows its value and explains
 * itself on hover.
 */
export function Heatmap({
  rows,
  columns,
  value,
  shade,
  format,
  describe,
  rowHeader,
  columnHeader,
  highlightDiagonal = false,
}: {
  rows: string[]
  columns: string[]
  value: (row: string, column: string) => number
  shade: (row: string, column: string) => number
  format: (v: number) => string
  describe: (row: string, column: string) => ReactNode
  rowHeader: (row: string) => ReactNode
  columnHeader: (column: string) => ReactNode
  highlightDiagonal?: boolean
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-0.5 text-xs">
        <thead>
          <tr>
            <th />
            {columns.map((c) => (
              <th key={c} className="px-1 pb-1 text-center font-normal text-muted-foreground">
                {columnHeader(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r}>
              <th className="whitespace-nowrap pr-2 text-left font-normal text-muted-foreground">
                {rowHeader(r)}
              </th>
              {columns.map((c) => {
                const share = Math.max(0, Math.min(1, shade(r, c)))
                const strong = share > 0.55
                return (
                  <td key={c} className="p-0">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div
                          className={cn(
                            "tabular flex h-9 min-w-11 items-center justify-center rounded-[4px]",
                            highlightDiagonal && r === c && "ring-1 ring-foreground/30",
                            strong ? "text-white" : "text-foreground/80",
                          )}
                          style={{
                            backgroundColor: `color-mix(in srgb, var(--chart-1) ${Math.round(
                              share * 85,
                            )}%, var(--muted))`,
                          }}
                        >
                          {format(value(r, c))}
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-60 text-xs">{describe(r, c)}</TooltipContent>
                    </Tooltip>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
