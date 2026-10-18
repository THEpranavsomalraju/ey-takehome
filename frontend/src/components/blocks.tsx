import type { ReactNode } from "react"
import { InfoIcon } from "lucide-react"

import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { CLASS_COLORS, shortClass } from "@/lib/domain"
import { ExplainChart, type ExplainTarget } from "@/components/explain-chart"
import { cn } from "@/lib/utils"

/** A titled card holding one chart or table, with an optional control and takeaway. */
export function Panel({
  title,
  description,
  why,
  action,
  takeaway,
  explain,
  children,
  className,
}: {
  title: string
  /** What the chart shows, in plain words. */
  description?: ReactNode
  /** Why the chart is useful to look at. */
  why?: ReactNode
  action?: ReactNode
  /** What the data in the chart tells us. */
  takeaway?: ReactNode
  /** Adds an "Explain" button that asks the AI about this chart. */
  explain?: ExplainTarget
  children: ReactNode
  className?: string
}) {
  return (
    <Card className={cn("gap-4", className)}>
      <CardHeader>
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {description && <CardDescription className="text-xs">{description}</CardDescription>}
        {(action || explain) && (
          <CardAction className="flex flex-wrap items-center justify-end gap-2">
            {action}
            {explain && (
              <ExplainChart
                target={explain}
                fallback={
                  <>
                    {why && <p><span className="font-medium">Why it matters: </span>{why}</p>}
                    {takeaway && <p><span className="font-medium">Finding: </span>{takeaway}</p>}
                  </>
                }
              />
            )}
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-3">
        {children}
        {(why || takeaway) && (
          <div className="mt-auto space-y-1.5 border-t pt-3 text-xs leading-relaxed text-muted-foreground">
            {why && (
              <p>
                <span className="font-medium text-foreground">Why it matters: </span>
                {why}
              </p>
            )}
            {takeaway && (
              <p>
                <span className="font-medium text-foreground">Finding: </span>
                {takeaway}
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

/** A page section: anchor, heading, one-line purpose, then its panels. */
export function Section({
  id,
  eyebrow,
  title,
  description,
  children,
}: {
  id: string
  eyebrow: string
  title: string
  description: ReactNode
  children: ReactNode
}) {
  return (
    <section id={id} className="scroll-mt-28 space-y-4">
      <div className="max-w-3xl space-y-1">
        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{eyebrow}</p>
        <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
        <p className="text-sm leading-relaxed text-muted-foreground">{description}</p>
      </div>
      {children}
    </section>
  )
}

/** A headline number with context underneath. */
export function Stat({
  label,
  value,
  detail,
  hint,
  emphasis = false,
}: {
  label: string
  value: ReactNode
  detail?: ReactNode
  hint?: string
  emphasis?: boolean
}) {
  return (
    <Card className="gap-1 py-4">
      <CardContent className="space-y-1 px-4">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          {label}
          {hint && (
            <Tooltip>
              <TooltipTrigger aria-label={`About ${label}`}>
                <InfoIcon className="size-3.5" />
              </TooltipTrigger>
              <TooltipContent className="max-w-64 text-xs">{hint}</TooltipContent>
            </Tooltip>
          )}
        </div>
        <div
          className={cn(
            "tabular text-2xl font-semibold tracking-tight",
            emphasis && "text-primary",
          )}
        >
          {value}
        </div>
        {detail && <div className="text-xs text-muted-foreground">{detail}</div>}
      </CardContent>
    </Card>
  )
}

/** Category name with its fixed color swatch (identity never relies on color alone). */
export function ClassLabel({ label, short = false }: { label: string; short?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span
        aria-hidden
        className="size-2.5 shrink-0 rounded-[3px]"
        style={{ backgroundColor: CLASS_COLORS[label] ?? "var(--muted-foreground)" }}
      />
      {short ? shortClass(label) : label}
    </span>
  )
}

/** Compact segmented control for switching what a chart shows. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
}: {
  value: T
  onChange: (value: T) => void
  options: { value: T; label: string }[]
  label: string
}) {
  return (
    <ToggleGroup
      type="single"
      size="sm"
      variant="outline"
      value={value}
      aria-label={label}
      onValueChange={(next) => next && onChange(next as T)}
    >
      {options.map((option) => (
        <ToggleGroupItem
          key={option.value}
          value={option.value}
          className="px-2.5 text-xs text-muted-foreground data-[state=on]:border-foreground/25 data-[state=on]:bg-foreground/10 data-[state=on]:text-foreground"
        >
          {option.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  )
}

export function ChartSkeleton({ height = 260 }: { height?: number }) {
  return <Skeleton className="w-full" style={{ height }} />
}

export function LoadError({ error }: { error: Error }) {
  return <p className="text-sm text-destructive">Could not load this view: {error.message}</p>
}

/** Legend row for the category colors. */
export function ClassLegend({ classes }: { classes: readonly string[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {classes.map((c) => (
        <ClassLabel key={c} label={c} />
      ))}
    </div>
  )
}

/** Legend for chart series that are not categories (lines or swatches with a label). */
export function SeriesLegend({
  items,
  shape = "line",
  className,
}: {
  items: { label: string; color: string }[]
  shape?: "line" | "swatch"
  className?: string
}) {
  return (
    <div className={cn("flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground", className)}>
      {items.map((item) => (
        <span key={item.label} className="inline-flex items-center gap-1.5 whitespace-nowrap">
          <span
            aria-hidden
            className={shape === "line" ? "h-0.5 w-4" : "size-2.5 rounded-[3px]"}
            style={{ backgroundColor: item.color }}
          />
          {item.label}
        </span>
      ))}
    </div>
  )
}
