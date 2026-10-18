import { useMemo, useState } from "react"
import { CartesianGrid, Line, LineChart, ReferenceDot, ReferenceLine, XAxis, YAxis } from "recharts"
import { Term } from "@/components/term"

import { ChartSkeleton, LoadError, Panel, Segmented, SeriesLegend } from "@/components/blocks"
import { DotPlot } from "@/components/dot-plot"
import { Badge } from "@/components/ui/badge"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Slider } from "@/components/ui/slider"
import { flagLabel, num, pct } from "@/lib/domain"
import { asNumber, useFinalModel, useJson, useTable } from "@/lib/reports"

const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1)

type ReliabilityRow = {
  rows: number
  mean_confidence: number | null
  accuracy: number | null
  model: string
  stage: string
}

export function CalibrationPanel() {
  const { data, error } = useTable<ReliabilityRow>("selection/calibration_reliability.csv")
  const summary = useJson<Record<string, { ece_before: number; ece_after: number; temperature: number }>>(
    "selection/calibration_summary.json",
  )
  const [stage, setStage] = useState<"both" | "after" | "before">("both")
  const final = useFinalModel().name
  const points = useMemo(() => {
    const rows = (data?.rows ?? []).filter((r) => r.model === final && asNumber(r.rows) > 0)
    const pick = (s: string) =>
      rows.filter((r) => r.stage === s).map((r) => ({
        confidence: asNumber(r.mean_confidence),
        accuracy: asNumber(r.accuracy),
        rows: asNumber(r.rows),
      }))
    return { before: pick("before"), after: pick("after") }
  }, [data, final])
  const stats = final ? summary.data?.[final] : undefined
  return (
    <Panel
      title="Does 90% confident mean 90% right?"
      explain={{ chart: "calibration", view: stage === "both" ? "Raw and calibrated" : stage === "before" ? "Raw" : "Calibrated" }}
      description={<>Across: how sure the model said it was. Up: how often it was actually right. Points on the dashed line are perfectly calibrated (<Term name="calibration" />).</>}
      why="The review threshold below depends on these confidences being accurate."
      action={
        <Segmented
          label="Calibration stage"
          value={stage}
          onChange={setStage}
          options={[
            { value: "both", label: "Both" },
            { value: "before", label: "Raw" },
            { value: "after", label: "Calibrated" },
          ]}
        />
      }
      takeaway={
        stats
          ? `One temperature (${stats.temperature.toFixed(2)}) divides every score. The calibration error falls from ${pct(stats.ece_before)} to ${pct(stats.ece_after, 2)}, and neither the predicted category nor the explanations change.`
          : undefined
      }
    >
      {error && <LoadError error={error} />}
      {!data ? (
        <ChartSkeleton height={260} />
      ) : (
        <ChartContainer
          config={{
            after: { label: "Calibrated", color: "var(--primary)" },
            before: { label: "Raw", color: "var(--muted-foreground)" },
          }}
          className="h-[340px] w-full"
        >
          <LineChart margin={{ left: -8, right: 12, top: 8 }}>
            <CartesianGrid stroke="var(--grid)" />
            <XAxis
              type="number"
              dataKey="confidence"
              domain={[0.3, 1]}
              ticks={[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
              tickFormatter={(v) => pct(v, 0)}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              type="number"
              domain={[0.3, 1]}
              ticks={[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
              tickFormatter={(v) => pct(v, 0)}
              tickLine={false}
              axisLine={false}
            />
            <ReferenceLine
              segment={[
                { x: 0.3, y: 0.3 },
                { x: 1, y: 1 },
              ]}
              stroke="var(--muted-foreground)"
              strokeDasharray="4 4"
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  hideLabel
                  formatter={(_, name, item) => (
                    <span className="flex flex-col gap-0.5">
                      <span className="font-medium">{name === "after" ? "Calibrated" : "Raw"}</span>
                      <span className="tabular text-muted-foreground">
                        Stated {pct(item.payload.confidence)} - actual {pct(item.payload.accuracy)} ({num(item.payload.rows)} rows)
                      </span>
                    </span>
                  )}
                />
              }
            />
            {stage !== "after" && (
              <Line isAnimationActive={false} data={points.before} dataKey="accuracy" name="before" stroke="var(--color-before)" strokeWidth={2} dot={{ r: 3 }} />
            )}
            {stage !== "before" && (
              <Line isAnimationActive={false} data={points.after} dataKey="accuracy" name="after" stroke="var(--color-after)" strokeWidth={2} dot={{ r: 4 }} />
            )}
          </LineChart>
        </ChartContainer>
      )}
    </Panel>
  )
}

type ThresholdRow = {
  threshold: number
  coverage: number
  accuracy_kept: number
  accuracy_flagged: number
  rows_flagged: number
  chosen: boolean | string
}

export function ThresholdPanel() {
  const { data, error } = useTable<ThresholdRow>("selection/threshold_table.csv")
  const rows = useMemo(
    () =>
      (data?.rows ?? []).map((r) => ({
        threshold: asNumber(r.threshold),
        coverage: asNumber(r.coverage),
        kept: asNumber(r.accuracy_kept),
        flagged: asNumber(r.accuracy_flagged),
        rowsFlagged: asNumber(r.rows_flagged),
        chosen: r.chosen === true || r.chosen === "True",
      })),
    [data],
  )
  const chosenIndex = Math.max(0, rows.findIndex((r) => r.chosen))
  const chosen = rows.find((r) => r.chosen)
  const [index, setIndex] = useState<number | null>(null)
  const current = rows[index ?? chosenIndex]
  return (
    <Panel
      title="Review threshold explorer"
      explain={{ chart: "review-threshold", view: current ? `Threshold ${current.threshold.toFixed(2)} selected` : "" }}
      description={<>Drag the slider to choose the <Term name="review threshold" />: lines below it go to a person. Figures are from cross-validation on 5,309 lines.</>}
      takeaway={
        chosen
          ? `The chosen threshold (${chosen.threshold.toFixed(2)}) is the lowest one at which the automatically decided rows reach the accuracy target. ${pct(1 - chosen.coverage, 0)} of rows go to review, and those rows are only ${pct(chosen.flagged, 0)} accurate, so review effort lands where the model is weak.`
          : undefined
      }
    >
      {error && <LoadError error={error} />}
      {!current ? (
        <ChartSkeleton height={300} />
      ) : (
        <>
          <div className="flex items-center gap-4">
            <span className="w-28 shrink-0 text-xs text-muted-foreground">
              Threshold <span className="tabular font-medium text-foreground">{current.threshold.toFixed(2)}</span>
            </span>
            <Slider
              value={[index ?? chosenIndex]}
              min={0}
              max={rows.length - 1}
              step={1}
              onValueChange={([v]) => setIndex(v)}
              aria-label="Review threshold"
            />
            {current.chosen && <Badge variant="outline" className="shrink-0 border-primary/60 text-primary">Chosen</Badge>}
          </div>
          <div className="grid grid-cols-3 gap-3 text-center">
            {[
              { label: "Decided automatically", value: pct(current.coverage), sub: `${num(current.rowsFlagged)} rows to review` },
              { label: "Accuracy, automatic rows", value: pct(current.kept) },
              { label: "Accuracy, reviewed rows", value: pct(current.flagged) },
            ].map((s) => (
              <div key={s.label} className="rounded-md border bg-muted/30 px-2 py-2">
                <div className="text-[11px] text-muted-foreground">{s.label}</div>
                <div className="tabular text-lg font-semibold">{s.value}</div>
                {s.sub && <div className="text-[11px] text-muted-foreground">{s.sub}</div>}
              </div>
            ))}
          </div>
          <ChartContainer
            config={{
              coverage: { label: "Decided automatically", color: "var(--chart-1)" },
              kept: { label: "Accuracy, automatic", color: "var(--primary)" },
              flagged: { label: "Accuracy, reviewed", color: "var(--against)" },
            }}
            className="h-[200px] w-full"
          >
            <LineChart data={rows} margin={{ left: -8, right: 12, top: 8 }}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              <XAxis dataKey="threshold" type="number" domain={[0.5, 1]} tickLine={false} axisLine={false} />
              <YAxis domain={[0.4, 1]} tickFormatter={(v) => pct(v, 0)} tickLine={false} axisLine={false} />
              <ReferenceLine x={current.threshold} stroke="var(--foreground)" strokeOpacity={0.4} />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_, p) => `Threshold ${p?.[0]?.payload?.threshold}`}
                    formatter={(v, name) => (
                      <span className="flex w-full justify-between gap-4">
                        <span className="text-muted-foreground">
                          {name === "coverage" ? "Automatic share" : name === "kept" ? "Accuracy, automatic" : "Accuracy, reviewed"}
                        </span>
                        <span className="tabular">{pct(asNumber(v))}</span>
                      </span>
                    )}
                  />
                }
              />
              <Line isAnimationActive={false} dataKey="coverage" stroke="var(--color-coverage)" strokeWidth={2} dot={false} />
              <Line isAnimationActive={false} dataKey="kept" stroke="var(--color-kept)" strokeWidth={2} dot={false} />
              <Line isAnimationActive={false} dataKey="flagged" stroke="var(--color-flagged)" strokeWidth={2} dot={false} />
              <ReferenceDot x={current.threshold} y={current.kept} r={4} fill="var(--primary)" stroke="var(--card)" />
            </LineChart>
          </ChartContainer>
          <SeriesLegend
            items={[
              { label: "Decided automatically", color: "var(--chart-1)" },
              { label: "Accuracy, automatic", color: "var(--primary)" },
              { label: "Accuracy, reviewed", color: "var(--against)" },
            ]}
          />
        </>
      )}
    </Panel>
  )
}

type FlagRow = { flag: string; rows: number; accuracy: number | null; severity?: string }

export function FlagReliabilityPanel() {
  const oof = useTable<FlagRow>("selection/accuracy_by_flag.csv")
  const holdout = useTable<FlagRow>("holdout/holdout_accuracy_by_flag.csv")
  const evaluation = useJson<{ champion: { accuracy: number } }>("holdout/holdout_evaluation.json")
  const [source, setSource] = useState<"oof" | "holdout">("oof")
  const table = source === "oof" ? oof : holdout
  // Severities are set out of fold; the holdout table reuses them by flag code.
  const severity = useMemo(
    () => new Map((oof.data?.rows ?? []).map((r) => [r.flag, r.severity])),
    [oof.data],
  )
  const overall =
    source === "oof"
      ? oof.data?.rows.find((r) => r.flag === "(all rows)")?.accuracy
      : evaluation.data?.champion.accuracy
  const items = useMemo(
    () =>
      (table.data?.rows ?? [])
        .filter((r) => asNumber(r.rows) > 0 && r.accuracy !== null)
        .map((r) => {
          const label = r.flag.startsWith("(") ? capitalize(r.flag.replace(/[()]/g, "")) : flagLabel(r.flag)
          return {
            key: r.flag,
            label: (
              <span className="inline-flex items-center gap-1.5">
                {label}
                {severity.get(r.flag) === "critical" && (
                  <Badge variant="outline" className="h-4 px-1 text-[10px] text-critical">review</Badge>
                )}
              </span>
            ),
            value: asNumber(r.accuracy),
            highlight: r.flag === "(no flag)",
            detail: `${label}: ${pct(asNumber(r.accuracy))} accurate on ${num(asNumber(r.rows))} rows`,
          }
        }),
    [table.data, severity],
  )
  return (
    <Panel
      title="How reliable is each warning?"
      explain={{ chart: "warning-reliability", view: source === "oof" ? "Cross-validation" : "Holdout" }}
      description="How accurate the model is on the lines that carry each warning. Warnings marked review send the line to a person."
      action={
        <Segmented
          label="Data"
          value={source}
          onChange={setSource}
          options={[
            { value: "oof", label: "Cross-validation" },
            { value: "holdout", label: "Holdout" },
          ]}
        />
      }
      takeaway={
        source === "oof"
          ? "Lines with no warning are about 98% accurate. New vendors, new GL accounts, and low confidence are where the model errs, so those lines go to review."
          : "On the holdout the same warnings mark the weaker lines. Which warnings send a line to review was decided from cross-validation, not from the holdout."
      }
    >
      {table.error && <LoadError error={table.error} />}
      {!table.data ? (
        <ChartSkeleton height={220} />
      ) : (
        <DotPlot items={items} domain={[0.5, 1]} format={(v) => pct(v, 1)} tickFormat={(v) => pct(v, 0)} ticks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]} reference={overall == null ? undefined : { value: asNumber(overall), label: source === "oof" ? "all rows, cross-validation" : "all holdout rows" }} />
      )}
    </Panel>
  )
}
