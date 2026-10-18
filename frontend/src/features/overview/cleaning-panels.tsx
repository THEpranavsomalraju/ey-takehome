import { useMemo, useState } from "react"
import { Bar, BarChart, CartesianGrid, LabelList, ReferenceArea, XAxis, YAxis } from "recharts"

import { ChartSkeleton, ClassLegend, LoadError, Panel, Segmented } from "@/components/blocks"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { CLASSES, CLASS_COLORS, columnName, num, pct } from "@/lib/domain"
import { asNumber, useTable } from "@/lib/reports"

type MissingRow = { column: string; missing_rows: number; missing_share: number }

export function MissingValuesPanel() {
  const { data, error } = useTable<MissingRow>("eda/missing_values.csv")
  const rows = (data?.rows ?? []).map((r) => ({
    name: `${columnName(r.column)} (${r.column})`,
    share: asNumber(r.missing_share),
    count: asNumber(r.missing_rows),
  }))
  return (
    <Panel
      title="Missing values"
      description="Share of training lines with an empty cell, per column."
      explain={{ chart: "missing-values" }}
      takeaway="Only the description is ever empty (2.7% of lines, all in the two common categories). An empty description becomes its own token instead of a reason to drop the line. In uploads, a missing amount or date is filled with the training median and the line is flagged."
    >
      {error && <LoadError error={error} />}
      {!data ? (
        <ChartSkeleton height={220} />
      ) : (
        <ChartContainer config={{ share: { label: "Missing", color: "var(--chart-1)" } }} className="h-[220px] w-full">
          <BarChart data={rows} layout="vertical" margin={{ left: 4, right: 72 }}>
            <CartesianGrid horizontal={false} stroke="var(--grid)" />
            <XAxis type="number" domain={[0, 0.05]} ticks={[0, 0.01, 0.02, 0.03, 0.04, 0.05]} tickFormatter={(v) => pct(v, 0)} tickLine={false} axisLine={false} />
            <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} width={150} interval={0} />
            <ChartTooltip
              cursor={{ fill: "var(--muted)", opacity: 0.4 }}
              content={<ChartTooltipContent hideIndicator formatter={(_, __, item) => `${num(item.payload.count)} lines (${pct(item.payload.share)})`} />}
            />
            <Bar isAnimationActive={false} dataKey="share" fill="var(--color-share)" radius={[0, 4, 4, 0]} barSize={14}>
              <LabelList dataKey="count" position="right" className="fill-muted-foreground text-[11px]" formatter={(v: unknown) => `${num(Number(v))} lines`} />
            </Bar>
          </BarChart>
        </ChartContainer>
      )}
    </Panel>
  )
}

type HistogramRow = Record<string, number> & { bin_left: number; bin_right: number }
type OutlierRow = { side: string; class: string; fence_low: number; fence_high: number }

// The histogram is on sign(x) * log(1 + |x|); label the axis in money again.
const fromSignedLog = (v: number) => Math.round(Math.sign(v) * Math.expm1(Math.abs(v)))
const money = (v: number) => {
  const a = Math.abs(v)
  const text = a >= 1e6 ? `${(a / 1e6).toFixed(0)}M` : a >= 1e3 ? `${(a / 1e3).toFixed(0)}k` : a.toFixed(0)
  return `${v < 0 ? "-" : ""}${text}`
}
// Ticks at round amounts: -100k ... -10, 0, 10 ... 1M.
const LOG_TICKS = [
  ...[1e5, 1e4, 1e3, 1e2, 10].map((v) => -Math.log1p(v)),
  0,
  ...[10, 1e2, 1e3, 1e4, 1e5, 1e6].map((v) => Math.log1p(v)),
]

export function AmountPanel() {
  const hist = useTable<HistogramRow>("eda/amount_histogram.csv")
  const outliers = useTable<OutlierRow>("eda/amount_outliers.csv")
  const [view, setView] = useState<"common" | "rare">("common")
  const shown = useMemo(() => (view === "common" ? CLASSES.slice(0, 2) : CLASSES.slice(2)), [view])
  const rows = useMemo(
    () => (hist.data?.rows ?? []).map((r) => ({ ...r, mid: (asNumber(r.bin_left) + asNumber(r.bin_right)) / 2 })),
    [hist.data],
  )
  const fences = outliers.data?.rows[0]
  const outlierCount = outliers.data?.rows.length ?? 0
  const outlierCommon = (outliers.data?.rows ?? []).filter((r) => r.class === "Category_1").length
  return (
    <Panel
      title="Amounts and outliers"
      description="Training lines per amount band (log scale, so credits sit left of zero). Shaded areas lie outside the outlier fences (1.5 times the interquartile range)."
      explain={{ chart: "amount-distribution", view: view === "common" ? "Common categories" : "Rare categories" }}
      action={
        <Segmented
          label="Categories shown"
          value={view}
          onChange={setView}
          options={[
            { value: "common", label: "Common" },
            { value: "rare", label: "Rare" },
          ]}
        />
      }
      takeaway={
        outlierCount
          ? `${num(outlierCount)} lines fall outside the fences, ${num(outlierCommon)} of them ordinary Category_1 invoices, so outliers are kept rather than removed. The amount enters the model as this signed log, plus flags for credits and round amounts (1,680 recurs in three rare categories).`
          : undefined
      }
    >
      {(hist.error || outliers.error) && <LoadError error={(hist.error ?? outliers.error) as Error} />}
      {!hist.data ? (
        <ChartSkeleton height={240} />
      ) : (
        <>
          <ChartContainer
            config={Object.fromEntries(CLASSES.map((c) => [c, { label: c, color: CLASS_COLORS[c] }]))}
            className="h-[240px] w-full"
          >
            <BarChart data={rows} margin={{ left: -12, right: 8 }} barCategoryGap={1}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              {fences && (
                <>
                  <ReferenceArea x1={-13} x2={asNumber(fences.fence_low)} fill="var(--muted-foreground)" fillOpacity={0.08} />
                  <ReferenceArea x1={asNumber(fences.fence_high)} x2={16.5} fill="var(--muted-foreground)" fillOpacity={0.08} />
                </>
              )}
              <XAxis dataKey="mid" type="number" domain={[-13, 16.5]} ticks={LOG_TICKS} tickFormatter={(v) => money(fromSignedLog(v))} tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} allowDecimals={false} />
              <ChartTooltip
                cursor={{ fill: "var(--muted)", opacity: 0.4 }}
                content={
                  <ChartTooltipContent
                    labelFormatter={(_, p) => {
                      const r = p?.[0]?.payload
                      return r ? `${money(fromSignedLog(r.bin_left))} to ${money(fromSignedLog(r.bin_right))}` : ""
                    }}
                  />
                }
              />
              {shown.map((c, i) => (
                <Bar isAnimationActive={false} key={c} dataKey={c} stackId="a" fill={CLASS_COLORS[c]} radius={i === shown.length - 1 ? [2, 2, 0, 0] : 0} />
              ))}
            </BarChart>
          </ChartContainer>
          <ClassLegend classes={shown} />
        </>
      )}
    </Panel>
  )
}

type LabelRow = { raw_label: string; canonical_label: string; rows: number }

export function LabelCleanupPanel() {
  const { data, error } = useTable<LabelRow>("eda/raw_label_mapping.csv")
  return (
    <Panel
      title="Label cleanup"
      description="The label column holds 11 spellings of 6 categories. One fixed rule maps them: lowercase, drop spaces and underscores, fix the typo categry, read the number."
      explain={{ chart: "label-cleanup" }}
      takeaway="The same rule runs on any labeled file a user uploads, so accuracy can be scored on it."
    >
      {error && <LoadError error={error} />}
      {!data ? (
        <ChartSkeleton height={240} />
      ) : (
        <Table className="text-xs">
          <TableHeader>
            <TableRow>
              <TableHead>Spelling in the file</TableHead>
              <TableHead>Category</TableHead>
              <TableHead className="text-right">Lines</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.rows.map((r) => (
              <TableRow key={r.raw_label}>
                <TableCell className="font-mono">{r.raw_label}</TableCell>
                <TableCell>{r.canonical_label}</TableCell>
                <TableCell className="tabular text-right">{num(asNumber(r.rows))}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Panel>
  )
}

type SampleRow = { row_id: number; column: string; raw_value: string | null; cleaned_value: string | null; model_input: string | null }

export function BeforeAfterPanel() {
  const { data, error } = useTable<SampleRow>("preprocessing/before_after_samples.csv")
  const ids = useMemo(() => [...new Set((data?.rows ?? []).map((r) => String(r.row_id)))], [data])
  const [chosen, setChosen] = useState<string | null>(null)
  const id = chosen ?? ids[0]
  const rows = (data?.rows ?? []).filter((r) => String(r.row_id) === id)
  return (
    <Panel
      title="Before and after preprocessing"
      description="One training line at three stages: the cell in the Excel file, the cleaned value, and what the feature pipeline sees."
      explain={{ chart: "before-after", view: id ? `Sample line ${id}` : "" }}
      action={
        ids.length > 0 && (
          <Segmented
            label="Sample line"
            value={id}
            onChange={setChosen}
            options={ids.map((value, i) => ({ value, label: `Line ${i + 1}` }))}
          />
        )
      }
      takeaway="Text is lowercased and split into words; amounts become a signed log plus flags; dates become a month index; the invoice number is not used."
    >
      {error && <LoadError error={error} />}
      {!data ? (
        <ChartSkeleton height={240} />
      ) : (
        <Table className="text-xs">
          <TableHeader>
            <TableRow>
              <TableHead>Input</TableHead>
              <TableHead>In the file</TableHead>
              <TableHead>Cleaned</TableHead>
              <TableHead>What the model sees</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={r.column}>
                <TableCell className="text-muted-foreground">{columnName(r.column)}</TableCell>
                <TableCell className="max-w-48 truncate font-mono" title={r.raw_value ?? ""}>{r.raw_value ?? "(empty)"}</TableCell>
                <TableCell className="max-w-48 truncate font-mono" title={r.cleaned_value ?? ""}>{r.cleaned_value ?? "(empty)"}</TableCell>
                <TableCell className="max-w-64 whitespace-normal font-mono text-muted-foreground">{r.model_input ?? ""}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Panel>
  )
}
