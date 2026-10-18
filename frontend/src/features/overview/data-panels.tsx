import { useMemo, useState } from "react"
import { Bar, BarChart, CartesianGrid, Cell, LabelList, XAxis, YAxis } from "recharts"
import { Term } from "@/components/term"

import { ChartSkeleton, ClassLegend, LoadError, Panel, Segmented, Stat } from "@/components/blocks"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { CLASSES, CLASS_COLORS, num, pct } from "@/lib/domain"
import { asNumber, useJson, useTable } from "@/lib/reports"

const classConfig = Object.fromEntries(
  CLASSES.map((c) => [c, { label: c, color: CLASS_COLORS[c] }]),
) satisfies ChartConfig

export function ClassDistributionPanel() {
  const { data, error } = useTable<{ class: string; rows: number; share: number }>(
    "eda/class_distribution.csv",
  )
  const [scale, setScale] = useState<"log" | "linear">("log")
  return (
    <Panel
      title="Category balance"
      explain={{ chart: "category-balance", view: `${scale} scale` }}
      description="How many past invoice lines carry each category (training rows only)."
      why={<>A model that always answers Category_1 already scores 88%, so models are also judged on <Term name="macro F1" />, which counts every category equally.</>}
      action={
        <Segmented
          label="Axis scale"
          value={scale}
          onChange={setScale}
          options={[
            { value: "log", label: "Log" },
            { value: "linear", label: "Linear" },
          ]}
        />
      }
      takeaway="88.4% of lines are Category_1, so always answering Category_1 scores 88.4%."
    >
      {error && <LoadError error={error} />}
      {!data ? (
        <ChartSkeleton height={220} />
      ) : (
        <ChartContainer config={classConfig} className="h-[220px] w-full">
          <BarChart data={data.rows} layout="vertical" margin={{ left: 4, right: 96 }}>
            <CartesianGrid horizontal={false} stroke="var(--grid)" />
            <XAxis
              type="number"
              scale={scale}
              domain={scale === "log" ? [1, 10000] : [0, "auto"]}
              allowDataOverflow
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => num(v)}
              ticks={scale === "log" ? [1, 10, 100, 1000, 10000] : undefined}
            />
            <YAxis
              type="category"
              dataKey="class"
              tickLine={false}
              axisLine={false}
              width={84}
            />
            <ChartTooltip
              cursor={{ fill: "var(--muted)", opacity: 0.4 }}
              content={
                <ChartTooltipContent
                  hideIndicator
                  formatter={(_, __, item) => (
                    <span className="tabular">
                      {num(item.payload.rows)} rows ({pct(item.payload.share, 2)})
                    </span>
                  )}
                />
              }
            />
            <Bar isAnimationActive={false} dataKey="rows" radius={[0, 4, 4, 0]} barSize={18}>
              {data.rows.map((row) => (
                <Cell key={row.class} fill={CLASS_COLORS[row.class]} />
              ))}
              <LabelList
                dataKey="rows"
                position="right"
                className="fill-muted-foreground text-[11px]"
                formatter={(v: unknown) => num(asNumber(v))}
              />
            </Bar>
          </BarChart>
        </ChartContainer>
      )}
    </Panel>
  )
}

type PeriodRow = Record<string, string | number> & { period: string }

export function MonthlyVolumePanel() {
  const { data, error } = useTable<PeriodRow>("eda/period_by_class.csv")
  const [view, setView] = useState<"all" | "rare">("rare")
  const shown = useMemo(() => (view === "all" ? [...CLASSES] : CLASSES.slice(2)), [view])
  const rows = useMemo(
    () => data?.rows.filter((r) => shown.some((c) => asNumber(r[c]) > 0)) ?? [],
    [data, shown],
  )
  return (
    <Panel
      title="When each category occurs"
      explain={{ chart: "monthly-volume", view: view === "rare" ? "Rare categories only" : "All categories" }}
      description={view === "rare" ? "Invoice lines per posting month for the four rare categories (months without one are left out)." : "Invoice lines per posting month, coloured by category."}
      why={<>Timing matters for the rare categories, so the posting month is an input and the model is also tested on later months (the <Term name="out-of-time" />).</>}
      action={
        <Segmented
          label="Categories shown"
          value={view}
          onChange={setView}
          options={[
            { value: "rare", label: "Rare only" },
            { value: "all", label: "All" },
          ]}
        />
      }
      takeaway="Category_3 starts in 2018-05, Category_4 peaks in 2018-07, and both Category_5 lines are from 2018-10."
    >
      {error && <LoadError error={error} />}
      {!data ? (
        <ChartSkeleton height={220} />
      ) : (
        <>
          <ChartContainer config={classConfig} className="h-[220px] w-full">
            <BarChart data={rows} margin={{ left: -16 }}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              <XAxis dataKey="period" tickLine={false} axisLine={false} interval="preserveStartEnd" minTickGap={16} />
              <YAxis tickLine={false} axisLine={false} allowDecimals={false} />
              <ChartTooltip cursor={{ fill: "var(--muted)", opacity: 0.4 }} content={<ChartTooltipContent />} />
              {shown.map((c, i) => (
                <Bar isAnimationActive={false}
                  key={c}
                  dataKey={c}
                  stackId="month"
                  fill={CLASS_COLORS[c]}
                  stroke="var(--card)"
                  strokeWidth={1}
                  radius={i === shown.length - 1 ? [3, 3, 0, 0] : 0}
                />
              ))}
            </BarChart>
          </ChartContainer>
          <ClassLegend classes={shown} />
        </>
      )}
    </Panel>
  )
}

// Report variable names (e.g. "Col1+Col6 pair") in business terms.
const INPUT_NAMES: Record<string, string> = {
  "Col1 vendor": "Vendor",
  "Col6 account": "GL account",
  "Col1+Col6 pair": "Vendor + GL account",
  "Col7 doc type": "Document flag",
  "Col5 period": "Posting month",
  "Col3 amount decile": "Amount (size band)",
  "Col4 description": "Description",
}
const inputName = (variable: string) => INPUT_NAMES[variable] ?? variable

type InfoRow = { variable: string; mutual_information_nats: number; normalized_mutual_information: number }
type CramerRow = Record<string, string | number> & { variable: string }

export function SignalPanel() {
  const info = useTable<InfoRow>("eda/mutual_information.csv")
  const cramer = useTable<CramerRow>("eda/cramers_v.csv")
  const [measure, setMeasure] = useState<"cramer" | "mi">("cramer")
  const rows = useMemo(() => {
    if (measure === "mi") {
      return (info.data?.rows ?? []).map((r) => ({ name: inputName(r.variable), value: asNumber(r.mutual_information_nats) }))
    }
    return (cramer.data?.rows ?? [])
      .filter((r) => r.variable !== "label")
      .map((r) => ({ name: inputName(r.variable), value: asNumber(r.label) }))
      .sort((a, b) => b.value - a.value)
  }, [measure, info.data, cramer.data])
  const loading = !info.data || !cramer.data
  return (
    <Panel
      title="Which inputs carry the label"
      explain={{ chart: "input-signal", view: measure === "cramer" ? "Cramer's V" : "Mutual information" }}
      description={
        measure === "cramer" ? (
          <>How strongly each input is linked to the category, as <Term name="Cramer's V" />: 0 = no link, 1 = the input decides the category.</>
        ) : (
          <>How much each input tells us about the category, as <Term name="mutual information" /> (higher = more).</>
        )
      }
      action={
        <Segmented
          label="Association measure"
          value={measure}
          onChange={setMeasure}
          options={[
            { value: "cramer", label: "Cramer's V" },
            { value: "mi", label: "Mutual info" },
          ]}
        />
      }
      takeaway="Vendor, GL account, and the pair are strongly tied to the category; the description carries the most information. Amount and month are weak alone but help separate the rare categories."
    >
      {(info.error || cramer.error) && <LoadError error={(info.error ?? cramer.error) as Error} />}
      {loading ? (
        <ChartSkeleton height={220} />
      ) : (
        <ChartContainer config={{ value: { label: measure === "cramer" ? "Cramer's V" : "Nats", color: "var(--chart-1)" } }} className="h-[220px] w-full">
          <BarChart data={rows} layout="vertical" margin={{ left: 4, right: 48 }}>
            <CartesianGrid horizontal={false} stroke="var(--grid)" />
            <XAxis
              type="number"
              domain={[0, measure === "cramer" ? 1 : 0.4]}
              ticks={measure === "cramer" ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.1, 0.2, 0.3, 0.4]}
              tickLine={false}
              axisLine={false}
            />
            <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} width={124} />
            <ChartTooltip cursor={{ fill: "var(--muted)", opacity: 0.4 }} content={<ChartTooltipContent hideIndicator />} />
            <Bar isAnimationActive={false} dataKey="value" fill="var(--color-value)" radius={[0, 4, 4, 0]} barSize={16}>
              <LabelList
                dataKey="value"
                position="right"
                className="fill-muted-foreground text-[11px]"
                formatter={(v: unknown) => asNumber(v).toFixed(2)}
              />
            </Bar>
          </BarChart>
        </ChartContainer>
      )}
    </Panel>
  )
}

interface Duplicates {
  rows: number
  rows_in_groups_with_copies: number
  duplicate_rows_group_key: number
  groups: number
  groups_with_copies: number
  label_conflict_groups: number
}

export function DuplicateStats() {
  const dup = useJson<Duplicates>("eda/duplicates_summary.json")
  const ceiling = useJson<{ text_amount_doc: { ceiling_accuracy: number } }>("eda/accuracy_ceiling.json")
  if (dup.error || ceiling.error) return <LoadError error={(dup.error ?? ceiling.error) as Error} />
  if (!dup.data || !ceiling.data) return <ChartSkeleton height={96} />
  const d = dup.data
  return (
    <div className="grid flex-1 grid-cols-2 gap-3">
      <Stat
        label="Rows that repeat an earlier row"
        value={pct(d.duplicate_rows_group_key / d.rows, 0)}
        detail={`${num(d.duplicate_rows_group_key)} of ${num(d.rows)} training rows`}
        hint="Identical to an earlier row on every model input except the invoice number. A random split would test the model on copies of training rows."
      />
      <Stat
        label="Rows in a group with copies"
        value={pct(d.rows_in_groups_with_copies / d.rows, 0)}
        detail={`${num(d.groups_with_copies)} groups of 2 or more`}
        hint={`${num(d.groups)} distinct input patterns in total. Grouped cross-validation keeps every group on one side of each split.`}
      />
      <Stat
        label="Groups with conflicting labels"
        value={num(d.label_conflict_groups)}
        detail={d.label_conflict_groups <= 1 ? "Copies almost always share a label" : "Some copies disagree on the label"}
      />
      <Stat
        label="Best possible accuracy"
        value={pct(ceiling.data.text_amount_doc.ceiling_accuracy, 2)}
        detail="If every repeated pattern were memorized"
        hint="Upper bound from rows that agree on vendor, description, account, amount, and document flag."
      />
    </div>
  )
}

