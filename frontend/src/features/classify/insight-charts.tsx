import { Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine, XAxis, YAxis } from "recharts"

import { Panel, SeriesLegend } from "@/components/blocks"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import type { Insights } from "@/lib/api"
import { CLASS_COLORS, flagLabel, num, pct } from "@/lib/domain"

export function PredictedClassesChart({
  uploadId,
  insights,
  selected,
  onSelect,
}: {
  uploadId: string
  insights: Insights
  selected: string | null
  onSelect: (label: string | null) => void
}) {
  const data = insights.by_class.filter((r) => r.rows > 0)
  return (
    <Panel
      title="Predicted categories"
      explain={{ chart: "upload-categories", uploadId }}
      description="How many lines in this file went to each category. Click a bar to show only those lines in the table."
    >
      <ChartContainer config={{ rows: { label: "Rows" } }} className="h-[200px] w-full">
        <BarChart data={data} layout="vertical" margin={{ left: 4, right: 56 }}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" tickLine={false} axisLine={false} hide />
          <YAxis type="category" dataKey="label" tickLine={false} axisLine={false} width={84} />
          <ChartTooltip
            cursor={{ fill: "var(--muted)", opacity: 0.4 }}
            content={
              <ChartTooltipContent
                hideIndicator
                formatter={(_, __, item) => (
                  <span className="tabular">
                    {num(item.payload.rows)} rows, {num(item.payload.review)} need review
                  </span>
                )}
              />
            }
          />
          <Bar isAnimationActive={false}
            dataKey="rows"
            radius={[0, 4, 4, 0]}
            barSize={18}
            className="cursor-pointer"
            onClick={(d: { payload?: { label: string } }) => {
              const label = d.payload?.label ?? null
              onSelect(label === selected ? null : label)
            }}
          >
            {data.map((r) => (
              <Cell
                key={r.label}
                fill={CLASS_COLORS[r.label]}
                fillOpacity={selected && selected !== r.label ? 0.3 : 1}
              />
            ))}
            <LabelList dataKey="rows" position="right" className="fill-muted-foreground text-[11px]" formatter={(v: unknown) => num(Number(v))} />
          </Bar>
        </BarChart>
      </ChartContainer>
    </Panel>
  )
}

export function ConfidenceChart({ uploadId, insights }: { uploadId: string; insights: Insights }) {
  const data = insights.confidence_histogram
    .filter((b) => b.high > 0.15)
    .map((b) => ({ ...b, bin: b.low + (b.high - b.low) / 2 }))
  return (
    <Panel
      title="How confident the model is"
      explain={{ chart: "upload-confidence", uploadId }}
      description={`How sure the model is across this file: lines per confidence level, automatic and review side by side (square-root scale, dashed line = ${pct(insights.threshold, 0)} threshold).`}
    >
      <ChartContainer
        config={{
          auto: { label: "Decided automatically", color: "var(--chart-1)" },
          review: { label: "Needs review", color: "var(--warning)" },
        }}
        className="h-[200px] w-full"
      >
        <BarChart data={data} margin={{ left: -16, right: 8 }} barCategoryGap={2} barGap={1}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis
            dataKey="bin"
            type="number"
            domain={[0.15, 1]}
            ticks={[0.2, 0.4, 0.6, 0.8, 1]}
            tickFormatter={(v) => pct(v, 0)}
            tickLine={false}
            axisLine={false}
          />
          <YAxis tickLine={false} axisLine={false} scale="sqrt" allowDecimals={false} />
          <ReferenceLine x={insights.threshold} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
          <ChartTooltip
            cursor={{ fill: "var(--muted)", opacity: 0.4 }}
            content={
              <ChartTooltipContent
                labelFormatter={(_, p) => {
                  const b = p?.[0]?.payload
                  return b ? `Confidence ${pct(b.low, 0)} to ${pct(b.high, 0)}` : ""
                }}
              />
            }
          />
          <Bar isAnimationActive={false} dataKey="auto" fill="var(--color-auto)" radius={[2, 2, 0, 0]} />
          <Bar isAnimationActive={false} dataKey="review" fill="var(--color-review)" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ChartContainer>
      <SeriesLegend
        shape="swatch"
        items={[
          { label: "Automatic", color: "var(--chart-1)" },
          { label: "Needs review", color: "var(--warning)" },
        ]}
      />
    </Panel>
  )
}

export function FlagsChart({
  uploadId,
  insights,
  selected,
  onSelect,
}: {
  uploadId: string
  insights: Insights
  selected: string | null
  onSelect: (flag: string | null) => void
}) {
  const data = insights.flags.map((f) => ({ ...f, name: flagLabel(f.flag) }))
  return (
    <Panel
      title="Warnings raised"
      explain={{ chart: "upload-warnings", uploadId }}
      description="Which warnings fired in this file and how often. Click a bar to show only those lines."
    >
      {data.length === 0 ? (
        <p className="py-10 text-center text-sm text-muted-foreground">No warnings: every input looked familiar.</p>
      ) : (
        <ChartContainer config={{ rows: { label: "Rows" } }} className="w-full" style={{ height: Math.max(200, data.length * 30) }}>
          <BarChart data={data} layout="vertical" margin={{ left: 4, right: 48 }}>
            <CartesianGrid horizontal={false} stroke="var(--grid)" />
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} width={150} interval={0} />
            <ChartTooltip cursor={{ fill: "var(--muted)", opacity: 0.4 }} content={<ChartTooltipContent hideIndicator />} />
            <Bar isAnimationActive={false}
              dataKey="rows"
              radius={[0, 4, 4, 0]}
              barSize={16}
              className="cursor-pointer"
              onClick={(d: { payload?: { flag: string } }) => {
                const flag = d.payload?.flag ?? null
                onSelect(flag === selected ? null : flag)
              }}
            >
              {data.map((f) => (
                <Cell key={f.flag} fill="var(--warning)" fillOpacity={selected && selected !== f.flag ? 0.3 : 0.85} />
              ))}
              <LabelList dataKey="rows" position="right" className="fill-muted-foreground text-[11px]" />
            </Bar>
          </BarChart>
        </ChartContainer>
      )}
    </Panel>
  )
}
