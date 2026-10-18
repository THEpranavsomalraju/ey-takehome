import { useMemo, useState } from "react"
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts"
import { Term } from "@/components/term"

import { ChartSkeleton, LoadError, Panel, Segmented, SeriesLegend } from "@/components/blocks"
import { DotPlot } from "@/components/dot-plot"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { CLASSES, niceTicks, pct, signed } from "@/lib/domain"
import { asNumber, useFinalModel, useTable } from "@/lib/reports"

const MODEL_LABELS: Record<string, string> = {
  dummy_majority: "Always Category_1",
  lookup_vendor_account: "Vendor + account lookup",
  complement_nb: "Complement naive Bayes",
  linear_svc: "Linear SVM",
  logistic_p0: "Logistic regression",
  logistic_p05: "Logistic, class weights 0.5",
  logistic_p1: "Logistic, balanced weights",
  lightgbm_p0: "LightGBM",
  lightgbm_leaf20: "LightGBM, min leaf 20",
  probability_blend_p0: "Probability-average blend",
  logodds_blend_p0: "Log-odds blend, unweighted",
  logodds_blend_p025: "Log-odds blend, weights 0.25",
}
const modelLabel = (key: string) => MODEL_LABELS[key] ?? key

type SchemeRow = { model: string; scheme: string; accuracy: number; macro_f1_5: number }
const SCHEME_LABELS: Record<string, string> = {
  "random CV (leakage demo)": "Random split (leaky)",
  "grouped CV (primary)": "Grouped CV (used)",
  "out-of-time (train <= 2018-09)": "Out-of-time",
  "unseen-vendor CV": "New vendors only",
}

export function ValidationPanel() {
  const { data, error } = useTable<SchemeRow>("benchmark/validation_schemes.csv")
  const [model, setModel] = useState("logistic_p0")
  const items = useMemo(
    () =>
      (data?.rows ?? [])
        .filter((r) => r.model === model)
        .map((r) => ({
          key: r.scheme,
          label: SCHEME_LABELS[r.scheme] ?? r.scheme,
          value: asNumber(r.accuracy),
          highlight: r.scheme.startsWith("grouped"),
          detail: `${SCHEME_LABELS[r.scheme] ?? r.scheme}: accuracy ${pct(asNumber(r.accuracy))}, macro F1 ${asNumber(r.macro_f1_5).toFixed(3)}`,
        })),
    [data, model],
  )
  const score = (prefix: string) => items.find((i) => i.key.startsWith(prefix))?.value
  const random = score("random"), grouped = score("grouped"), unseen = score("unseen")
  const schemeFinding =
    random != null && grouped != null && unseen != null
      ? `For this model a random split reports ${pct(random)}, ${((random - grouped) * 100).toFixed(1)} points above grouped testing (${pct(grouped)}). On vendors never seen in training it scores ${pct(unseen)}, which is why new vendors are flagged for review.`
      : undefined
  return (
    <Panel
      title="How the score depends on the test design"
      explain={{ chart: "test-design", view: `Model: ${modelLabel(model)}` }}
      description="The same model's accuracy under four different ways of testing it."
      why={<>A random split puts lines matching the test lines into training. The figure reported everywhere else is <Term name="grouped cross-validation" />, computed here for three representative models.</>}
      action={
        <Select value={model} onValueChange={setModel}>
          <SelectTrigger size="sm" className="w-48 text-xs" aria-label="Model">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {["logistic_p0", "logodds_blend_p0", "logistic_p1"].map((m) => (
              <SelectItem key={m} value={m} className="text-xs">
                {modelLabel(m)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
      takeaway={schemeFinding}
    >
      {error && <LoadError error={error} />}
      {!data ? <ChartSkeleton height={160} /> : <DotPlot items={items} domain={[0.86, 0.98]} format={(v) => pct(v)} tickFormat={(v) => pct(v, 0)} ticks={[0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]} />}
    </Panel>
  )
}

type BoardRow = Record<string, number | string> & { model: string }
type Metric = "accuracy" | "macro_f1_5" | "recall"

export function LeaderboardPanel() {
  const { data, error } = useTable<BoardRow>("benchmark/leaderboard.csv")
  const [metric, setMetric] = useState<Metric>("accuracy")
  const [rareClass, setRareClass] = useState("Category_2")
  const final = useFinalModel().leaderboardKey
  const baseline = asNumber(data?.rows.find((r) => r.model === "dummy_majority")?.accuracy_mean ?? NaN)
  const items = useMemo(() => {
    const rows = data?.rows ?? []
    const value = (r: BoardRow) =>
      metric === "recall" ? asNumber(r[`recall_${rareClass}`]) : asNumber(r[`${metric}_mean`])
    const std = (r: BoardRow) => (metric === "recall" ? 0 : asNumber(r[`${metric}_std`]))
    return [...rows]
      .sort((a, b) => value(b) - value(a))
      .map((r) => ({
        key: r.model,
        label: modelLabel(r.model),
        value: value(r),
        low: std(r) ? value(r) - std(r) : undefined,
        high: std(r) ? value(r) + std(r) : undefined,
        highlight: r.model === final,
        detail:
          metric === "recall"
            ? `${modelLabel(r.model)}: finds ${pct(value(r))} of ${rareClass} rows`
            : metric === "accuracy"
              ? `${modelLabel(r.model)}: ${pct(value(r))} ± ${pct(std(r))} over 15 test folds`
              : `${modelLabel(r.model)}: ${pct(value(r))} ± ${pct(std(r))} across 3 seeds`,
      }))
  }, [data, metric, rareClass, final])
  const domain: [number, number] = metric === "accuracy" ? [0.86, 0.98] : [0, 1]
  return (
    <Panel
      title="Model comparison"
      explain={{ chart: "model-comparison", view: metric === "recall" ? `Recall for ${rareClass}` : metric === "accuracy" ? "Accuracy" : "Macro F1" }}
      description={<>Each candidate model's score on identical tests. The dot is the average; the line shows how much it varies between test runs. Switch between <Term name="accuracy" />, <Term name="macro F1" />, and <Term name="recall" /> for one category.</>}
      action={
        <div className="flex flex-wrap items-center gap-2">
          <Segmented
            label="Metric"
            value={metric}
            onChange={setMetric}
            options={[
              { value: "accuracy", label: "Accuracy" },
              { value: "macro_f1_5", label: "Macro F1" },
              { value: "recall", label: "Recall" },
            ]}
          />
          {metric === "recall" && (
            <Select value={rareClass} onValueChange={setRareClass}>
              <SelectTrigger size="sm" className="w-32 text-xs" aria-label="Category">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CLASSES.map((c) => (
                  <SelectItem key={c} value={c} className="text-xs">
                    {c}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      }
      takeaway="The log-odds blend (highlighted) leads on accuracy and macro F1. Logistic regression and LightGBM make different mistakes, so the blend beats both."
    >
      {error && <LoadError error={error} />}
      {!data ? (
        <ChartSkeleton height={300} />
      ) : (
        <DotPlot
          items={items}
          domain={domain}
          format={(v) => pct(v)}
          tickFormat={(v) => pct(v, 0)}
          ticks={metric === "accuracy" ? [0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98] : [0, 0.25, 0.5, 0.75, 1]}
          reference={metric === "accuracy" && Number.isFinite(baseline) ? { value: baseline, label: "always answering Category_1" } : undefined}
        />
      )}
    </Panel>
  )
}

const ABLATION_LABELS: Record<string, string> = {
  "drop col1_tokens": "Vendor words",
  "drop col1_value": "Whole vendor name",
  "drop col4_tokens": "Description words",
  "drop col6_tokens": "GL account words",
  "drop col6_value": "Whole GL account",
  "drop pair": "Vendor + account pair",
  "drop col7": "Document flag",
  "drop amount": "Amount",
  "drop amount_flags": "Credit / round flags",
  "drop period": "Posting month",
  "add col2 format features": "+ Invoice number format",
  "add token bigrams": "+ Word pairs",
}

export function AblationPanel() {
  const { data, error } = useTable<{ variant: string; delta_accuracy: number; delta_macro_f1_5: number }>(
    "benchmark/ablation.csv",
  )
  const [metric, setMetric] = useState<"delta_accuracy" | "delta_macro_f1_5">("delta_macro_f1_5")
  const rows = useMemo(
    () =>
      (data?.rows ?? [])
        .filter((r) => r.variant !== "full feature set")
        .map((r) => ({ name: ABLATION_LABELS[r.variant] ?? r.variant, value: asNumber(r[metric]) }))
        .sort((a, b) => a.value - b.value),
    [data, metric],
  )
  return (
    <Panel
      title="What each input is worth"
      explain={{ chart: "ablation", view: metric === "delta_accuracy" ? "Change in accuracy" : "Change in macro F1" }}
      description={<>How much the score changes when one group of inputs is taken away (or added). This is an <Term name="ablation" />, run with the logistic model.</>}
      action={
        <Segmented
          label="Metric"
          value={metric}
          onChange={setMetric}
          options={[
            { value: "delta_macro_f1_5", label: "Macro F1" },
            { value: "delta_accuracy", label: "Accuracy" },
          ]}
        />
      }
      takeaway="Dropping the description words or the posting month hurts most, mainly for the rare categories."
    >
      {error && <LoadError error={error} />}
      {!data ? (
        <ChartSkeleton height={320} />
      ) : (
        <ChartContainer config={{ value: { label: "Change", color: "var(--support)" } }} className="h-[320px] w-full">
          <BarChart data={rows} layout="vertical" margin={{ left: 4, right: 16 }}>
            <CartesianGrid horizontal={false} stroke="var(--grid)" />
            <XAxis
              type="number"
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => (metric === "delta_accuracy" ? `${signed(v * 100, 1)}pt` : signed(v, 2))}
            />
            <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} width={150} />
            <ReferenceLine x={0} stroke="var(--muted-foreground)" />
            <ChartTooltip
              cursor={{ fill: "var(--muted)", opacity: 0.4 }}
              content={
                <ChartTooltipContent
                  hideIndicator
                  formatter={(v) => (
                    <span className="tabular">
                      {metric === "delta_accuracy"
                        ? `${signed(asNumber(v) * 100, 2)} accuracy points`
                        : `${signed(asNumber(v), 3)} macro F1`}
                    </span>
                  )}
                />
              }
            />
            <Bar isAnimationActive={false} dataKey="value" radius={4} barSize={14}>
              {rows.map((r) => (
                <Cell key={r.name} fill={r.value < 0 ? "var(--against)" : "var(--support)"} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      )}
    </Panel>
  )
}

type DialRow = {
  model: string
  power: number
  cv_accuracy: number
  cv_macro_f1_5: number
  oot_majority_rows_predicted_rare: number
}

export function ClassWeightPanel() {
  const { data, error } = useTable<DialRow>("benchmark/class_weight_dial.csv")
  const [metric, setMetric] = useState<"cv_macro_f1_5" | "cv_accuracy" | "oot_majority_rows_predicted_rare">(
    "cv_macro_f1_5",
  )
  const rows = useMemo(() => {
    const byPower = new Map<number, Record<string, number>>()
    for (const r of data?.rows ?? []) {
      const power = asNumber(r.power)
      const entry = byPower.get(power) ?? { power }
      entry[r.model] = asNumber(r[metric])
      byPower.set(power, entry)
    }
    return [...byPower.values()].sort((a, b) => a.power - b.power)
  }, [data, metric])
  const percent = metric !== "oot_majority_rows_predicted_rare"
  const yTicks = useMemo(() => {
    const values = rows.flatMap((r) => Object.entries(r).filter(([k]) => k !== "power").map(([, v]) => v))
    return values.length ? niceTicks(Math.min(...values), Math.max(...values)) : undefined
  }, [rows])
  return (
    <Panel
      title="Class weights: rare-category recall against false alarms"
      explain={{ chart: "class-weights", view: metric === "cv_macro_f1_5" ? "Macro F1" : metric === "cv_accuracy" ? "Accuracy" : "False rare alarms" }}
      description={<>Scores as the model is told to pay more attention to rare categories (<Term name="class weights" />): 0 = no extra attention, 1 = every category weighted equally.</>}
      action={
        <Segmented
          label="Metric"
          value={metric}
          onChange={setMetric}
          options={[
            { value: "cv_macro_f1_5", label: "Macro F1" },
            { value: "cv_accuracy", label: "Accuracy" },
            { value: "oot_majority_rows_predicted_rare", label: "False alarms" },
          ]}
        />
      }
      takeaway="Power 0.25 (dashed line) gets most of the rare-category gain without extra false alarms."
    >
      {error && <LoadError error={error} />}
      {!data ? (
        <ChartSkeleton height={240} />
      ) : (
        <ChartContainer
          config={{
            logodds_blend: { label: "Log-odds blend", color: "var(--primary)" },
            logistic: { label: "Logistic regression", color: "var(--muted-foreground)" },
          }}
          className="h-[240px] w-full"
        >
          <LineChart data={rows} margin={{ left: -8, right: 12, top: 8 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="power" type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} />
            <YAxis
              tickLine={false}
              axisLine={false}
              domain={yTicks ? [yTicks[0], yTicks[yTicks.length - 1]] : ["auto", "auto"]}
              ticks={yTicks}
              tickFormatter={(v) => (percent ? pct(v, yTicks && yTicks[1] - yTicks[0] < 0.01 ? 1 : 0) : String(v))}
              allowDecimals={!percent ? false : true}
            />
            <ReferenceLine x={0.25} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(_, payload) => `Power ${payload?.[0]?.payload?.power}`}
                  formatter={(v, name) => (
                    <span className="flex w-full justify-between gap-4">
                      <span className="text-muted-foreground">{name === "logodds_blend" ? "Log-odds blend" : "Logistic"}</span>
                      <span className="tabular">{percent ? pct(asNumber(v)) : `${v} rows`}</span>
                    </span>
                  )}
                />
              }
            />
            <Line isAnimationActive={false} dataKey="logodds_blend" stroke="var(--color-logodds_blend)" strokeWidth={2} dot={{ r: 4 }} />
            <Line isAnimationActive={false} dataKey="logistic" stroke="var(--color-logistic)" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ChartContainer>
      )}
      <SeriesLegend
        items={[
          { label: "Log-odds blend", color: "var(--primary)" },
          { label: "Logistic regression", color: "var(--muted-foreground)" },
        ]}
      />
    </Panel>
  )
}
