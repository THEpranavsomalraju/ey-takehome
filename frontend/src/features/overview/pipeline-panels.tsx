import { ArrowRightIcon, FileSpreadsheetIcon, GaugeIcon, LayersIcon, ShieldCheckIcon, SparklesIcon } from "lucide-react"
import { Term } from "@/components/term"

import { ChartSkeleton, LoadError, Panel, Stat } from "@/components/blocks"
import { Card, CardContent } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { api, type ModelInfo } from "@/lib/api"
import { columnName, flagLabel, num, pct } from "@/lib/domain"
import { useAsync } from "@/lib/use-async"
import { asNumber, useJson, useTable } from "@/lib/reports"

function stages(model: ModelInfo | null) {
  const weight = Number(model?.champion.linear_weight ?? 0.7)
  const critical = Object.entries(model?.review_policy.severities ?? {})
    .filter(([code, severity]) => severity === "critical" && code !== "LOW_CONFIDENCE")
    .map(([code]) => flagLabel(code).toLowerCase())
    .sort()
  return [
    {
      icon: FileSpreadsheetIcon,
      title: "Read and validate",
      text: "Any Excel or CSV layout: header detection, column mapping, encodings, delimiters. Problems become warnings; every row is kept.",
    },
    {
      icon: LayersIcon,
      title: "Features",
      text: "Words per column (TF-IDF), recurring vendors and accounts, amount as signed log plus credit and round flags, posting month.",
    },
    {
      icon: GaugeIcon,
      title: "Blend model",
      text: `${weight.toFixed(1)} x logistic regression + ${(1 - weight).toFixed(1)} x LightGBM in log-odds, calibrated with one temperature.`,
    },
    {
      icon: ShieldCheckIcon,
      title: "Review routing",
      text: model
        ? `Confidence below ${pct(model.review_policy.threshold, 0)}, or a critical warning (${critical.join(", ")}), sends the row to a person.`
        : "Low confidence or a critical warning sends the row to a person.",
    },
    {
      icon: SparklesIcon,
      title: "Explanation",
      text: "Exact per-input contributions plus training evidence; Claude turns them into plain language on request.",
    },
  ]
}

export function PipelineFlow() {
  const model = useAsync<ModelInfo>("model-info", api.model)
  const list = stages(model.data)
  return (
    <div className="grid gap-3 md:grid-cols-5">
      {list.map((stage, i) => (
        <div key={stage.title} className="relative">
          <Card className="h-full gap-2 py-4">
            <CardContent className="space-y-2 px-4">
              <div className="flex items-center gap-2">
                <span className="flex size-6 items-center justify-center rounded-md border text-[11px] tabular text-muted-foreground">
                  {i + 1}
                </span>
                <stage.icon className="size-4 text-muted-foreground" />
              </div>
              <p className="text-sm font-medium">{stage.title}</p>
              <p className="text-xs leading-relaxed text-muted-foreground">{stage.text}</p>
            </CardContent>
          </Card>
          {i < list.length - 1 && (
            <ArrowRightIcon className="absolute top-1/2 -right-3 z-10 hidden size-4 -translate-y-1/2 text-muted-foreground md:block" />
          )}
        </div>
      ))}
    </div>
  )
}

interface Catalog {
  dropped: Record<string, string>
  transformers: { name: string; columns: string | string[]; reason: string; decision: string; steps: { class: string }[] }[]
}

const TRANSFORMER_NAMES: Record<string, string> = {
  col1_tokens: "Vendor words",
  col1_value: "Whole vendor",
  col4_tokens: "Description words",
  col6_tokens: "GL account words",
  col6_value: "Whole GL account",
  pair: "Vendor + account pair",
  col7: "Document flag",
  amount: "Amount (signed log)",
  amount_flags: "Credit / round flags",
  period: "Posting month",
}

export function FeatureCatalogPanel() {
  const catalog = useJson<Catalog>("preprocessing/transformer_catalog.json")
  const space = useTable<{ transformer: string; features: number; mean_nonzero_per_row: number }>(
    "preprocessing/feature_space_summary.csv",
  )
  const loading = !catalog.data || !space.data
  const total = (space.data?.rows ?? []).reduce((s, r) => s + asNumber(r.features), 0)
  return (
    <Panel
      title="Feature pipeline"
      explain={{ chart: "feature-pipeline" }}
      description={<>Every group of inputs the model computes from a spreadsheet row, and how many numbers each produces ({num(total)} with single words; the final model adds word pairs). Text uses <Term name="TF-IDF" />.</>}
      why="It documents exactly what the model sees, and that the same steps run for training and for every uploaded file."
    >
      {(catalog.error || space.error) && <LoadError error={(catalog.error ?? space.error) as Error} />}
      {loading ? (
        <ChartSkeleton height={320} />
      ) : (
        <Table className="text-xs">
          <TableHeader>
            <TableRow>
              <TableHead>Feature group</TableHead>
              <TableHead>Input</TableHead>
              <TableHead className="text-right">Features</TableHead>
              <TableHead>Why</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {catalog.data!.transformers.map((t) => {
              const s = space.data!.rows.find((r) => r.transformer === t.name)
              const columns = Array.isArray(t.columns) ? t.columns : [t.columns]
              return (
                <TableRow key={t.name}>
                  <TableCell className="font-medium">
                    <Tooltip>
                      <TooltipTrigger className="text-left underline decoration-dotted underline-offset-4">
                        {TRANSFORMER_NAMES[t.name] ?? t.name}
                      </TooltipTrigger>
                      <TooltipContent className="text-xs">{t.steps.map((st) => st.class).join(" -> ")}</TooltipContent>
                    </Tooltip>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{columns.map(columnName).join(" + ")}</TableCell>
                  <TableCell className="tabular text-right">{num(asNumber(s?.features ?? 0))}</TableCell>
                  <TableCell className="max-w-[28rem] whitespace-normal text-muted-foreground">{t.reason}</TableCell>
                </TableRow>
              )
            })}
            <TableRow>
              <TableCell className="font-medium">Invoice number</TableCell>
              <TableCell className="text-muted-foreground">Invoice no.</TableCell>
              <TableCell className="tabular text-right">0</TableCell>
              <TableCell className="whitespace-normal text-muted-foreground">{catalog.data!.dropped.Col2}</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      )}
    </Panel>
  )
}

interface Nested {
  outer_accuracy_mean: number
  outer_accuracy_std: number
  non_nested_best_accuracy: number
  optimism_gap: number
}

export function TuningStats() {
  const nested = useJson<Nested>("tuning/nested_cv.json")
  const linear = useTable("tuning/linear_trials.csv")
  const tree = useTable("tuning/tree_trials.csv")
  const blend = useTable("tuning/blend_trials.csv")
  if (nested.error) return <LoadError error={nested.error} />
  if (!nested.data || !linear.data || !tree.data || !blend.data) return <ChartSkeleton height={104} />
  const trials = linear.data.rows.length + tree.data.rows.length + blend.data.rows.length
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <Stat label="Logged tuning trials" value={num(trials)} detail="Logistic, LightGBM, and blend searches" />
      <Stat
        label="Nested CV accuracy"
        value={pct(nested.data.outer_accuracy_mean)}
        detail={`±${pct(nested.data.outer_accuracy_std)}, tuning repeated per fold`}
        hint="Repeats the whole search inside each outer fold, so the score is not flattered by picking the best of many trials."
      />
      <Stat
        label="Optimism from tuning"
        value={`${(nested.data.optimism_gap * 100).toFixed(2)} pt`}
        detail="Picking the best trial did not inflate the score"
      />
      <Stat
        label="Decision rule"
        value="Fixed in advance"
        detail="Best accuracy among models that pass macro F1 and out-of-time floors"
      />
    </div>
  )
}
