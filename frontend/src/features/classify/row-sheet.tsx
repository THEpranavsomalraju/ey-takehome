import { useEffect, useState } from "react"
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine, XAxis, YAxis } from "recharts"
import { AlertTriangleIcon, ChevronDownIcon, SparklesIcon, WandSparklesIcon } from "lucide-react"
import { toast } from "sonner"

import { ClassLabel, Segmented } from "@/components/blocks"
import { Markdown } from "@/components/markdown"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { Spinner } from "@/components/ui/spinner"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api, type AgentAnswer, type ExplanationPayload, type WhatIfResult } from "@/lib/api"
import { CLASSES, CLASS_COLORS, COLUMNS, columnName, num, pct, signed } from "@/lib/domain"
import { useSettings } from "@/lib/settings"
import { cn } from "@/lib/utils"

export function RowSheet({
  uploadId,
  row,
  open,
  onClose,
}: {
  uploadId: string
  row: number | null
  open: boolean
  onClose: () => void
}) {
  const key = `${uploadId}:${row}`
  const [loaded, setLoaded] = useState<{ key: string; payload?: ExplanationPayload; error?: string }>({ key: "" })
  const payload = loaded.key === key ? (loaded.payload ?? null) : null
  const error = loaded.key === key ? (loaded.error ?? null) : null

  useEffect(() => {
    if (row == null) return
    let active = true
    api
      .payload(uploadId, row)
      .then((p) => active && setLoaded({ key, payload: p }))
      .catch((e: Error) => active && setLoaded({ key, error: e.message }))
    return () => {
      active = false
    }
  }, [uploadId, row, key])

  return (
    <Sheet open={open && row != null} onOpenChange={(next) => !next && onClose()}>
      <SheetContent className="w-full gap-0 p-0 data-[side=right]:w-full data-[side=right]:sm:max-w-2xl">
        <SheetHeader className="border-b px-6 py-4">
          <SheetTitle>Row {row}</SheetTitle>
          <SheetDescription>
            {payload ? `Spreadsheet row ${payload.source_row} of the uploaded file` : "Loading the explanation..."}
          </SheetDescription>
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
          <div className="space-y-6 px-6 py-5">
            {error && <p className="text-sm text-destructive">{error}</p>}
            {!payload && !error && <Skeleton className="h-96 w-full" />}
            {payload && row != null && (
              <>
                <PredictionHeader payload={payload} />
                <AgentSection key={key} uploadId={uploadId} row={row} />
                <Separator />
                <DriversSection payload={payload} />
                <ProbabilitySection payload={payload} />
                <EvidenceSection payload={payload} />
                <WhatIfSection key={key} uploadId={uploadId} row={row} payload={payload} />
                <InputsSection payload={payload} />
              </>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}

function PredictionHeader({ payload }: { payload: ExplanationPayload }) {
  const p = payload.prediction
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xl font-semibold tracking-tight">
          <ClassLabel label={p.label} />
        </span>
        <span className="tabular text-xl text-muted-foreground">{pct(p.probability)}</span>
        {p.review_required ? (
          <Badge variant="outline" className="border-warning/50 text-warning">Needs review</Badge>
        ) : (
          <Badge variant="outline" className="text-muted-foreground">Decided automatically</Badge>
        )}
      </div>
      <p className="text-sm text-muted-foreground">
        Runner-up: {p.runner_up.label} at {pct(p.runner_up.probability)}. Model: {payload.model.fallback_active ? "fallback logistic regression" : "blend of logistic regression and LightGBM"}.
      </p>
      {payload.flags.length > 0 && (
        <ul className="space-y-1.5 rounded-md border border-warning/30 bg-warning/5 px-3 py-2.5">
          {payload.flags.map((f) => (
            <li key={f.code} className="flex gap-2 text-sm">
              <AlertTriangleIcon className="mt-0.5 size-4 shrink-0 text-warning" />
              <span>{f.message}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function AgentSection({ uploadId, row }: { uploadId: string; row: number }) {
  const { apiKey, useLlm } = useSettings()
  const [answer, setAnswer] = useState<AgentAnswer | null>(null)
  const [busy, setBusy] = useState(false)

  const ask = () => {
    setBusy(true)
    api
      .explain(uploadId, row, useLlm, apiKey || null)
      .then((r) => setAnswer(r.answer))
      .catch((e: Error) => toast.error(e.message))
      .finally(() => setBusy(false))
  }

  return (
    <section className="space-y-3 rounded-lg border bg-muted/20 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-medium">Explanation</h3>
          <p className="text-xs text-muted-foreground">
            A plain-language summary, the exact facts, and what to check. Every number is verified against the model.
          </p>
        </div>
        <Button size="sm" onClick={ask} disabled={busy}>
          {busy ? <Spinner /> : <SparklesIcon />}
          {answer ? "Explain again" : "Explain with AI"}
        </Button>
      </div>
      {busy && !answer && (
        <div className="space-y-2">
          <Skeleton className="h-4 w-11/12" />
          <Skeleton className="h-4 w-10/12" />
          <Skeleton className="h-4 w-8/12" />
        </div>
      )}
      {answer && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge variant="outline" className={cn(answer.engine === "claude" && "border-primary/50 text-primary")}>
              {answer.engine === "claude" ? `Claude (${answer.model})` : "Offline explanation"}
            </Badge>
            <span className="tabular text-muted-foreground">{answer.seconds.toFixed(1)} s</span>
            {answer.fallback_reason && (
              <span className="text-muted-foreground">Used the offline engine because: {answer.fallback_reason}</span>
            )}
          </div>
          <Markdown text={answer.text} />
          {answer.tool_calls.length > 0 && (
            <Collapsible>
              <CollapsibleTrigger className="group inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
                <ChevronDownIcon className="size-3.5 transition-transform group-data-[state=open]:rotate-180" />
                Evidence the agent looked up ({answer.tool_calls.length} {answer.tool_calls.length === 1 ? "call" : "calls"})
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2 space-y-1.5">
                {answer.tool_calls.map((call, i) => (
                  <div key={i} className="rounded-md border bg-background/40 px-3 py-2 font-mono text-[11px]">
                    <span className="text-foreground">{call.tool}</span>
                    <span className="text-muted-foreground">({JSON.stringify(call.input)})</span>
                    {call.error && <span className="ml-2 text-critical">{call.error}</span>}
                  </div>
                ))}
              </CollapsibleContent>
            </Collapsible>
          )}
        </div>
      )}
    </section>
  )
}

function DriversSection({ payload }: { payload: ExplanationPayload }) {
  const [view, setView] = useState<"inputs" | "features">("inputs")
  const { label, runner_up } = payload.prediction
  const data =
    view === "inputs"
      ? payload.contributions.by_group.map((g) => ({ name: g.label.charAt(0).toUpperCase() + g.label.slice(1), value: g.contribution }))
      : payload.contributions.top_features.map((f) => ({ name: `${groupName(f.group)}: ${f.feature.replace(/^(token|value|doc|pair):/, "")}`, value: f.contribution }))
  const height = Math.max(160, data.length * 34 + 40)
  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-medium">What drove the prediction</h3>
          <p className="text-xs text-muted-foreground">
            Exact contribution of each input, in log-odds of {label} versus {runner_up.label}. Blue supports {label}, red pulls toward {runner_up.label}.
          </p>
        </div>
        <Segmented
          label="Detail level"
          value={view}
          onChange={setView}
          options={[
            { value: "inputs", label: "By input" },
            { value: "features", label: "Top features" },
          ]}
        />
      </div>
      <ChartContainer config={{ value: { label: "Contribution" } }} className="w-full" style={{ height }}>
        <BarChart data={data} layout="vertical" margin={{ left: 4, right: 40 }}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" tickLine={false} axisLine={false} tickFormatter={(v) => signed(v, 1)} />
          <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} width={190} interval={0} />
          <ReferenceLine x={0} stroke="var(--muted-foreground)" />
          <ChartTooltip
            cursor={{ fill: "var(--muted)", opacity: 0.4 }}
            content={
              <ChartTooltipContent
                hideIndicator
                formatter={(v) => (
                  <span className="tabular">
                    {signed(Number(v), 3)} log-odds {Number(v) >= 0 ? `toward ${label}` : `toward ${runner_up.label}`}
                  </span>
                )}
              />
            }
          />
          <Bar isAnimationActive={false} dataKey="value" radius={4} barSize={14}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.value >= 0 ? "var(--support)" : "var(--against)"} />
            ))}
            <LabelList dataKey="value" position="right" className="fill-muted-foreground text-[10px]" formatter={(v: unknown) => signed(Number(v), 2)} />
          </Bar>
        </BarChart>
      </ChartContainer>
      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="rounded-md border px-2 py-2">
          <div className="text-muted-foreground">Inputs combined</div>
          <div className="tabular text-sm font-medium">{signed(payload.contributions.total - payload.contributions.bias)}</div>
        </div>
        <div className="rounded-md border px-2 py-2">
          <div className="text-muted-foreground">Category base rate</div>
          <div className="tabular text-sm font-medium">{signed(payload.contributions.bias)}</div>
        </div>
        <div className="rounded-md border px-2 py-2">
          <div className="text-muted-foreground">Total margin</div>
          <div className="tabular text-sm font-medium">{signed(payload.contributions.total)}</div>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        The base rate is how common {label} is compared with {runner_up.label}. Inputs plus base rate equal the model's margin exactly.
      </p>
    </section>
  )
}

function ProbabilitySection({ payload }: { payload: ExplanationPayload }) {
  return (
    <section className="space-y-2">
      <h3 className="text-sm font-medium">Probability per category</h3>
      <div className="space-y-1.5">
        {CLASSES.map((c) => {
          const value = payload.probabilities[c] ?? 0
          return (
            <div key={c} className="grid grid-cols-[7rem_1fr_3.5rem] items-center gap-3 text-xs">
              <ClassLabel label={c} />
              <div className="h-2 rounded-full bg-muted">
                <div className="h-full rounded-full" style={{ width: pct(value, 1), backgroundColor: CLASS_COLORS[c] }} />
              </div>
              <span className="tabular text-right">{pct(value)}</span>
            </div>
          )
        })}
      </div>
    </section>
  )
}

function EvidenceSection({ payload }: { payload: ExplanationPayload }) {
  const label = payload.prediction.label
  return (
    <section className="space-y-2">
      <h3 className="text-sm font-medium">Evidence from past invoices</h3>
      <p className="text-xs text-muted-foreground">How training invoices with the same values were labeled.</p>
      <Table className="text-xs">
        <TableHeader>
          <TableRow>
            <TableHead>Input</TableHead>
            <TableHead>Value</TableHead>
            <TableHead className="text-right">Past invoices</TableHead>
            <TableHead className="w-36">Share {label}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {payload.evidence.map((e, i) => (
            <TableRow key={`${e.group}-${i}`}>
              <TableCell className="text-muted-foreground">{evidenceName(e)}</TableCell>
              <TableCell className="max-w-52 truncate" title={evidenceValue(e)}>{evidenceValue(e)}</TableCell>
              <TableCell className="tabular text-right">{e.seen_in_training ? num(e.training_rows) : <Badge variant="outline" className="text-warning">new</Badge>}</TableCell>
              <TableCell>
                {e.share_predicted_class == null ? (
                  <span className="text-muted-foreground">-</span>
                ) : (
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 flex-1 rounded-full bg-muted">
                      <div className="h-full rounded-full" style={{ width: pct(e.share_predicted_class, 0), backgroundColor: CLASS_COLORS[label] }} />
                    </div>
                    <span className="tabular w-9 text-right">{pct(e.share_predicted_class, 0)}</span>
                  </div>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  )
}

const GROUP_NAMES: Record<string, string> = {
  Col1: "Vendor",
  Col3: "Amount",
  Col4: "Description",
  Col5: "Posting month",
  Col6: "GL account",
  Col7: "Doc flag",
  "Col1+Col6": "Vendor + account",
}
const groupName = (group: string) => GROUP_NAMES[group] ?? columnName(group)
const TOKEN = /^token (.+) in Col\d$/
const evidenceName = (e: { group: string; value: string }) =>
  TOKEN.test(e.value) ? `${groupName(e.group)} word` : groupName(e.group)
const evidenceValue = (e: { value: string }) => e.value.match(TOKEN)?.[1] ?? e.value.replace(" || ", " + ")

const WHAT_IF_COLUMNS = COLUMNS.filter((c) => c.key !== "Col2")

function WhatIfSection({ uploadId, row, payload }: { uploadId: string; row: number; payload: ExplanationPayload }) {
  const [column, setColumn] = useState("Col6")
  const [value, setValue] = useState(payload.input.Col6 ?? "")
  const [result, setResult] = useState<WhatIfResult | null>(null)
  const [busy, setBusy] = useState(false)

  const chooseColumn = (next: string) => {
    setColumn(next)
    setValue(payload.input[next] ?? "")
    setResult(null)
  }

  const run = () => {
    setBusy(true)
    api
      .whatIf(uploadId, row, column, value)
      .then(setResult)
      .catch((e: Error) => toast.error(e.message))
      .finally(() => setBusy(false))
  }

  const changed = result && result.new_prediction.label !== result.original_prediction.label
  return (
    <section className="space-y-3 rounded-lg border p-4">
      <div>
        <h3 className="flex items-center gap-2 text-sm font-medium">
          <WandSparklesIcon className="size-4 text-muted-foreground" />
          What if?
        </h3>
        <p className="text-xs text-muted-foreground">Change one input and re-score this row with the same model. Useful to test a suspected miscoding.</p>
      </div>
      <form
        className="flex flex-wrap items-end gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          run()
        }}
      >
        <div className="space-y-1">
          <Label htmlFor="what-if-column" className="text-xs">Input</Label>
          <Select value={column} onValueChange={chooseColumn}>
            <SelectTrigger id="what-if-column" size="sm" className="w-36 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {WHAT_IF_COLUMNS.map((c) => (
                <SelectItem key={c.key} value={c.key} className="text-xs">
                  {c.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="min-w-48 flex-1 space-y-1">
          <Label htmlFor="what-if-value" className="text-xs">New value</Label>
          <Input id="what-if-value" value={value} onChange={(e) => setValue(e.target.value)} className="h-8 text-xs" maxLength={1000} />
        </div>
        <Button type="submit" size="sm" variant="secondary" disabled={busy}>
          {busy && <Spinner />}
          Re-score
        </Button>
      </form>
      {result && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md bg-muted/40 px-3 py-2 text-sm">
          <ClassLabel label={result.original_prediction.label} />
          <span className="tabular text-muted-foreground">{pct(result.original_prediction.probability)}</span>
          <span className="text-muted-foreground">-&gt;</span>
          <ClassLabel label={result.new_prediction.label} />
          <span className="tabular font-medium">{pct(result.new_prediction.probability)}</span>
          <Badge variant="outline" className={cn(changed ? "border-primary/50 text-primary" : "text-muted-foreground")}>
            {changed ? "Category changes" : "Same category"}
          </Badge>
        </div>
      )}
    </section>
  )
}

function InputsSection({ payload }: { payload: ExplanationPayload }) {
  return (
    <section className="space-y-2">
      <h3 className="text-sm font-medium">Inputs as read</h3>
      <dl className="grid grid-cols-[8rem_1fr] gap-x-4 gap-y-1.5 text-xs">
        {COLUMNS.map((c) => (
          <div key={c.key} className="contents">
            <dt className="text-muted-foreground">
              {c.name} <span className="font-mono text-[10px]">({c.key})</span>
            </dt>
            <dd className="break-words">{payload.input[c.key] ?? "-"}</dd>
          </div>
        ))}
      </dl>
      <p className="text-xs text-muted-foreground">
        Training covered {payload.context.period_range[0]} to {payload.context.period_range[1]}.
        {payload.context.amount_percentile != null &&
          ` This amount is higher than ${pct(payload.context.amount_percentile, 0)} of training amounts.`}
      </p>
    </section>
  )
}
