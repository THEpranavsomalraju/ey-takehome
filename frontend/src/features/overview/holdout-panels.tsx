import { useMemo, useState } from "react"

import { ChartSkeleton, ClassLabel, LoadError, Panel, Segmented, Stat } from "@/components/blocks"
import { Heatmap } from "@/components/heatmap"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { CLASSES, num, pct, shortClass } from "@/lib/domain"
import { asNumber, useJson, useTable } from "@/lib/reports"

interface ModelScores {
  accuracy: number
  macro_f1_5: number
  ece: number
  review_share: number
  accuracy_auto: number
  accuracy_reviewed: number
}

interface Evaluation {
  champion: ModelScores
  fallback_logistic: ModelScores
  cv_estimate_champion: { cv_accuracy: number; cv_accuracy_std: number; macro_f1_5: number }
  rows: number
}

export function HeadlineStats() {
  const { data, error } = useJson<Evaluation>("holdout/holdout_evaluation.json")
  if (error) return <LoadError error={error} />
  if (!data) return <ChartSkeleton height={104} />
  const c = data.champion
  const cv = data.cv_estimate_champion
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <Stat
        label="Holdout accuracy"
        value={pct(c.accuracy)}
        emphasis
        detail={`${num(data.rows)} rows never used in development`}
        hint="Scored once, at the very end, with the final model exactly as trained. The fallback model scores 95.3%."
      />
      <Stat
        label="Expected accuracy on new data"
        value={pct(cv.cv_accuracy)}
        detail={`Grouped cross-validation, ±${pct(cv.cv_accuracy_std)}`}
        hint="Average over 15 grouped test folds. The honest estimate for future invoices; the holdout came in slightly higher."
      />
      <Stat
        label="Baseline"
        value="88.4%"
        detail="Always answering Category_1"
        hint="The share of the most common category. Every model must beat this."
      />
      <Stat
        label="Accuracy without review"
        value={pct(c.accuracy_auto)}
        detail={`${pct(c.review_share)} of holdout rows sent to review`}
        hint="Rows the model is confident about, with no risk flag, are decided automatically. The rest go to a person."
      />
    </div>
  )
}

type MatrixRow = Record<string, string | number> & { true_label: string }

export function ConfusionPanel() {
  const holdout = useTable<MatrixRow>("holdout/holdout_confusion_matrix.csv")
  const oof = useTable<MatrixRow>("selection/oof_confusion_matrix.csv")
  const [source, setSource] = useState<"holdout" | "oof">("holdout")
  const table = source === "holdout" ? holdout : oof
  const matrix = useMemo(() => {
    const rows = (table.data?.rows ?? []).filter((r) => CLASSES.some((c) => asNumber(r[c]) > 0))
    const totals = new Map(rows.map((r) => [r.true_label, CLASSES.reduce((s, c) => s + asNumber(r[c]), 0)]))
    return { rows, totals }
  }, [table.data])
  const cell = (r: string, c: string) => asNumber(matrix.rows.find((x) => x.true_label === r)?.[c] ?? 0)
  return (
    <Panel
      title="Where the model is right and wrong"
      description="Rows: true category. Columns: predicted category. Shading shows the share of each true category."
      action={
        <Segmented
          label="Data"
          value={source}
          onChange={setSource}
          options={[
            { value: "holdout", label: "Holdout" },
            { value: "oof", label: "Cross-validation" },
          ]}
        />
      }
      takeaway="Almost all errors are swaps between Category_1 and Category_2, whose vendors and accounts overlap. Category_3 is found reliably; Category_4 and Category_6 have only a handful of examples and are the hardest."
    >
      {table.error && <LoadError error={table.error} />}
      {!table.data ? (
        <ChartSkeleton height={240} />
      ) : (
        <Heatmap
          rows={matrix.rows.map((r) => r.true_label)}
          columns={[...CLASSES]}
          value={cell}
          shade={(r, c) => cell(r, c) / (matrix.totals.get(r) || 1)}
          format={(v) => (v ? num(v) : "")}
          highlightDiagonal
          rowHeader={(r) => <ClassLabel label={r} short />}
          columnHeader={(c) => shortClass(c)}
          describe={(r, c) => {
            const n = cell(r, c)
            const share = n / (matrix.totals.get(r) || 1)
            return r === c
              ? `${num(n)} ${r} rows predicted correctly (${pct(share)} of ${r})`
              : `${num(n)} ${r} rows predicted as ${c} (${pct(share)} of ${r})`
          }}
        />
      )}
    </Panel>
  )
}

type ReportRow = { class: string; precision: number; recall: number; f1: number; support: number }

export function PerClassPanel() {
  const holdout = useTable<ReportRow>("holdout/holdout_classification_report.csv")
  const oof = useTable<ReportRow>("selection/oof_classification_report.csv")
  const train = useTable<{ class: string; rows: number }>("eda/class_distribution.csv")
  const loading = !holdout.data || !oof.data || !train.data
  return (
    <Panel
      title="Quality per category"
      description="Recall: share of a category's rows the model finds. Precision: share of its predictions that are right."
      takeaway="Common categories are near perfect. Rare categories have 2 to 20 training rows, so their scores move a lot with each row; those predictions are best confirmed by a person."
    >
      {(holdout.error || oof.error) && <LoadError error={(holdout.error ?? oof.error) as Error} />}
      {loading ? (
        <ChartSkeleton height={240} />
      ) : (
        <Table className="text-xs">
          <TableHeader>
            <TableRow>
              <TableHead>Category</TableHead>
              <TableHead className="text-right">Training rows</TableHead>
              <TableHead className="text-right">Holdout rows</TableHead>
              <TableHead className="w-[28%]">Holdout recall</TableHead>
              <TableHead className="text-right">Holdout precision</TableHead>
              <TableHead className="text-right">CV recall</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {CLASSES.map((c) => {
              const h = holdout.data!.rows.find((r) => r.class === c)
              const o = oof.data!.rows.find((r) => r.class === c)
              const t = train.data!.rows.find((r) => r.class === c)
              const support = asNumber(h?.support ?? 0)
              return (
                <TableRow key={c}>
                  <TableCell><ClassLabel label={c} /></TableCell>
                  <TableCell className="tabular text-right">{num(asNumber(t?.rows ?? 0))}</TableCell>
                  <TableCell className="tabular text-right">{num(support)}</TableCell>
                  <TableCell>
                    {support ? (
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 flex-1 rounded-full bg-muted">
                          <div className="h-full rounded-full bg-primary" style={{ width: pct(asNumber(h!.recall), 0) }} />
                        </div>
                        <span className="tabular w-11 text-right">{pct(asNumber(h!.recall), 0)}</span>
                      </div>
                    ) : (
                      <span className="text-muted-foreground">no rows</span>
                    )}
                  </TableCell>
                  <TableCell className="tabular text-right">{support ? pct(asNumber(h!.precision), 0) : "-"}</TableCell>
                  <TableCell className="tabular text-right">{pct(asNumber(o?.recall ?? 0), 0)}</TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      )}
    </Panel>
  )
}
