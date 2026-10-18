import { useCallback, useState } from "react"
import { toast } from "sonner"

import { Card, CardContent } from "@/components/ui/card"
import { api, type RowQuery, type Upload } from "@/lib/api"
import { num, pct } from "@/lib/domain"
import { ConfidenceChart, FlagsChart, PredictedClassesChart } from "./insight-charts"
import { ResultsTable } from "./results-table"
import { RowSheet } from "./row-sheet"
import { UploadPanel } from "./upload-panel"
import { FileBar, IssuesList, LabeledMetrics } from "./upload-summary"

const EMPTY_QUERY: RowQuery = { q: "", label: null, review: false, flag: null, sort: "row" }

export function ClassifyTab() {
  const [upload, setUpload] = useState<Upload | null>(null)
  const [busy, setBusy] = useState(false)
  const [query, setQuery] = useState<RowQuery>(EMPTY_QUERY)
  const [selected, setSelected] = useState<number | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [autoExplain, setAutoExplain] = useState(false)

  const run = useCallback((request: () => Promise<Upload>) => {
    setBusy(true)
    request()
      .then((next) => {
        setUpload(next)
        setQuery(EMPTY_QUERY)
        setSelected(null)
        setDrawerOpen(false)
      })
      .catch((e: Error) => toast.error(e.message, { duration: 10000 }))
      .finally(() => setBusy(false))
  }, [])

  const onFile = (file: File) => run(() => api.predictFile(file))
  const onSample = () => run(() => api.predictSample())

  if (!upload) {
    return (
      <div className="mx-auto max-w-3xl space-y-6 py-6">
        <div className="space-y-2 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">Classify a ledger</h1>
          <p className="text-sm text-muted-foreground">
            Upload a file to get a category, a confidence, and a review decision for each line.
            Explain on a line shows what drove the prediction.
          </p>
        </div>
        <UploadPanel busy={busy} onFile={onFile} onSample={onSample} />
        <ol className="grid gap-3 sm:grid-cols-3">
          {[
            ["Upload", "An Excel or CSV ledger with the seven input columns (Col1 to Col7 or business names like Vendor and Amount), or the holdout sample."],
            ["Review", "See how many lines were decided automatically and which need a person. Filter by category, warning, or confidence."],
            ["Explain", "Click Explain on a line to see what drove its category, which past invoices support it, and a short explanation from the AI."],
          ].map(([title, text], i) => (
            <li key={title} className="space-y-1 rounded-lg border p-3">
              <p className="text-sm font-medium">
                <span className="tabular mr-2 text-muted-foreground">{i + 1}</span>
                {title}
              </p>
              <p className="text-xs leading-relaxed text-muted-foreground">{text}</p>
            </li>
          ))}
        </ol>
      </div>
    )
  }

  const { summary, insights } = upload
  const reviewShare = summary.rows ? summary.review_required / summary.rows : 0
  return (
    <div className="space-y-4">
      <FileBar upload={upload}>
        <UploadPanel busy={busy} onFile={onFile} onSample={onSample} compact />
      </FileBar>

      <Card className="py-0">
        <CardContent className="grid grid-cols-2 divide-border px-0 lg:grid-cols-4 lg:divide-x">
          {[
            ["Lines classified", num(summary.rows), `${num(summary.timings_ms.total)} ms`],
            ["Decided automatically", num(summary.rows - summary.review_required), pct(1 - reviewShare)],
            ["Need a person's review", num(summary.review_required), pct(reviewShare)],
            ["Average confidence", pct(insights.mean_confidence), `threshold ${pct(insights.threshold, 0)}`],
          ].map(([label, value, detail]) => (
            <div key={label} className="flex items-baseline justify-between gap-3 px-4 py-3">
              <span className="text-xs text-muted-foreground">{label}</span>
              <span className="tabular text-lg font-semibold">
                {value} <span className="text-xs font-normal text-muted-foreground">{detail}</span>
              </span>
            </div>
          ))}
        </CardContent>
      </Card>

      <IssuesList issues={summary.report.issues} fallbackReason={summary.fallback_reason} />

      <ResultsTable
        key={upload.upload_id}
        uploadId={upload.upload_id}
        insights={insights}
        query={query}
        onQuery={setQuery}
        labeled={!!summary.metrics}
        selectedRow={drawerOpen ? selected : null}
        onSelect={(row) => {
          setSelected(row)
          setAutoExplain(false)
          setDrawerOpen(true)
        }}
        onExplain={(row) => {
          setSelected(row)
          setAutoExplain(true)
          setDrawerOpen(true)
        }}
      />

      <div className="space-y-4 pt-4">
        <h2 className="text-base font-semibold tracking-tight">Summary of this file</h2>
        <LabeledMetrics upload={upload} />
        <div className="grid gap-4 lg:grid-cols-3">
          <PredictedClassesChart
            uploadId={upload.upload_id}
            insights={insights}
            selected={query.label}
            onSelect={(label) => setQuery({ ...query, label })}
          />
          <ConfidenceChart uploadId={upload.upload_id} insights={insights} />
          <FlagsChart uploadId={upload.upload_id} insights={insights} selected={query.flag} onSelect={(flag) => setQuery({ ...query, flag })} />
        </div>
      </div>

      <RowSheet
        uploadId={upload.upload_id}
        row={selected}
        open={drawerOpen}
        autoExplain={autoExplain}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  )
}
