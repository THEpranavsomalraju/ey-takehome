import { useCallback, useState } from "react"
import { toast } from "sonner"

import { Stat } from "@/components/blocks"
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
        toast.success(`${num(next.summary.rows)} rows classified in ${num(next.summary.timings_ms.total)} ms`)
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
            Every row gets a category, a confidence, and a review decision. Click any row to see
            exactly why, and ask the AI agent for a plain-language explanation.
          </p>
        </div>
        <UploadPanel busy={busy} onFile={onFile} onSample={onSample} />
        <ol className="grid gap-3 sm:grid-cols-3">
          {[
            ["Upload", "Any ledger file with the columns Col1 to Col7, or the holdout sample. It is checked and every line is classified in well under a second."],
            ["Review", "See how many lines were decided automatically and which need a person. Filter by category, warning, or confidence."],
            ["Explain", "Click a line to see what drove its category, which past invoices support it, and a plain-language explanation from the AI agent."],
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
    <div className="space-y-6">
      <div className="space-y-3">
        <FileBar upload={upload} />
        <UploadPanel busy={busy} onFile={onFile} onSample={onSample} compact />
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Rows classified" value={num(summary.rows)} detail={`${num(summary.report.rows_read)} rows read`} />
        <Stat
          label="Decided automatically"
          value={num(summary.rows - summary.review_required)}
          detail={`${pct(1 - reviewShare)} of rows`}
        />
        <Stat
          label="Need a person's review"
          value={num(summary.review_required)}
          detail={`${pct(reviewShare)} of rows`}
          hint="Confidence below the threshold, or a warning such as a new vendor or GL account."
        />
        <Stat
          label="Average confidence"
          value={pct(insights.mean_confidence)}
          detail={`Review threshold ${pct(insights.threshold, 0)}`}
        />
      </div>

      <IssuesList issues={summary.report.issues} fallbackReason={summary.fallback_reason} />
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
