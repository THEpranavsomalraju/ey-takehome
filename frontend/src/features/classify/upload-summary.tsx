import { useState, type ReactNode } from "react"
import { AlertTriangleIcon, ChevronDownIcon, CircleCheckIcon, DownloadIcon, InfoIcon, OctagonXIcon } from "lucide-react"

import { ClassLabel, Stat } from "@/components/blocks"
import { Heatmap } from "@/components/heatmap"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { api, type Issue, type Upload } from "@/lib/api"
import { CLASSES, modelName, num, pct, shortClass } from "@/lib/domain"
import { cn } from "@/lib/utils"

const SEVERITY = {
  ERROR: { icon: OctagonXIcon, className: "text-critical", label: "Error" },
  WARNING: { icon: AlertTriangleIcon, className: "text-warning", label: "Warning" },
  INFO: { icon: InfoIcon, className: "text-muted-foreground", label: "Note" },
} as const

export function FileBar({ upload, children }: { upload: Upload; children?: ReactNode }) {
  const { summary, upload_id } = upload
  const info = summary.read_info
  const details = [
    info?.detected_format?.toUpperCase(),
    info?.sheet_name ? `sheet "${info.sheet_name}"` : null,
    info?.delimiter ? `delimiter "${info.delimiter}"` : null,
    `${num(summary.report.rows_predicted)} rows predicted`,
    `${num(summary.timings_ms.total)} ms`,
  ].filter(Boolean)
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="min-w-0 space-y-1">
        <h2 className="truncate text-lg font-semibold tracking-tight">{info?.file_name ?? "Upload"}</h2>
        <p className="text-xs text-muted-foreground">{details.join(" - ")}</p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {children}
        <Badge variant="outline" className="hidden font-normal sm:inline-flex">
          {modelName(summary.model_used)}
        </Badge>
        <Button size="sm" variant="outline" asChild>
          <a href={api.downloadUrl(upload_id)}>
            <DownloadIcon />
            Download predictions (CSV)
          </a>
        </Button>
      </div>
    </div>
  )
}

export function IssuesList({ issues, fallbackReason }: { issues: Issue[]; fallbackReason: string | null }) {
  const [open, setOpen] = useState(() => fallbackReason != null || issues.some((i) => i.severity !== "INFO"))
  const warnings = issues.filter((i) => i.severity !== "INFO").length + (fallbackReason ? 1 : 0)
  const shown = open ? issues : []
  const total = issues.length + (fallbackReason ? 1 : 0)
  return (
    <Card className="gap-0 py-0">
      <div>
        <div className="flex items-center justify-between gap-3 px-4 py-3">
          <div className="flex items-center gap-2 text-sm">
            {warnings ? (
              <AlertTriangleIcon className="size-4 text-warning" />
            ) : (
              <CircleCheckIcon className="size-4 text-good" />
            )}
            <span className="font-medium">
              {total === 0 ? "File read cleanly" : `${total} validation ${total === 1 ? "message" : "messages"}`}
            </span>
            {warnings > 0 && <span className="text-muted-foreground">({warnings} need attention)</span>}
          </div>
          {total > 0 && (
            <Button size="sm" variant="ghost" className="text-xs" aria-expanded={open} onClick={() => setOpen(!open)}>
              {open ? "Hide" : "Show"}
              <ChevronDownIcon className={cn("transition-transform", open && "rotate-180")} />
            </Button>
          )}
        </div>
        {open && (
          <ul className="divide-y border-t">
            {fallbackReason && (
              <IssueRow severity="WARNING" code="FALLBACK_MODEL" message={`Fallback model in use: ${fallbackReason}`} />
            )}
            {shown.map((issue, i) => (
              <IssueRow key={`${issue.code}-${i}`} {...issue} />
            ))}
          </ul>
        )}
      </div>
    </Card>
  )
}

function IssueRow({ severity, code, message }: Pick<Issue, "severity" | "code" | "message">) {
  const style = SEVERITY[severity] ?? SEVERITY.INFO
  return (
    <li className="flex gap-3 px-4 py-2.5 text-sm">
      <style.icon className={cn("mt-0.5 size-4 shrink-0", style.className)} aria-label={style.label} />
      <div className="min-w-0">
        <p className="leading-relaxed">{message}</p>
        <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">{code}</p>
      </div>
    </li>
  )
}

export function LabeledMetrics({ upload }: { upload: Upload }) {
  const metrics = upload.summary.metrics
  if (!metrics) return null
  const matrix = metrics.confusion_matrix
  const truths = matrix
    .map((r) => String(r.true_label))
    .filter((t) => CLASSES.some((c) => Number(matrix.find((r) => r.true_label === t)?.[c] ?? 0) > 0))
  const cell = (t: string, c: string) => Number(matrix.find((r) => r.true_label === t)?.[c] ?? 0)
  const total = (t: string) => CLASSES.reduce((s, c) => s + cell(t, c), 0)
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
      <div className="grid grid-cols-2 gap-3">
        <Stat label="Accuracy on this file" value={pct(metrics.accuracy)} emphasis detail={`${num(metrics.labeled_rows)} labeled rows`} />
        <Stat label="Macro F1 (5 categories)" value={metrics.macro_f1_5.toFixed(3)} detail="Every category weighted equally" />
        <Stat label="Balanced accuracy" value={pct(metrics.balanced_accuracy)} detail="Mean recall per category" />
        <Stat
          label="Calibration error"
          value={pct(metrics.ece, 2)}
          detail="Gap between stated and actual confidence"
        />
      </div>
      <Card className="gap-3">
        <CardContent className="space-y-2">
          <p className="text-sm font-medium">Confusion matrix for this file</p>
          <Heatmap
            rows={truths}
            columns={[...CLASSES]}
            value={cell}
            shade={(t, c) => cell(t, c) / (total(t) || 1)}
            format={(v) => (v ? num(v) : "")}
            highlightDiagonal
            rowHeader={(t) => <ClassLabel label={t} short />}
            columnHeader={(c) => shortClass(c)}
            describe={(t, c) => `${num(cell(t, c))} ${t} rows predicted as ${c}`}
          />
        </CardContent>
      </Card>
    </div>
  )
}
