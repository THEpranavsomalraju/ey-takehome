import { useRef, useState, type DragEvent } from "react"
import { DownloadIcon, FileSpreadsheetIcon, UploadIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Spinner } from "@/components/ui/spinner"
import { api } from "@/lib/api"
import { cn } from "@/lib/utils"

const ACCEPT = ".xlsx,.xls,.xlsm,.csv,.txt"

export function UploadPanel({
  busy,
  onFile,
  onSample,
  compact = false,
}: {
  busy: boolean
  onFile: (file: File) => void
  onSample: () => void
  compact?: boolean
}) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const drop = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files?.[0]
    if (file && !busy) onFile(file)
  }

  const picker = (
    <input
      ref={input}
      type="file"
      accept={ACCEPT}
      className="hidden"
      onChange={(e) => {
        const file = e.target.files?.[0]
        if (file) onFile(file)
        e.target.value = ""
      }}
    />
  )

  if (compact) {
    return (
      <div className="flex flex-wrap items-center gap-2">
        {picker}
        <Button size="sm" variant="outline" disabled={busy} onClick={() => input.current?.click()}>
          {busy ? <Spinner /> : <UploadIcon />}
          Upload another file
        </Button>
        <Button size="sm" variant="ghost" disabled={busy} onClick={onSample}>
          Use holdout sample
        </Button>
      </div>
    )
  }

  return (
    <Card
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={drop}
      className={cn("border-dashed transition-colors", dragging && "border-primary bg-primary/5")}
    >
      <CardContent className="flex flex-col items-center gap-5 px-6 py-12 text-center">
        {picker}
        <div className="flex size-12 items-center justify-center rounded-lg border bg-muted/40">
          {busy ? <Spinner className="size-5" /> : <FileSpreadsheetIcon className="size-5 text-muted-foreground" />}
        </div>
        <div className="space-y-1.5">
          <h3 className="text-base font-semibold">
            {busy ? "Validating and predicting..." : "Drop a ledger file to classify it"}
          </h3>
          <p className="mx-auto max-w-md text-sm text-muted-foreground">
            Excel (.xlsx, .xls) or CSV with the columns Col1 to Col7. Column order, extra columns,
            title rows, and most formatting problems are handled. A label column, if present, is
            scored.
          </p>
        </div>
        <div className="flex flex-wrap justify-center gap-2">
          <Button disabled={busy} onClick={() => input.current?.click()}>
            <UploadIcon />
            Choose file
          </Button>
          <Button variant="outline" disabled={busy} onClick={onSample}>
            Use the holdout sample (590 rows)
          </Button>
        </div>
        <a
          href={api.sampleUrl}
          className="inline-flex items-center gap-1.5 text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
        >
          <DownloadIcon className="size-3.5" />
          Download the holdout sample to try your own edits
        </a>
      </CardContent>
    </Card>
  )
}
