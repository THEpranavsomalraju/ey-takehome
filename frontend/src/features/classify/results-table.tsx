import { useEffect, useState } from "react"
import { CheckIcon, ChevronLeftIcon, ChevronRightIcon, SearchIcon, XIcon } from "lucide-react"

import { ClassLabel } from "@/components/blocks"
import { CategoryHint } from "@/components/category-guide"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group"
import { Label } from "@/components/ui/label"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { api, type Insights, type Page, type PredictionRow, type RowQuery, type SortOrder } from "@/lib/api"
import { CLASSES, flagCodes, flagLabel, num, pct } from "@/lib/domain"
import { cn } from "@/lib/utils"

const PAGE_SIZE = 25
const ALL = "__all__"

export function ResultsTable({
  uploadId,
  insights,
  query,
  onQuery: setQuery,
  labeled,
  selectedRow,
  onSelect,
}: {
  uploadId: string
  insights: Insights
  query: RowQuery
  onQuery: (query: RowQuery) => void
  labeled: boolean
  selectedRow: number | null
  onSelect: (row: number) => void
}) {
  // The page position belongs to one query: a new filter or search starts at page 1.
  const [position, setPosition] = useState<{ query: RowQuery; offset: number }>({ query, offset: 0 })
  const offset = position.query === query ? position.offset : 0
  const setOffset = (next: number) => setPosition({ query, offset: next })
  const [page, setPage] = useState<Page | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState(query.q)
  const onQuery = setQuery

  useEffect(() => {
    const handle = setTimeout(() => {
      if (search !== query.q) onQuery({ ...query, q: search })
    }, 250)
    return () => clearTimeout(handle)
  }, [search, query, onQuery])

  useEffect(() => {
    let active = true
    api
      .rows(uploadId, query, offset, PAGE_SIZE)
      .then((next) => active && (setPage(next), setError(null)))
      .catch((e: Error) => active && setError(e.message))
    return () => {
      active = false
    }
  }, [uploadId, query, offset])

  const pages = page ? Math.max(1, Math.ceil(page.matched / PAGE_SIZE)) : 1
  const current = Math.floor(offset / PAGE_SIZE) + 1
  const filtered = query.q || query.label || query.review || query.flag

  return (
    <Card className="gap-0 py-0">
      <div className="flex flex-wrap items-center gap-2 border-b p-3">
        <InputGroup className="h-8 w-full sm:w-64">
          <InputGroupAddon>
            <SearchIcon />
          </InputGroupAddon>
          <InputGroupInput
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search vendor, account, text..."
            aria-label="Search rows"
            className="text-xs"
          />
        </InputGroup>
        <Select value={query.label ?? ALL} onValueChange={(v) => onQuery({ ...query, label: v === ALL ? null : v })}>
          <SelectTrigger size="sm" className="w-40 text-xs" aria-label="Predicted category">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL} className="text-xs">All categories</SelectItem>
            {CLASSES.map((c) => (
              <SelectItem key={c} value={c} className="text-xs">
                <ClassLabel label={c} />
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={query.flag ?? ALL} onValueChange={(v) => onQuery({ ...query, flag: v === ALL ? null : v })}>
          <SelectTrigger size="sm" className="w-44 text-xs" aria-label="Warning">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL} className="text-xs">Any warning</SelectItem>
            {insights.flags.map((f) => (
              <SelectItem key={f.flag} value={f.flag} className="text-xs">
                {flagLabel(f.flag)} ({num(f.rows)})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={query.sort} onValueChange={(v) => onQuery({ ...query, sort: v as SortOrder })}>
          <SelectTrigger size="sm" className="w-48 text-xs" aria-label="Sort">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="row" className="text-xs">File order</SelectItem>
            <SelectItem value="confidence_asc" className="text-xs">Least confident first</SelectItem>
            <SelectItem value="confidence_desc" className="text-xs">Most confident first</SelectItem>
          </SelectContent>
        </Select>
        <div className="flex items-center gap-2 pl-1">
          <Switch id="review-only" checked={query.review} onCheckedChange={(v) => onQuery({ ...query, review: v })} />
          <Label htmlFor="review-only" className="text-xs font-normal">Needs review only</Label>
        </div>
        {filtered && (
          <Button
            size="sm"
            variant="ghost"
            className="text-xs"
            onClick={() => {
              setSearch("")
              onQuery({ q: "", label: null, review: false, flag: null, sort: query.sort })
            }}
          >
            <XIcon />
            Clear filters
          </Button>
        )}
        <span className="ml-auto text-xs text-muted-foreground tabular">
          {page ? `${num(page.matched)} of ${num(page.total)} rows` : ""}
        </span>
      </div>

      {error && <p className="p-4 text-sm text-destructive">{error}</p>}
      <ScrollArea className="h-[min(70vh,44rem)]">
        <Table className="text-xs">
          <TableHeader className="sticky top-0 z-10 bg-card">
            <TableRow>
              <TableHead className="w-12 text-right">Row</TableHead>
              <TableHead>Vendor</TableHead>
              <TableHead className="text-right">Amount</TableHead>
              <TableHead className="hidden xl:table-cell">Description</TableHead>
              <TableHead>Period</TableHead>
              <TableHead>GL account</TableHead>
              <TableHead className="hidden xl:table-cell">Doc</TableHead>
              <TableHead>Prediction</TableHead>
              <TableHead className="w-32">Confidence</TableHead>
              <TableHead>Status</TableHead>
              {labeled && <TableHead>True label</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {!page
              ? Array.from({ length: 10 }, (_, i) => (
                  <TableRow key={i}>
                    <TableCell colSpan={labeled ? 11 : 10}>
                      <Skeleton className="h-5 w-full" />
                    </TableCell>
                  </TableRow>
                ))
              : page.rows.map((row) => (
                  <ResultRow
                    key={row.row_number}
                    row={row}
                    labeled={labeled}
                    selected={row.row_number === selectedRow}
                    onSelect={onSelect}
                  />
                ))}
            {page && page.rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={labeled ? 11 : 10} className="py-10 text-center text-muted-foreground">
                  No rows match these filters.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>

      <div className="flex items-center justify-between gap-2 border-t px-3 py-2 text-xs text-muted-foreground">
        <span className="hidden sm:inline">Click a row to see why it got its prediction.</span>
        <div className="flex items-center gap-2">
          <span className="tabular">
            Page {current} of {pages}
          </span>
          <Button size="icon-sm" variant="outline" aria-label="Previous page" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
            <ChevronLeftIcon />
          </Button>
          <Button size="icon-sm" variant="outline" aria-label="Next page" disabled={current >= pages} onClick={() => setOffset(offset + PAGE_SIZE)}>
            <ChevronRightIcon />
          </Button>
        </div>
      </div>
    </Card>
  )
}

function ResultRow({
  row,
  labeled,
  selected,
  onSelect,
}: {
  row: PredictionRow
  labeled: boolean
  selected: boolean
  onSelect: (row: number) => void
}) {
  const flags = flagCodes(row.flags)
  return (
    <TableRow
      tabIndex={0}
      onClick={() => onSelect(row.row_number)}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onSelect(row.row_number))}
      data-state={selected ? "selected" : undefined}
      className="cursor-pointer focus-visible:bg-muted/60 focus-visible:outline-none"
    >
      <TableCell className="tabular text-right text-muted-foreground">{row.row_number}</TableCell>
      <TableCell className="max-w-44 truncate" title={row.Col1 ?? ""}>{row.Col1 ?? "-"}</TableCell>
      <TableCell className="tabular text-right">{row.Col3 ?? "-"}</TableCell>
      <TableCell className="hidden max-w-48 truncate text-muted-foreground xl:table-cell" title={row.Col4 ?? ""}>{row.Col4 ?? "-"}</TableCell>
      <TableCell className="tabular text-muted-foreground">{row.Col5 ?? "-"}</TableCell>
      <TableCell className="max-w-40 truncate" title={row.Col6 ?? ""}>{row.Col6 ?? "-"}</TableCell>
      <TableCell className="hidden text-muted-foreground xl:table-cell">{row.Col7 ?? "-"}</TableCell>
      <TableCell>
        <CategoryHint label={row.predicted_label}>
          <ClassLabel label={row.predicted_label} />
        </CategoryHint>
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-14 rounded-full bg-muted">
            <div
              className={cn("h-full rounded-full", row.review_required ? "bg-warning" : "bg-foreground/60")}
              style={{ width: pct(row.confidence, 0) }}
            />
          </div>
          <span className="tabular w-10 text-right">{pct(row.confidence, 0)}</span>
        </div>
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-1">
          {row.review_required ? (
            <Badge variant="outline" className="border-warning/50 text-warning">Review</Badge>
          ) : (
            <Badge variant="outline" className="text-muted-foreground">Auto</Badge>
          )}
          {flags.length > 0 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge variant="secondary" className="tabular">{flags.length} {flags.length === 1 ? "warning" : "warnings"}</Badge>
              </TooltipTrigger>
              <TooltipContent className="text-xs">
                {flags.map((f) => (
                  <div key={f}>{flagLabel(f)}</div>
                ))}
              </TooltipContent>
            </Tooltip>
          )}
        </div>
      </TableCell>
      {labeled && (
        <TableCell>
          {row.true_label ? (
            <span className="inline-flex items-center gap-1.5">
              {row.correct ? (
                <CheckIcon className="size-3.5 text-good" aria-label="Correct" />
              ) : (
                <XIcon className="size-3.5 text-critical" aria-label="Wrong" />
              )}
              <ClassLabel label={row.true_label} short />
            </span>
          ) : (
            <span className="text-muted-foreground">-</span>
          )}
        </TableCell>
      )}
    </TableRow>
  )
}
