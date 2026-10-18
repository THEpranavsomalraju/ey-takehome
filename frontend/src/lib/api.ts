// Typed client for the ledgerclf HTTP API (served by `python -m ledgerclf serve`).

export type Severity = "INFO" | "WARNING" | "ERROR"

export interface Issue {
  severity: Severity
  code: string
  message: string
  column: string | null
  row_numbers: number[]
}

export interface PredictionRow {
  row_number: number
  source_row: number
  Col1: string | null
  Col2: string | null
  Col3: string | null
  Col4: string | null
  Col5: string | null
  Col6: string | null
  Col7: string | null
  predicted_label: string
  confidence: number
  runner_up_label: string
  runner_up_probability: number
  review_required: boolean
  flags: string
  true_label?: string | null
  correct?: boolean | null
  [probability: `p_${string}`]: number
}

export interface Metrics {
  accuracy: number
  macro_f1_5: number
  balanced_accuracy: number
  ece: number
  labeled_rows: number
  unlabeled_rows: number
  per_class: { class: string; precision: number; recall: number; f1: number; support: number }[]
  confusion_matrix: Record<string, string | number>[]
}

export interface Summary {
  rows: number
  model_used: string
  fallback_reason: string | null
  review_required: number
  predicted_counts: Record<string, number>
  report: {
    issues: Issue[]
    rows_read: number
    rows_dropped_empty: number
    rows_predicted: number
    column_mapping: Record<string, string>
  }
  metrics: Metrics | null
  timings_ms: Record<string, number>
  read_info: {
    file_name: string
    detected_format: string
    sheet_name: string | null
    encoding: string | null
    delimiter: string | null
    header_row_index: number | null
  } | null
}

export interface Insights {
  threshold: number
  confidence_histogram: { low: number; high: number; auto: number; review: number }[]
  by_class: { label: string; rows: number; review: number }[]
  flags: { flag: string; rows: number }[]
  mean_confidence: number | null
}

export interface Page {
  offset: number
  limit: number
  matched: number
  total: number
  rows: PredictionRow[]
}

export interface Upload {
  upload_id: string
  summary: Summary
  insights: Insights
  page: Page
}

export type SortOrder = "row" | "confidence_asc" | "confidence_desc"

export interface RowQuery {
  q: string
  label: string | null
  review: boolean
  flag: string | null
  sort: SortOrder
}

export interface Contribution {
  group: string
  label: string
  contribution: number
}

export interface Evidence {
  group: string
  value: string
  seen_in_training: boolean
  training_rows: number
  class_counts: Record<string, number>
  share_predicted_class: number | null
}

export interface ExplanationPayload {
  row_number: number
  source_row: number
  input: Record<string, string | null>
  prediction: {
    label: string
    probability: number
    runner_up: { label: string; probability: number }
    logit_margin: number
    review_required: boolean
  }
  probabilities: Record<string, number>
  model: { name: string; linear_weight: number; temperature: number; fallback_active: boolean }
  contributions: {
    target: string
    total: number
    bias: number
    by_group: Contribution[]
    top_features: {
      group: string
      feature: string
      transformer: string
      present_in_row: boolean
      contribution: number
    }[]
  }
  evidence: Evidence[]
  flags: { code: string; message: string }[]
  context: {
    training_rows: number
    class_prior: Record<string, number>
    period_range: [string, string]
    amount_percentile: number | null
  }
  fallback_text: string
}

export interface AgentAnswer {
  text: string
  engine: "claude" | "offline"
  model: string | null
  tool_calls: { tool: string; input: Record<string, unknown>; output?: unknown; error?: string }[]
  fallback_reason: string | null
  seconds: number
}

export interface ChartAnswer {
  text: string | null
  engine: "claude" | "offline"
  model: string | null
  fallback_reason: string | null
  seconds: number
  cached: boolean
}

export interface WhatIfResult {
  changed: { column: string; value: string }
  original_prediction: { label: string; probability: number }
  new_prediction: { label: string; probability: number }
  new_probabilities: Record<string, number>
}

export interface ModelInfo {
  model_used: string
  fallback_active: boolean
  fallback_reason: string | null
  champion: Record<string, unknown> & { name: string; temperature: number }
  fallback: Record<string, unknown> & { name: string; temperature: number }
  training_rows: number
  created_utc: string
  review_policy: { threshold: number; severities: Record<string, string> }
  cv_summary: Record<string, number | string>
  library_versions: Record<string, string>
}

export interface AgentStatus {
  route: "direct" | "relay" | "offline"
  model: string | null
}

export interface HealthCheck {
  name: string
  status: "PASS" | "WARN" | "FAIL"
  detail: string
  fix: string
}

export interface Table<Row = Record<string, string | number | boolean | null>> {
  columns: string[]
  rows: Row[]
}

export class ApiError extends Error {
  code: string
  status: number

  constructor(message: string, code: string, status: number) {
    super(message)
    this.code = code
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch {
    throw new ApiError(
      "The server is not reachable. Start it with: python -m ledgerclf serve",
      "OFFLINE",
      0,
    )
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const message = body?.message ?? `Request failed (${response.status}).`
    throw new ApiError(message, body?.code ?? `HTTP_${response.status}`, response.status)
  }
  return body as T
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

export const api = {
  report: <T = Table>(file: string) => request<T>(`/api/reports/${file}`),
  model: () => request<ModelInfo>("/api/model"),
  agentStatus: () => request<AgentStatus>("/api/agent/status"),
  health: () => request<{ exit_code: number; checks: HealthCheck[] }>("/api/health"),
  predictFile: (file: File) => {
    const form = new FormData()
    form.append("file", file)
    return request<Upload>("/api/predict", { method: "POST", body: form })
  },
  predictSample: () => request<Upload>("/api/predict/sample", { method: "POST" }),
  rows: (uploadId: string, query: RowQuery, offset: number, limit: number) => {
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) })
    if (query.q) params.set("q", query.q)
    if (query.label) params.set("label", query.label)
    if (query.review) params.set("review", "true")
    if (query.flag) params.set("flag", query.flag)
    params.set("sort", query.sort)
    return request<Page>(`/api/predict/${uploadId}/rows?${params}`)
  },
  payload: (uploadId: string, row: number) =>
    request<ExplanationPayload>(`/api/explain/${uploadId}/${row}`),
  explain: (uploadId: string, row: number, useLlm: boolean, apiKey: string | null) =>
    post<{ answer: AgentAnswer; payload: ExplanationPayload }>("/api/explain", {
      upload_id: uploadId,
      row_number: row,
      use_llm: useLlm,
      api_key: apiKey || null,
    }),
  whatIf: (uploadId: string, row: number, column: string, value: string) =>
    post<WhatIfResult>("/api/whatif", { upload_id: uploadId, row_number: row, column, value }),
  explainChart: (chart: string, view: string, uploadId: string | null, useLlm: boolean, apiKey: string | null) =>
    post<ChartAnswer>("/api/explain-chart", {
      chart,
      view,
      upload_id: uploadId,
      use_llm: useLlm,
      api_key: apiKey || null,
    }),
  downloadUrl: (uploadId: string) => `/api/predict/${uploadId}/download`,
  sampleUrl: "/api/sample/holdout",
}
