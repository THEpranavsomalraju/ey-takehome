// Shared vocabulary of the dashboard: categories, columns, flags, and number formats.

export const CLASSES = [
  "Category_1",
  "Category_2",
  "Category_3",
  "Category_4",
  "Category_5",
  "Category_6",
] as const

// Fixed order, never cycled: a category keeps its color on every chart.
export const CLASS_COLORS: Record<string, string> = {
  Category_1: "var(--chart-1)",
  Category_2: "var(--chart-2)",
  Category_3: "var(--chart-3)",
  Category_4: "var(--chart-4)",
  Category_5: "var(--chart-5)",
  Category_6: "var(--chart-6)",
}

export const shortClass = (label: string) => label.replace("Category_", "C")

export const COLUMNS = [
  { key: "Col1", name: "Vendor" },
  { key: "Col2", name: "Invoice no." },
  { key: "Col3", name: "Amount" },
  { key: "Col4", name: "Description" },
  { key: "Col5", name: "Period" },
  { key: "Col6", name: "GL account" },
  { key: "Col7", name: "Doc flag" },
] as const

const DERIVED: Record<string, string> = { amount: "Amount", period: "Posting month" }
export const columnName = (key: string) => COLUMNS.find((c) => c.key === key)?.name ?? DERIVED[key] ?? key

// Plain-English names for the flag codes the API returns.
export const FLAG_LABELS: Record<string, string> = {
  LOW_CONFIDENCE: "Low confidence",
  UNSEEN_VENDOR: "New vendor",
  UNSEEN_ACCOUNT: "New GL account",
  UNSEEN_VENDOR_ACCOUNT_PAIR: "New vendor + account",
  UNSEEN_DOC_TYPE: "New doc flag",
  ALL_TOKENS_UNSEEN: "Unfamiliar text",
  TEXT_MISSING: "Missing vendor/account",
  AMOUNT_IMPUTED: "Missing amount",
  AMOUNT_OUT_OF_RANGE: "Unusual amount",
  DATE_IMPUTED: "Missing date",
  DATE_OUT_OF_RANGE: "Date outside history",
  TEXT_TRUNCATED: "Long text cut",
  ROW_STRUCTURE_SUSPECT: "Possible file problem",
}

export const flagLabel = (code: string) => FLAG_LABELS[code] ?? code

export const flagCodes = (value: string | null | undefined) =>
  (value ?? "").split("|").filter(Boolean)

export const pct = (value: number | null | undefined, digits = 1) =>
  value == null || Number.isNaN(value) ? "-" : `${(value * 100).toFixed(digits)}%`

export const num = (value: number | null | undefined, digits = 0) =>
  value == null || Number.isNaN(value)
    ? "-"
    : value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })

export const signed = (value: number, digits = 2) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`

export const MODEL_NAMES: Record<string, string> = {
  champion: "Blend (logistic + LightGBM)",
  fallback_logistic: "Fallback (logistic)",
}

export const modelName = (name: string) => MODEL_NAMES[name] ?? name
