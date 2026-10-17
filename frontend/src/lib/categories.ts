import { num, pct } from "@/lib/domain"
import { useTable } from "@/lib/reports"

export interface CategoryProfile {
  class: string
  rows: number
  share: number
  vendors: number
  top_vendor: string
  top_vendor_share: number
  accounts: number
  top_account: string
  top_account_share: number
  top_doc_flag: string
  top_doc_flag_share: number
  amount_median: number | null
  credit_share: number | null
  first_month: string | null
  last_month: string | null
  peak_month: string | null
}

export function useCategoryProfiles() {
  return useTable<CategoryProfile>("eda/category_profiles.csv")
}

/** One-line headline for a category, derived from its profile. */
export function categoryHeadline(p: CategoryProfile): string {
  if (p.share >= 0.5) return "The default: most invoice lines"
  if ((p.credit_share ?? 0) >= 0.5) return "Credit notes (negative amounts)"
  if (p.share >= 0.05) return "The second most common category"
  if (p.top_doc_flag !== "NoDoc" && p.top_doc_flag_share >= 0.5) return `Rare, mostly document flag ${p.top_doc_flag}`
  if (p.top_vendor_share >= 0.5) return "Rare, mostly one vendor"
  return "Rare, spread over a few vendors"
}

/** Plain-language facts about a category, each backed by the profile numbers. */
export function categoryFacts(p: CategoryProfile): string[] {
  const facts = [`${pct(p.share, p.share < 0.01 ? 2 : 1)} of invoice lines (${num(p.rows)} training rows).`]
  facts.push(
    p.top_vendor_share >= 0.5
      ? `Mostly one vendor: ${p.top_vendor} (${pct(p.top_vendor_share, 0)} of its rows).`
      : `Spread over ${num(p.vendors)} vendors; the largest has ${pct(p.top_vendor_share, 0)} of its rows.`,
  )
  facts.push(
    p.top_account_share >= 0.5
      ? `Mostly GL account ${p.top_account} (${pct(p.top_account_share, 0)}).`
      : `${num(p.accounts)} GL accounts; the largest has ${pct(p.top_account_share, 0)} of its rows.`,
  )
  if (p.top_doc_flag !== "NoDoc") facts.push(`Usually document flag ${p.top_doc_flag} (${pct(p.top_doc_flag_share, 0)}).`)
  if ((p.credit_share ?? 0) >= 0.5) {
    facts.push(`Credits: every amount is negative (median ${num(p.amount_median, 2)}).`)
  } else if (p.amount_median != null) {
    facts.push(`Median amount ${num(p.amount_median, 2)}.`)
  }
  if (p.first_month && p.first_month === p.last_month) {
    facts.push(`Only in ${p.first_month}.`)
  } else if (p.first_month) {
    facts.push(`From ${p.first_month} to ${p.last_month}, most in ${p.peak_month}.`)
  }
  return facts
}

export const CATEGORY_CAVEAT =
  "The ledger is anonymized: the categories are only called Category_1 to Category_6, and their business meaning (for example a tax treatment or cost type) is not given. The descriptions below are what each category looks like in the data."
