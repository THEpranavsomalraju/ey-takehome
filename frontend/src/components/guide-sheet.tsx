import { BookOpenIcon } from "lucide-react"

import { CategoryCards } from "@/components/category-guide"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { CATEGORY_CAVEAT } from "@/lib/categories"
import { GLOSSARY } from "@/lib/glossary"

const STEPS = [
  {
    title: "Data & Model tab",
    text: "How the model was built and how far to trust it: the data, cleaning, testing, the model comparison, confidence and review, and the final check on 590 unseen lines.",
  },
  {
    title: "Classify & Explain tab",
    text: "Try it: upload an Excel or CSV ledger, or click \"Use the holdout sample\". Each line gets a category, a confidence, and a review decision.",
  },
  {
    title: "Explain a row",
    text: "Click Explain on a row to see why it got its category and which past invoices support it. The AI writes a short explanation; \"What if?\" re-scores the line with one input changed.",
  },
]

/** First-time visitor guide: what this is, how to use it, the categories, and the terms. */
export function GuideSheet() {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button size="sm" variant="ghost" className="gap-2" aria-label="Guide">
          <BookOpenIcon />
          <span className="hidden lg:inline">Guide</span>
        </Button>
      </SheetTrigger>
      <SheetContent className="w-full gap-0 p-0 data-[side=right]:w-full data-[side=right]:sm:max-w-2xl">
        <SheetHeader className="border-b px-6 py-4">
          <SheetTitle>Guide</SheetTitle>
          <SheetDescription>What this tool does and how to use it.</SheetDescription>
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="space-y-6 px-6 py-5 text-sm">
            <section className="space-y-2">
              <h3 className="font-medium">What this is</h3>
              <p className="leading-relaxed text-muted-foreground">
                Finance teams code every accounts payable invoice line to a category. This tool learns
                that coding from 5,899 past lines and suggests the category for new ones. It says how
                sure it is, sends uncertain or unfamiliar lines to a person, and explains every
                suggestion. On 590 lines it had never seen, it chose the right category 97% of the time.
              </p>
            </section>
            <section className="space-y-3">
              <h3 className="font-medium">How to use it</h3>
              <ol className="space-y-3">
                {STEPS.map((step, i) => (
                  <li key={step.title} className="flex gap-3">
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-md border text-xs tabular text-muted-foreground">
                      {i + 1}
                    </span>
                    <div>
                      <p className="font-medium">{step.title}</p>
                      <p className="text-muted-foreground leading-relaxed">{step.text}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>
            <Separator />
            <section className="space-y-3">
              <h3 className="font-medium">The six categories</h3>
              <p className="text-xs leading-relaxed text-muted-foreground">{CATEGORY_CAVEAT}</p>
              <CategoryCards columns="lg:grid-cols-2" />
            </section>
            <Separator />
            <section className="space-y-3">
              <h3 className="font-medium">Terms used on this site</h3>
              <dl className="space-y-2.5">
                {Object.values(GLOSSARY).map((entry) => (
                  <div key={entry.term}>
                    <dt className="font-medium">{entry.term}</dt>
                    <dd className="text-muted-foreground leading-relaxed">{entry.definition}</dd>
                  </div>
                ))}
              </dl>
            </section>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
