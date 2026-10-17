import { ArrowRightIcon } from "lucide-react"

import { Card, CardContent } from "@/components/ui/card"

const PARTS = [
  {
    title: "The data",
    text: "5,899 lines from supplier invoices a company paid (an accounts payable ledger). Each line has a vendor, invoice number, amount, description, posting date, GL account, and document flag. Words are anonymized as codes such as Word563, but the same code is always the same word.",
  },
  {
    title: "What the model predicts",
    text: "The category a person assigned to each line (Category_1 to Category_6). Given a new line, the model suggests its category, says how sure it is, flags lines a person should check, and explains why.",
  },
  {
    title: "How to read this site",
    text: "This tab shows how the model was built and tested, top to bottom. The Classify & Explain tab lets you try it on a file. New to the terms? Hover any dotted word, or open the Guide at the top right.",
  },
]

/** Plain-language introduction for first-time visitors. */
export function IntroCard() {
  return (
    <Card className="py-0">
      <CardContent className="grid gap-0 p-0 md:grid-cols-3">
        {PARTS.map((part, i) => (
          <div key={part.title} className="relative space-y-1.5 border-b p-4 last:border-b-0 md:border-r md:border-b-0 md:last:border-r-0">
            <p className="text-sm font-medium">{part.title}</p>
            <p className="text-xs leading-relaxed text-muted-foreground">{part.text}</p>
            {i < PARTS.length - 1 && (
              <ArrowRightIcon className="absolute top-1/2 -right-2 z-10 hidden size-4 -translate-y-1/2 rounded-full bg-card text-muted-foreground md:block" />
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
