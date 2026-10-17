import type { ReactNode } from "react"

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { GLOSSARY, type GlossaryKey } from "@/lib/glossary"

// Terms read mid-sentence in lower case, except names and acronyms (Cramer's V, TF-IDF).
const inSentence = (term: string) => (/^[A-Z][a-z]/.test(term) && !term.startsWith("Cramer") ? term.charAt(0).toLowerCase() + term.slice(1) : term)

/** A technical term with a dotted underline; hovering or focusing shows its plain-English definition. */
export function Term({ name, children }: { name: GlossaryKey; children?: ReactNode }) {
  const entry = GLOSSARY[name]
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className="cursor-help underline decoration-muted-foreground/60 decoration-dotted underline-offset-4 outline-none focus-visible:decoration-foreground"
        >
          {children ?? inSentence(entry.term)}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-72 text-xs leading-relaxed">
        <p>
          <span className="font-medium">{entry.term}: </span>
          {entry.definition}
        </p>
      </TooltipContent>
    </Tooltip>
  )
}
