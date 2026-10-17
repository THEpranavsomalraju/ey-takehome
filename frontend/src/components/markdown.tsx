import type { ReactNode } from "react"

/**
 * Renders the small Markdown subset the explanation agent writes: paragraphs,
 * `**bold**`, and (nested) bullet lists. Everything is plain text otherwise,
 * so nothing from the model is ever injected as HTML.
 */
export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = []
  let bullets: { depth: number; text: string }[] = []

  const flush = () => {
    if (bullets.length) {
      blocks.push(
        <ul key={`list-${blocks.length}`} className="space-y-1">
          {bullets.map((b, i) => (
            <li
              key={i}
              className="flex gap-2 leading-relaxed"
              style={{ paddingLeft: `${b.depth * 1}rem` }}
            >
              <span aria-hidden className="mt-2 size-1 shrink-0 rounded-full bg-muted-foreground" />
              <span>{inline(b.text)}</span>
            </li>
          ))}
        </ul>,
      )
      bullets = []
    }
  }

  for (const raw of text.split("\n")) {
    const bullet = raw.match(/^(\s*)[-*]\s+(.*)$/)
    if (bullet) {
      bullets.push({ depth: Math.floor(bullet[1].length / 2), text: bullet[2] })
      continue
    }
    flush()
    if (raw.trim()) {
      blocks.push(
        <p key={`p-${blocks.length}`} className="leading-relaxed">
          {inline(raw.trim())}
        </p>,
      )
    }
  }
  flush()
  return <div className="space-y-3 text-sm">{blocks}</div>
}

function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i} className="font-semibold text-foreground">
        {part.slice(2, -2)}
      </strong>
    ) : (
      part
    ),
  )
}
