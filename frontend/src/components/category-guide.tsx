import type { ReactNode } from "react"

import { ChartSkeleton, ClassLabel, LoadError } from "@/components/blocks"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { categoryFacts, categoryHeadline, useCategoryProfiles } from "@/lib/categories"
import { cn } from "@/lib/utils"

/** One card per category: a headline and the facts that support it. */
export function CategoryCards({ columns = "lg:grid-cols-3" }: { columns?: string }) {
  const { data, error } = useCategoryProfiles()
  if (error) return <LoadError error={error} />
  if (!data) return <ChartSkeleton height={220} />
  return (
    <div className={cn("grid gap-3 sm:grid-cols-2", columns)}>
      {data.rows.map((profile) => (
        <div key={profile.class} className="space-y-2 rounded-lg border bg-muted/20 p-3">
          <div className="space-y-0.5">
            <p className="text-sm font-medium">
              <ClassLabel label={profile.class} />
            </p>
            <p className="text-xs text-muted-foreground">{categoryHeadline(profile)}</p>
          </div>
          <ul className="space-y-1 text-xs leading-relaxed">
            {categoryFacts(profile).map((fact) => (
              <li key={fact} className="flex gap-2">
                <span aria-hidden className="mt-1.5 size-1 shrink-0 rounded-full bg-muted-foreground" />
                {fact}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}

/** Wraps a category name so hovering it shows what that category looks like. */
export function CategoryHint({ label, children }: { label: string; children: ReactNode }) {
  const { data } = useCategoryProfiles()
  const profile = data?.rows.find((p) => p.class === label)
  if (!profile) return <>{children}</>
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="cursor-help">{children}</span>
      </TooltipTrigger>
      <TooltipContent className="max-w-72 text-xs leading-relaxed">
        <div className="space-y-1">
          <p className="font-medium">
            {profile.class}: {categoryHeadline(profile)}
          </p>
          {categoryFacts(profile)
            .slice(0, 3)
            .map((fact) => (
              <p key={fact}>{fact}</p>
            ))}
        </div>
      </TooltipContent>
    </Tooltip>
  )
}
