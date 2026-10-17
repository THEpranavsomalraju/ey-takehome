import { useEffect, useState } from "react"

import { Panel, Section } from "@/components/blocks"
import { cn } from "@/lib/utils"
import { ClassDistributionPanel, DuplicateStats, MonthlyVolumePanel, SignalPanel } from "./data-panels"
import { ConfusionPanel, HeadlineStats, PerClassPanel } from "./holdout-panels"
import { AblationPanel, ClassWeightPanel, LeaderboardPanel, ValidationPanel } from "./model-panels"
import { FeatureCatalogPanel, PipelineFlow, TuningStats } from "./pipeline-panels"
import { CalibrationPanel, FlagReliabilityPanel, ThresholdPanel } from "./trust-panels"

const SECTIONS = [
  { id: "results", label: "Results" },
  { id: "data", label: "The data" },
  { id: "validation", label: "Honest testing" },
  { id: "models", label: "Model choice" },
  { id: "trust", label: "Confidence & review" },
  { id: "holdout", label: "Holdout detail" },
  { id: "pipeline", label: "Pipeline" },
]

/** Highlight the section currently in view in the side navigation. */
function useActiveSection() {
  const [active, setActive] = useState(SECTIONS[0].id)
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible[0]) setActive(visible[0].target.id)
      },
      { rootMargin: "-120px 0px -60% 0px" },
    )
    SECTIONS.forEach((s) => {
      const element = document.getElementById(s.id)
      if (element) observer.observe(element)
    })
    return () => observer.disconnect()
  }, [])
  return active
}

export function OverviewTab() {
  const active = useActiveSection()
  return (
    <div className="grid gap-10 xl:grid-cols-[11rem_1fr]">
      <nav aria-label="Sections" className="hidden xl:block">
        <ul className="sticky top-28 space-y-0.5 border-l text-sm">
          {SECTIONS.map((s) => (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                className={cn(
                  "-ml-px block border-l py-1 pl-3 text-muted-foreground transition-colors hover:text-foreground",
                  active === s.id && "border-primary font-medium text-foreground",
                )}
              >
                {s.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      <div className="min-w-0 space-y-14">
        <Section
          id="results"
          eyebrow="Results"
          title="An invoice classifier that knows when to ask for help"
          description="The model suggests one of six categories for every accounts payable line, says how sure it is, and sends uncertain or unfamiliar lines to a person. The holdout figures come from 590 rows that were never used during development; the expected accuracy is the cross-validation estimate."
        >
          <HeadlineStats />
        </Section>

        <Section
          id="data"
          eyebrow="1 - The data"
          title="5,899 anonymized invoice lines, one dominant category"
          description="Seven inputs (vendor, invoice number, amount, description, posting date, GL account, document flag) and a label. 10% was set aside as the holdout before any analysis; everything below uses the 5,309 training rows."
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <ClassDistributionPanel />
            <MonthlyVolumePanel />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <SignalPanel />
            <Panel
              title="Repeated rows"
              description="Rows identical on every model input except the invoice number."
              takeaway="A quarter of the rows repeat an earlier row exactly, and copies almost always share a label. The model can learn these patterns, but testing must keep copies apart (next section)."
            >
              <DuplicateStats />
            </Panel>
          </div>
        </Section>

        <Section
          id="validation"
          eyebrow="2 - Honest testing"
          title="Grouped cross-validation, out-of-time, and a single holdout"
          description="Copies of a row always stay on the same side of a split, and every preprocessing step is fitted inside each training fold. Otherwise the model would be tested on rows it had effectively already seen."
        >
          <ValidationPanel />
        </Section>

        <Section
          id="models"
          eyebrow="3 - Model choice"
          title="Eleven candidates, one rule fixed before tuning"
          description="Baselines, linear models, naive Bayes, gradient boosting, and two blends were compared on identical folds. The final model was picked by a rule written down in advance: the most accurate model that also handles rare categories and later months well."
        >
          <LeaderboardPanel />
          <div className="grid gap-4 lg:grid-cols-2">
            <AblationPanel />
            <ClassWeightPanel />
          </div>
          <TuningStats />
        </Section>

        <Section
          id="trust"
          eyebrow="4 - Confidence and review"
          title="Calibrated confidence decides what a person checks"
          description="A confidence score is only useful if it is honest. After calibration, the review threshold and the warning flags route the risky rows to a reviewer."
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <CalibrationPanel />
            <ThresholdPanel />
          </div>
          <FlagReliabilityPanel />
        </Section>

        <Section
          id="holdout"
          eyebrow="5 - Holdout detail"
          title="The final check on unseen data"
          description="The holdout was scored exactly once, with the final model as trained above. Switch to cross-validation to compare with the development estimate."
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <ConfusionPanel />
            <PerClassPanel />
          </div>
        </Section>

        <Section
          id="pipeline"
          eyebrow="6 - Pipeline"
          title="From spreadsheet cell to explained prediction"
          description="The same code path serves training, the holdout, and every file uploaded in the Classify tab."
        >
          <PipelineFlow />
          <FeatureCatalogPanel />
        </Section>
      </div>
    </div>
  )
}
