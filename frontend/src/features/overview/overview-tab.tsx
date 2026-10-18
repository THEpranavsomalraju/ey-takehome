import { useEffect, useState } from "react"

import { Panel, Section } from "@/components/blocks"
import { CategoryCards } from "@/components/category-guide"
import { Term } from "@/components/term"
import { CATEGORY_CAVEAT } from "@/lib/categories"
import { cn } from "@/lib/utils"
import { AmountPanel, BeforeAfterPanel, LabelCleanupPanel, MissingValuesPanel } from "./cleaning-panels"
import { ClassDistributionPanel, DuplicateStats, MonthlyVolumePanel, SignalPanel } from "./data-panels"
import { ConfusionPanel, HeadlineStats, PerClassPanel } from "./holdout-panels"
import { IntroCard } from "./intro-card"
import { AblationPanel, ClassWeightPanel, LeaderboardPanel, ValidationPanel } from "./model-panels"
import { FeatureCatalogPanel, PipelineFlow, TuningStats } from "./pipeline-panels"
import { CalibrationPanel, FlagReliabilityPanel, ThresholdPanel } from "./trust-panels"

const SECTIONS = [
  { id: "results", label: "Results" },
  { id: "data", label: "Exploration" },
  { id: "cleaning", label: "Cleaning & features" },
  { id: "split", label: "Split & testing" },
  { id: "models", label: "Model choice" },
  { id: "review", label: "Confidence & review" },
  { id: "holdout", label: "Holdout results" },
]
const HEADER_OFFSET = 140 // sticky header plus a little margin, in pixels

/** The section whose top has scrolled past the sticky header. */
function useActiveSection() {
  const [active, setActive] = useState(SECTIONS[0].id)
  useEffect(() => {
    const update = () => {
      let current = SECTIONS[0].id
      for (const s of SECTIONS) {
        const element = document.getElementById(s.id)
        if (element && element.getBoundingClientRect().top <= HEADER_OFFSET) current = s.id
      }
      setActive(current)
    }
    update()
    window.addEventListener("scroll", update, { passive: true })
    return () => window.removeEventListener("scroll", update)
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
          title="Accounts payable line classifier"
          description={
            <>
              The model suggests one of six categories for each invoice line and sends uncertain or unfamiliar lines to a person. The <Term name="holdout" /> figures come from 590 lines set aside before any analysis; the expected accuracy is the <Term name="grouped cross-validation" /> estimate.
            </>
          }
        >
          <IntroCard />
          <HeadlineStats />
        </Section>

        <Section
          id="data"
          eyebrow="1. Exploration"
          title="5,899 anonymized invoice lines, one dominant category"
          description="Seven inputs and a label. Everything in this section uses the 5,309 training lines; the holdout stays untouched."
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <ClassDistributionPanel />
            <MonthlyVolumePanel />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <MissingValuesPanel />
            <AmountPanel />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <SignalPanel />
            <Panel
              title="Repeated rows"
              description="Lines that match an earlier line on every input except the invoice number."
              why="Matching lines split across training and test would make the test too easy, so every test keeps them together."
              explain={{ chart: "repeated-rows" }}
              takeaway="A quarter of the lines match an earlier line apart from the invoice number (12% are exact repeats), and matching lines almost always share a label."
            >
              <DuplicateStats />
            </Panel>
          </div>
          <Panel
            title="What each category looks like"
            description={CATEGORY_CAVEAT}
            why="Hover a category in the Classify tab to see its profile there."
            explain={{ chart: "category-profiles" }}
          >
            <CategoryCards />
          </Panel>
        </Section>

        <Section
          id="cleaning"
          eyebrow="2. Cleaning and features"
          title="From a spreadsheet cell to model input"
          description="The same code runs for training, the holdout, and every uploaded file, so nothing is prepared by hand."
        >
          <PipelineFlow />
          <div className="grid gap-4 lg:grid-cols-[2fr_3fr]">
            <LabelCleanupPanel />
            <BeforeAfterPanel />
          </div>
          <FeatureCatalogPanel />
        </Section>

        <Section
          id="split"
          eyebrow="3. Split and testing"
          title="A 10% holdout, then grouped cross-validation"
          description="590 lines (10%) were set aside first and scored once at the end. All development used the other 5,309 lines with grouped folds: lines that match on every input except the invoice number stay on the same side of a split, and every preprocessing step is fitted inside each training fold."
        >
          <ValidationPanel />
        </Section>

        <Section
          id="models"
          eyebrow="4. Model choice"
          title="Eleven models compared, one picked by a rule set in advance"
          description="Baselines, linear models, naive Bayes, gradient boosting, and two blends ran on identical folds. The rule, written before tuning: the most accurate model that also handles rare categories and later months."
        >
          <LeaderboardPanel />
          <div className="grid gap-4 lg:grid-cols-2">
            <AblationPanel />
            <ClassWeightPanel />
          </div>
          <TuningStats />
        </Section>

        <Section
          id="review"
          eyebrow="5. Confidence and review"
          title="Calibrated confidence decides what a person checks"
          description="After calibration, a confidence threshold and a set of warnings decide which lines go to a reviewer."
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <CalibrationPanel />
            <ThresholdPanel />
          </div>
          <FlagReliabilityPanel />
        </Section>

        <Section
          id="holdout"
          eyebrow="6. Holdout results"
          title="The final check on unseen data"
          description="Scored once, with the final model as trained above. Switch to cross-validation to compare with the development estimate."
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <ConfusionPanel />
            <PerClassPanel />
          </div>
        </Section>
      </div>
    </div>
  )
}
