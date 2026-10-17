import { useState } from "react"

import { AgentBadge, ModelSheet, SettingsPopover } from "@/components/app-header"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ClassifyTab } from "@/features/classify/classify-tab"
import { OverviewTab } from "@/features/overview/overview-tab"

type Tab = "overview" | "classify"

export default function App() {
  const [tab, setTab] = useState<Tab>("overview")
  return (
    <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)} className="min-h-dvh gap-0">
      <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-4 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-2.5">
            <span aria-hidden className="size-4 shrink-0 rounded-[3px] bg-primary" />
            <div className="min-w-0 leading-tight">
              <p className="truncate text-sm font-semibold">Ledger Classifier</p>
              <p className="hidden truncate text-[11px] text-muted-foreground sm:block">
                Accounts payable categorization
              </p>
            </div>
          </div>
          <TabsList className="mx-auto">
            <TabsTrigger value="overview" className="px-3 text-xs sm:text-sm">Data &amp; Model</TabsTrigger>
            <TabsTrigger value="classify" className="px-3 text-xs sm:text-sm">Classify &amp; Explain</TabsTrigger>
          </TabsList>
          <div className="flex items-center gap-2">
            <AgentBadge />
            <ModelSheet />
            <SettingsPopover />
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1440px] flex-1 px-4 py-8 sm:px-6">
        <TabsContent value="overview" forceMount hidden={tab !== "overview"}>
          <OverviewTab />
        </TabsContent>
        <TabsContent value="classify" forceMount hidden={tab !== "classify"}>
          <ClassifyTab />
        </TabsContent>
      </main>
      <footer className="border-t">
        <div className="mx-auto max-w-[1440px] px-4 py-4 text-xs text-muted-foreground sm:px-6">
          EY AI Technical Assessment. Figures come from the report files in reports/ and the loaded model.
        </div>
      </footer>
    </Tabs>
  )
}
