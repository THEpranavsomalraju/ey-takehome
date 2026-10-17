import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import App from "@/App"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { SettingsProvider } from "@/lib/settings"
import "./index.css"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SettingsProvider>
      <TooltipProvider delayDuration={150}>
        <App />
        <Toaster position="bottom-right" />
      </TooltipProvider>
    </SettingsProvider>
  </StrictMode>,
)
