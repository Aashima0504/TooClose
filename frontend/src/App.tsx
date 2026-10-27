import { AdvicePanel } from './components/AdvicePanel'
import { DebugPanel } from './components/DebugPanel'
import { PrivacyBadge } from './components/PrivacyBadge'
import { RiskGauge } from './components/RiskGauge'
import { SettingsPanel } from './components/SettingsPanel'
import { ShapChart } from './components/ShapChart'
import { Sparkline } from './components/Sparkline'
import { VideoOverlay } from './components/VideoOverlay'

export default function App() {
  return (
    <main className="mx-auto flex min-h-svh max-w-6xl flex-col gap-4 p-4 md:p-8">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">TooClose</h1>
          <p className="text-sm text-zinc-500">
            Phase 1 scaffold — webcam and risk model arrive in later phases.
          </p>
        </div>
        <PrivacyBadge />
      </header>
      <div className="grid gap-4 lg:grid-cols-3">
        <section className="lg:col-span-2">
          <VideoOverlay />
        </section>
        <aside className="flex flex-col gap-4">
          <RiskGauge />
          <AdvicePanel />
          <ShapChart />
          <Sparkline />
          <SettingsPanel />
          <DebugPanel />
        </aside>
      </div>
    </main>
  )
}
