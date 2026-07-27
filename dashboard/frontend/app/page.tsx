"use client";
import { useEffect, useState } from "react";
import { CommandPalette } from "@/components/layout/CommandPalette";
import { Navbar } from "@/components/layout/Navbar";
import { OnboardingTour } from "@/components/layout/OnboardingTour";
import { useSim } from "@/components/providers/SimulationProvider";
import { AdversarialScreen } from "@/components/screens/AdversarialScreen";
import { BenchmarkScreen } from "@/components/screens/BenchmarkScreen";
import { AblationScreen } from "@/components/screens/AblationScreen";
import { CompareScreen } from "@/components/screens/CompareScreen";
import { ExperimentsScreen } from "@/components/screens/ExperimentsScreen";
import { GovernanceScreen } from "@/components/screens/GovernanceScreen";
import { HITLScreen } from "@/components/screens/HITLScreen";
import { LearningScreen } from "@/components/screens/LearningScreen";
import { StatisticsScreen } from "@/components/screens/StatisticsScreen";
import { ScalabilityScreen } from "@/components/screens/ScalabilityScreen";
import { LiveScreen } from "@/components/screens/LiveScreen";
import { VIIRSScreen } from "@/components/screens/VIIRSScreen";

export default function Page() {
  const { view } = useSim();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    try {
      if (!localStorage.getItem("tour_seen")) setTourOpen(true);
    } catch {
      /* ignore */
    }
  }, []);

  return (
    <main className="min-h-screen">
      <Navbar onOpenPalette={() => setPaletteOpen(true)} onOpenHelp={() => setTourOpen(true)} />
      <div className="mx-auto max-w-[1600px] p-4">
        {view === "live" && <LiveScreen />}
        {view === "adversarial" && <AdversarialScreen />}
        {view === "governance" && <GovernanceScreen />}
        {view === "compare" && <CompareScreen />}
        {view === "viirs" && <VIIRSScreen />}
        {view === "benchmark" && <BenchmarkScreen />}
        {view === "experiments" && <ExperimentsScreen />}
        {view === "ablation" && <AblationScreen />}
        {view === "scalability" && <ScalabilityScreen />}
        {view === "learning" && <LearningScreen />}
        {view === "hitl" && <HITLScreen />}
        {view === "statistics" && <StatisticsScreen />}
      </div>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} onOpenTour={() => setTourOpen(true)} />
      <OnboardingTour open={tourOpen} onClose={() => setTourOpen(false)} />
    </main>
  );
}
