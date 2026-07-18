"use client";
import { useEffect, useState } from "react";
import { CommandPalette } from "@/components/layout/CommandPalette";
import { Navbar } from "@/components/layout/Navbar";
import { OnboardingTour } from "@/components/layout/OnboardingTour";
import { useSim } from "@/components/providers/SimulationProvider";
import { AdversarialScreen } from "@/components/screens/AdversarialScreen";
import { BenchmarkScreen } from "@/components/screens/BenchmarkScreen";
import { CompareScreen } from "@/components/screens/CompareScreen";
import { GovernanceScreen } from "@/components/screens/GovernanceScreen";
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
      </div>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} onOpenTour={() => setTourOpen(true)} />
      <OnboardingTour open={tourOpen} onClose={() => setTourOpen(false)} />
    </main>
  );
}
