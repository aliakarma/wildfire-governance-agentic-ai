"use client";
import { useEffect, useState } from "react";
import { useLang } from "@/components/providers/LanguageProvider";
import { useSim, type View } from "@/components/providers/SimulationProvider";

const STEPS: Array<{ title: string; body: string; view: View | null }> = [
  { title: "tour.welcome.title", body: "tour.welcome.body", view: null },
  { title: "tour.live.title", body: "tour.live.body", view: "live" },
  { title: "tour.gov.title", body: "tour.gov.body", view: "governance" },
  { title: "tour.adv.title", body: "tour.adv.body", view: "adversarial" },
  { title: "tour.compare.title", body: "tour.compare.body", view: "compare" },
  { title: "tour.viirs.title", body: "tour.viirs.body", view: "viirs" },
  { title: "tour.bench.title", body: "tour.bench.body", view: "benchmark" },
];

export function OnboardingTour({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useLang();
  const { setView } = useSim();
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (open) setStep(0);
  }, [open]);

  useEffect(() => {
    if (open && STEPS[step]?.view) setView(STEPS[step].view as View);
  }, [open, step, setView]);

  if (!open) return null;
  const s = STEPS[step];
  const last = step === STEPS.length - 1;

  const finish = () => {
    try {
      localStorage.setItem("tour_seen", "1");
    } catch {
      /* ignore */
    }
    setView("live");
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/40 p-4 pb-[8vh] sm:items-center sm:pb-4" role="dialog" aria-modal="true">
      <div className="w-full max-w-md rounded-card border bg-[var(--surface)] p-5 shadow-2xl">
        <div className="mb-1 flex items-center gap-2">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-[var(--accent)] text-white">🔥</span>
          <div className="text-[11px] text-muted">
            {step + 1} {t("tour.of")} {STEPS.length}
          </div>
        </div>
        <h2 className="mt-2 text-base font-semibold">{t(s.title)}</h2>
        <p className="mt-2 text-sm text-muted">{t(s.body)}</p>

        {/* progress dots */}
        <div className="mt-4 flex gap-1.5">
          {STEPS.map((_, i) => (
            <span
              key={i}
              className="h-1.5 flex-1 rounded-full"
              style={{ background: i <= step ? "var(--accent)" : "var(--surface-2)" }}
            />
          ))}
        </div>

        <div className="mt-5 flex items-center justify-between">
          <button onClick={finish} className="text-xs text-muted hover:text-text">
            {t("tour.skip")}
          </button>
          <div className="flex items-center gap-2">
            {step > 0 && (
              <button
                onClick={() => setStep((n) => n - 1)}
                className="rounded-md border px-3 py-1.5 text-sm hover:bg-[var(--surface-2)]"
              >
                {t("tour.back")}
              </button>
            )}
            <button
              onClick={() => (last ? finish() : setStep((n) => n + 1))}
              className="rounded-md bg-[var(--accent)] px-4 py-1.5 text-sm font-semibold text-white"
            >
              {last ? t("tour.done") : t("tour.next")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
