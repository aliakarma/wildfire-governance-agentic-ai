"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import { useSim } from "@/components/providers/SimulationProvider";
import type { SimParams } from "@/lib/types";

const PRESETS: Array<{ id: string; key: string; patch: Partial<SimParams> }> = [
  { id: "none", key: "adv.preset.none", patch: { attack_type: "none", p_spoof: 0, n_byzantine: 0 } },
  { id: "spoof", key: "adv.preset.spoof", patch: { attack_type: "spoofing", p_spoof: 0.2, n_byzantine: 0 } },
  { id: "injection", key: "adv.preset.injection", patch: { attack_type: "injection", p_spoof: 0, n_byzantine: 0 } },
  { id: "byzantine", key: "adv.preset.byzantine", patch: { attack_type: "byzantine", p_spoof: 0, n_byzantine: 3 } },
];

export function AttackConsole() {
  const { t } = useLang();
  const s = useSim();
  const running = s.status === "running" || s.status === "connecting";

  const activeId =
    s.params.attack_type === "none"
      ? "none"
      : s.params.attack_type === "injection"
        ? "injection"
        : s.params.attack_type === "byzantine"
          ? "byzantine"
          : "spoof";

  return (
    <div className="card flex flex-wrap items-center gap-3 p-3">
      <h1 className="text-sm font-semibold">{t("adv.console")}</h1>
      <div className="flex flex-wrap items-center gap-2">
        {PRESETS.map((p) => {
          const active = activeId === p.id;
          return (
            <button
              key={p.id}
              onClick={() => s.setParam({ ...p.patch, method: "greedy_gomdp" })}
              className={`rounded-md border px-3 py-1.5 text-xs font-medium transition ${
                active ? "border-[var(--accent)] bg-[var(--surface-2)] text-accent" : "text-muted hover:bg-[var(--surface-2)]/60"
              }`}
            >
              {t(p.key)}
            </button>
          );
        })}
      </div>
      <button
        onClick={running ? s.stop : s.run}
        className={`ms-auto rounded-lg px-4 py-2 text-sm font-semibold text-white ${running ? "bg-[var(--danger)]" : "bg-[var(--accent)]"}`}
      >
        {running ? t("cta.stop") : t("cta.run")}
      </button>
    </div>
  );
}
