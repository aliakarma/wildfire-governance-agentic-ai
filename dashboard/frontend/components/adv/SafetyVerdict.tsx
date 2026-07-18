"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import { useSim } from "@/components/providers/SimulationProvider";

export function SafetyVerdict() {
  const { t } = useLang();
  const s = useSim();
  const sum = s.summary;
  const held = sum ? Number(sum.compliance) >= 99.999 : true;
  const attempted = sum ? Number(sum.n_injections_attempted ?? 0) : 0;
  const blocked = sum ? Number(sum.n_injections_blocked ?? 0) : 0;

  return (
    <div className="card p-4">
      <div className="mb-3 text-sm font-semibold">{t("adv.verdict")}</div>
      <div
        className="mb-3 rounded-lg px-3 py-3 text-center text-sm font-semibold"
        style={{
          background: held ? "color-mix(in srgb, var(--ok) 14%, transparent)" : "color-mix(in srgb, var(--danger) 14%, transparent)",
          color: held ? "var(--ok)" : "var(--danger)",
        }}
      >
        {held ? `✓ ${t("adv.held")}` : `✕ ${t("adv.breached")}`}
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="rounded-lg border bg-[var(--surface-2)]/40 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted">{t("adv.injections_attempted")}</div>
          <div className="tnum text-lg font-semibold">{attempted}</div>
        </div>
        <div className="rounded-lg border bg-[var(--surface-2)]/40 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted">{t("adv.injections_blocked")}</div>
          <div className="tnum text-lg font-semibold" style={{ color: "var(--ok)" }}>
            {blocked}
          </div>
        </div>
        <div className="rounded-lg border bg-[var(--surface-2)]/40 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted">{t("metric.fp")}</div>
          <div className="tnum text-lg font-semibold" style={{ color: "var(--warn)" }}>
            {sum ? `${sum.fp_pct}%` : "—"}
          </div>
        </div>
        <div className="rounded-lg border bg-[var(--surface-2)]/40 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted">{t("metric.compliance")}</div>
          <div className="tnum text-lg font-semibold" style={{ color: held ? "var(--ok)" : "var(--danger)" }}>
            {sum ? `${sum.compliance}%` : "—"}
          </div>
        </div>
      </div>
    </div>
  );
}
