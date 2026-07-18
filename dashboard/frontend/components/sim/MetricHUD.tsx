"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import type { Frame } from "@/lib/types";

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-lg border bg-[var(--surface)] px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className="tnum text-lg font-semibold" style={{ color: accent }}>
        {value}
      </div>
    </div>
  );
}

export function MetricHUD({ frame }: { frame: Frame | null }) {
  const { t } = useLang();
  const m = frame?.metrics;
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
      <Stat label={t("metric.step")} value={frame ? `${frame.t}` : "—"} />
      <Stat label={t("metric.ld")} value={m?.ld != null ? `${m.ld}` : "—"} />
      <Stat
        label={t("metric.fp")}
        value={m ? `${m.fp_pct}%` : "—"}
        accent="var(--warn)"
      />
      <Stat
        label={t("metric.compliance")}
        value={m ? `${m.compliance}%` : "—"}
        accent={m && m.compliance >= 99.999 ? "var(--ok)" : "var(--danger)"}
      />
      <Stat label={t("metric.injections")} value={m ? `${m.n_injections_blocked}` : "—"} accent="var(--info)" />
    </div>
  );
}
