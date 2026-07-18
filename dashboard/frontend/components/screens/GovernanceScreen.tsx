"use client";
import { useEffect } from "react";
import { useLang } from "@/components/providers/LanguageProvider";
import { useSim } from "@/components/providers/SimulationProvider";
import { AuditLog } from "@/components/gov/AuditLog";
import { PredicateInspector } from "@/components/gov/PredicateInspector";
import { ValidatorRing } from "@/components/gov/ValidatorRing";

export function GovernanceScreen() {
  const { t } = useLang();
  const s = useSim();
  const running = s.status === "running" || s.status === "connecting";

  const inspectable = s.events.filter((e) => e.predicate);
  // Auto-select the latest inspectable decision when nothing is selected.
  useEffect(() => {
    if (!s.selected && inspectable.length > 0) {
      s.setSelected(inspectable[inspectable.length - 1]);
    }
  }, [inspectable.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const nVal = (s.meta?.n_validators as number) ?? 7;
  const nByz = (s.meta?.n_byzantine as number) ?? s.params.n_byzantine;
  const thr = (s.meta?.byzantine_threshold as number) ?? Math.floor((nVal - 1) / 3);

  return (
    <div className="flex flex-col gap-4">
      <div className="card flex flex-wrap items-center gap-3 p-3">
        <h1 className="text-sm font-semibold">{t("gov.title")}</h1>
        <div className="ms-auto flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-muted">
            {t("param.n_byzantine")}
            <span className="inline-flex items-center rounded-md border">
              <button
                className="px-2 py-1 hover:bg-[var(--surface-2)]"
                onClick={() => s.setParam({ n_byzantine: Math.max(0, s.params.n_byzantine - 1), attack_type: "byzantine" })}
              >
                −
              </button>
              <span className="tnum w-6 text-center text-text">{s.params.n_byzantine}</span>
              <button
                className="px-2 py-1 hover:bg-[var(--surface-2)]"
                onClick={() => s.setParam({ n_byzantine: Math.min(3, s.params.n_byzantine + 1), attack_type: "byzantine" })}
              >
                +
              </button>
            </span>
          </label>
          <button
            onClick={running ? s.stop : s.run}
            className={`rounded-lg px-4 py-2 text-sm font-semibold text-white ${running ? "bg-[var(--danger)]" : "bg-[var(--accent)]"}`}
          >
            {running ? t("cta.stop") : t("cta.run")}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="flex flex-col gap-4">
          <PredicateInspector entry={s.selected} />
          <ValidatorRing nValidators={nVal} nByzantine={nByz} threshold={thr} />
        </div>
        <AuditLog events={s.events} selected={s.selected} onSelect={s.setSelected} />
      </div>
    </div>
  );
}
