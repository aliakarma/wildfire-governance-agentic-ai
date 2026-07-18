"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import type { LedgerEntry } from "@/lib/types";

const COLORS: Record<string, string> = {
  ALERT_APPROVED: "var(--ok)",
  ALERT_SIGNED: "var(--info)",
  ALERT_BLOCKED: "var(--danger)",
  ALERT_UNGOVERNED: "var(--warn)",
  HITL_REJECTED: "var(--warn)",
  INJECTION_BLOCKED: "var(--danger)",
};

interface Props {
  events: LedgerEntry[];
  selected: LedgerEntry | null;
  onSelect: (e: LedgerEntry) => void;
}

export function AuditLog({ events, selected, onSelect }: Props) {
  const { t } = useLang();
  const list = events.slice(-120).reverse();

  return (
    <div className="flex h-full min-h-[280px] flex-col rounded-card border bg-[var(--surface)]">
      <div className="border-b px-3 py-2 text-sm font-semibold">{t("gov.audit")}</div>
      <div className="flex-1 overflow-y-auto p-2">
        {list.length === 0 ? (
          <p className="p-3 text-xs text-muted">{t("gov.select_hint")}</p>
        ) : (
          <ul className="space-y-1">
            {list.map((e, i) => {
              const color = COLORS[e.kind] ?? "var(--text-muted)";
              const active = selected?.t === e.t && selected?.kind === e.kind;
              const inspectable = Boolean(e.predicate);
              return (
                <li key={`${e.t}-${i}`}>
                  <button
                    onClick={() => inspectable && onSelect(e)}
                    disabled={!inspectable}
                    className={`flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-start text-xs transition ${
                      active ? "border-[var(--accent)] bg-[var(--surface-2)]" : "bg-[var(--surface-2)]/30 hover:bg-[var(--surface-2)]/60"
                    } ${inspectable ? "" : "opacity-70"}`}
                  >
                    <span className="h-2 w-2 flex-none rounded-full" style={{ background: color }} aria-hidden />
                    <span className="tnum text-muted">t={e.t}</span>
                    <span className="font-medium" style={{ color }}>
                      {t(`event.${e.kind}`)}
                    </span>
                    {e.cert ? (
                      <code dir="ltr" className="ms-auto font-mono text-[10px] text-muted">{e.cert}…</code>
                    ) : e.conf != null ? (
                      <span className="tnum ms-auto text-muted">conf {e.conf}</span>
                    ) : null}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
