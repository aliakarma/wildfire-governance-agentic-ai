"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import type { LedgerEntry } from "@/lib/types";

function Term({ label, state, detail }: { label: string; state: boolean | null; detail?: string }) {
  const color = state === true ? "var(--ok)" : state === false ? "var(--danger)" : "var(--text-muted)";
  const glyph = state === true ? "✓" : state === false ? "✕" : "—";
  return (
    <div className="flex items-center gap-3 rounded-lg border bg-[var(--surface-2)]/40 px-3 py-2.5">
      <span
        className="grid h-6 w-6 flex-none place-items-center rounded-full text-sm font-bold text-white"
        style={{ background: color }}
        aria-hidden
      >
        {glyph}
      </span>
      <span className="text-sm">{label}</span>
      {detail && <span dir="ltr" className="tnum ms-auto text-xs text-muted">{detail}</span>}
    </div>
  );
}

export function PredicateInspector({ entry }: { entry: LedgerEntry | null }) {
  const { t } = useLang();
  const p = entry?.predicate;

  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-semibold">{t("gov.predicate")}</div>
      <p className="mb-3 text-xs text-muted">{t("gov.predicate.desc")}</p>
      <code dir="ltr" className="mb-3 block rounded-md bg-[var(--surface-2)] px-3 py-2 font-mono text-[11px] text-muted">
        G = [Conf &gt; τ] ∧ HA ∧ sig_valid ∧ consensus
      </code>

      {!p ? (
        <p className="rounded-lg border border-dashed px-3 py-6 text-center text-xs text-muted">
          {t("gov.no_event")}
        </p>
      ) : (
        <div className="space-y-2">
          <Term
            label={t("gov.term.conf")}
            state={p.conf_ok}
            detail={entry?.conf != null ? `conf ${entry.conf} / τ ${entry.tau ?? ""}` : undefined}
          />
          <Term label={t("gov.term.ha")} state={p.human_approval} />
          <Term label={t("gov.term.sig")} state={p.signature_ok} />
          <Term label={t("gov.term.consensus")} state={p.consensus_ok} />
          <div className="my-2 border-t" />
          <Term
            label={t("gov.term.result")}
            state={p.satisfied}
            detail={entry?.cert ? `cert ${entry.cert}…` : undefined}
          />
        </div>
      )}
    </div>
  );
}
