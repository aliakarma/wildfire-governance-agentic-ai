"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import type { Frame } from "@/lib/types";

const COLORS: Record<string, string> = {
  ALERT_APPROVED: "var(--ok)",
  ALERT_SIGNED: "var(--info)",
  ALERT_BLOCKED: "var(--danger)",
  ALERT_UNGOVERNED: "var(--warn)",
  ALERT_UNAUTHORISED: "var(--danger)",
  HITL_REJECTED: "var(--warn)",
  INJECTION_BLOCKED: "var(--danger)",
};

export function EventFeed({ frames, index }: { frames: Frame[]; index?: number }) {
  const { t } = useLang();
  // Reveal events in step with playback: only frames up to the current index.
  // Without an index we fall back to the full list (backward-compatible).
  const visible = index == null ? frames : frames.slice(0, index + 1);
  const events = visible
    .filter((f) => f.event)
    .slice(-80)
    .reverse();

  return (
    <div className="flex h-full flex-col rounded-card border bg-[var(--surface)]">
      <div className="border-b px-3 py-2 text-sm font-semibold">{t("events.title")}</div>
      <div className="flex-1 overflow-y-auto p-2">
        {events.length === 0 ? (
          <p className="p-2 text-xs text-muted">{t("events.empty")}</p>
        ) : (
          <ul className="space-y-1">
            {events.map((f, i) => {
              const ev = f.event!;
              const color = COLORS[ev.kind] ?? "var(--text-muted)";
              return (
                <li
                  key={`${f.t}-${i}`}
                  className="flex items-center gap-2 rounded-md border bg-[var(--surface-2)]/40 px-2 py-1.5 text-xs"
                >
                  <span
                    className="h-2 w-2 flex-none rounded-full"
                    style={{ background: color }}
                    aria-hidden
                  />
                  <span className="tnum text-muted">t={f.t}</span>
                  <span className="font-medium" style={{ color }}>
                    {t(`event.${ev.kind}`)}
                  </span>
                  {ev.cert && (
                    <code dir="ltr" className="ms-auto font-mono text-[10px] text-muted">
                      {ev.cert}…
                    </code>
                  )}
                  {ev.conf != null && !ev.cert && (
                    <span className="tnum ms-auto text-muted">conf {ev.conf}</span>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
