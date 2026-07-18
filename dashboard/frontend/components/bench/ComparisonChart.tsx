"use client";

export interface Series {
  label: string;
  value: number;
  ci?: number;
  color: string;
}

interface Props {
  title: string;
  series: Series[];
  unit?: string;
  max?: number;
}

/** Horizontal bar chart comparing one metric across methods. */
export function ComparisonChart({ title, series, unit, max }: Props) {
  const hi = max ?? Math.max(1, ...series.map((s) => s.value + (s.ci ?? 0)));
  return (
    <div className="card p-4">
      <div className="mb-3 text-sm font-semibold">{title}</div>
      <div dir="ltr" className="space-y-2.5">
        {series.map((s) => {
          const pct = Math.max(0, Math.min(100, (s.value / hi) * 100));
          const ciPct = s.ci ? (s.ci / hi) * 100 : 0;
          return (
            <div key={s.label} className="grid grid-cols-[120px_1fr_auto] items-center gap-2">
              <span className="truncate text-xs text-muted" title={s.label}>
                {s.label}
              </span>
              <div className="relative h-4 rounded bg-[var(--surface-2)]">
                <div className="h-4 rounded" style={{ width: `${pct}%`, background: s.color }} />
                {s.ci ? (
                  <div
                    className="absolute top-1/2 h-2 -translate-y-1/2 border-x"
                    style={{ left: `calc(${pct}% - ${ciPct}%)`, width: `${2 * ciPct}%`, borderColor: "var(--text-muted)" }}
                    aria-hidden
                  />
                ) : null}
              </div>
              <span className="tnum w-14 text-end text-xs font-medium">
                {s.value.toFixed(1)}
                {unit === "%" ? "%" : ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
