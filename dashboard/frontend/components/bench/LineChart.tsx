"use client";
import { useTheme } from "@/components/providers/ThemeProvider";

export interface LinePoint { x: number; y: number }
export interface LineSeries { label: string; color: string; points: LinePoint[] }

interface Props {
  title: string;
  series: LineSeries[];
  xLabel: string;
  yLabel: string;
  yUnit?: string;
  yMax?: number;
}

/** Generic theme-aware multi-series line chart (SVG, no chart lib). Used for the
 *  manuscript figures F_p-vs-N (Fig 2), L_d-vs-N (Fig 4), learning curve (Fig 3). */
export function LineChart({ title, series, xLabel, yLabel, yUnit, yMax }: Props) {
  const { theme } = useTheme();
  const axis = theme === "dark" ? "#263041" : "#DAD6CE";
  const W = 420, H = 260;
  const pad = { l: 44, r: 12, t: 14, b: 40 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;

  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 1);
  const yHi = yMax ?? Math.max(1, ...ys) * 1.08;
  const x = (v: number) => pad.l + ((v - xMin) / (xMax - xMin || 1)) * iw;
  const y = (v: number) => pad.t + (1 - v / yHi) * ih;

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yHi);
  const xTicks = Array.from(new Set(xs)).sort((a, b) => a - b);

  return (
    <div className="card p-4">
      <div className="mb-2 text-sm font-semibold">{title}</div>
      <div dir="ltr">
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
          {/* gridlines + y ticks */}
          {yTicks.map((tv) => (
            <g key={tv}>
              <line x1={pad.l} y1={y(tv)} x2={W - pad.r} y2={y(tv)} stroke={axis} strokeDasharray="2 3" opacity={0.5} />
              <text x={pad.l - 6} y={y(tv) + 3} textAnchor="end" fontSize={9} fill="var(--text-muted)">{tv.toFixed(0)}</text>
            </g>
          ))}
          {/* axes */}
          <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke={axis} />
          <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke={axis} />
          {/* x ticks */}
          {xTicks.map((tv) => (
            <text key={tv} x={x(tv)} y={H - pad.b + 14} textAnchor="middle" fontSize={9} fill="var(--text-muted)">{tv}</text>
          ))}
          <text x={(pad.l + W - pad.r) / 2} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--text-muted)">{xLabel}</text>
          <text x={12} y={pad.t + 4} fontSize={10} fill="var(--text-muted)">{yLabel}{yUnit ? ` (${yUnit})` : ""}</text>
          {/* series */}
          {series.map((s) => {
            const pts = [...s.points].sort((a, b) => a.x - b.x);
            const d = pts.map((p) => `${x(p.x)},${y(p.y)}`).join(" ");
            return (
              <g key={s.label}>
                <polyline points={d} fill="none" stroke={s.color} strokeWidth={2} />
                {pts.map((p, i) => (
                  <circle key={i} cx={x(p.x)} cy={y(p.y)} r={3} fill={s.color} stroke="var(--surface)" strokeWidth={1} />
                ))}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px]">
        {series.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1 text-muted">
            <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
