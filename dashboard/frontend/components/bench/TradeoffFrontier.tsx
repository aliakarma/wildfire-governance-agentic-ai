"use client";
import { useEffect, useState } from "react";
import { useLang } from "@/components/providers/LanguageProvider";
import { useTheme } from "@/components/providers/ThemeProvider";
import { fetchPaperResults } from "@/lib/api";

const COLORS: Record<string, { light: string; dark: string; label: string }> = {
  ppo_gomdp: { light: "#E4572E", dark: "#FF6B3D", label: "PPO-GOMDP" },
  greedy_gomdp: { light: "#C77D0A", dark: "#F2B455", label: "Greedy-GOMDP" },
  ppo_cmdp: { light: "#7A5AF8", dark: "#A78BFA", label: "PPO-CMDP" },
  wcsac: { light: "#0F8A6A", dark: "#35D0A5", label: "WCSAC" },
  adaptive_ai: { light: "#8A8F98", dark: "#9AA6B8", label: "Adaptive AI" },
  static: { light: "#5B616E", dark: "#6B7280", label: "Static" },
};

interface Pt {
  config: string;
  ld: number;
  fp: number;
}

export function TradeoffFrontier() {
  const { t } = useLang();
  const { theme } = useTheme();
  const [pts, setPts] = useState<Pt[]>([]);

  useEffect(() => {
    let alive = true;
    fetchPaperResults("fig5_tradeoff_data")
      .then((d) => {
        if (!alive || !d.rows) return;
        setPts(
          d.rows.map((r: Record<string, string>) => ({
            config: r.config,
            ld: parseFloat(r.ld_mean),
            fp: parseFloat(r.fp_mean),
          })),
        );
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const W = 360;
  const H = 240;
  const pad = { l: 40, r: 12, t: 12, b: 34 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const maxLd = Math.max(40, ...pts.map((p) => p.ld)) * 1.05;
  const maxFp = Math.max(25, ...pts.map((p) => p.fp)) * 1.05;
  const x = (ld: number) => pad.l + (ld / maxLd) * iw;
  const y = (fp: number) => pad.t + (1 - fp / maxFp) * ih;
  const axis = theme === "dark" ? "#263041" : "#DAD6CE";

  return (
    <div className="card p-4">
      <div className="mb-1 flex items-center gap-2">
        <span className="text-sm font-semibold">{t("bench.frontier")}</span>
        <span className="rounded-full border border-fuchsia-500/40 bg-fuchsia-500/10 px-2 py-0.5 text-[10px] font-medium text-fuchsia-300">
          Supplementary — not in the paper
        </span>
      </div>
      <p className="mb-2 text-[11px] text-muted">↙ lower-left is better (fast detection, few false alerts)</p>
      <div dir="ltr">
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Latency vs false-alert tradeoff">
          <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke={axis} />
          <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke={axis} />
          <text x={pad.l - 6} y={pad.t + 6} textAnchor="end" fontSize={9} fill="var(--text-muted)">
            {Math.round(maxFp)}
          </text>
          <text x={pad.l - 6} y={H - pad.b} textAnchor="end" fontSize={9} fill="var(--text-muted)">
            0
          </text>
          <text x={W - pad.r} y={H - pad.b + 14} textAnchor="end" fontSize={9} fill="var(--text-muted)">
            L_d {Math.round(maxLd)}
          </text>
          <text x={pad.l} y={H - pad.b + 14} textAnchor="start" fontSize={9} fill="var(--text-muted)">
            F_p
          </text>
          {pts.map((p) => {
            const c = COLORS[p.config] ?? { light: "#8A8F98", dark: "#9AA6B8", label: p.config };
            const fill = theme === "dark" ? c.dark : c.light;
            return (
              <g key={p.config}>
                <circle cx={x(p.ld)} cy={y(p.fp)} r={6} fill={fill} stroke="var(--surface)" strokeWidth={1} />
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px]">
        {pts.map((p) => {
          const c = COLORS[p.config] ?? { light: "#8A8F98", dark: "#9AA6B8", label: p.config };
          return (
            <span key={p.config} className="inline-flex items-center gap-1 text-muted">
              <span className="h-2 w-2 rounded-full" style={{ background: theme === "dark" ? c.dark : c.light }} />
              {c.label}
            </span>
          );
        })}
      </div>
    </div>
  );
}
