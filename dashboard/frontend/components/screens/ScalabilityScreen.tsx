"use client";
import { useEffect, useMemo, useState } from "react";
import { LineChart, type LineSeries } from "@/components/bench/LineChart";
import { useTheme } from "@/components/providers/ThemeProvider";
import { fetchPaperResults } from "@/lib/api";

// Method colours keyed by a normalized name so both display labels ("PPO-GOMDP")
// and config ids ("ppo_gomdp") resolve to the same colour.
const COLORS: Record<string, { light: string; dark: string }> = {
  ppogomdp: { light: "#E4572E", dark: "#FF6B3D" },
  greedygomdp: { light: "#C77D0A", dark: "#F2B455" },
  centralsig: { light: "#2D6BB0", dark: "#6FB1FF" },
  shieldppo: { light: "#1E8E6A", dark: "#3FD9A6" },
  safelayer: { light: "#0E9EAE", dark: "#42D4E4" },
  ppocmdp: { light: "#7A5AF8", dark: "#A78BFA" },
  wcsac: { light: "#B14AA0", dark: "#E27CD0" },
  adaptiveai: { light: "#8A8F98", dark: "#9AA6B8" },
  static: { light: "#5B616E", dark: "#6B7280" },
  bound: { light: "#9AA6B8", dark: "#5B616E" },
};
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

type Row = Record<string, string>;

function groupSeries(rows: Row[], yKey: string, resolve: (c: string) => string): LineSeries[] {
  const by = new Map<string, LineSeries>();
  for (const r of rows) {
    const cfg = r.config;
    if (!cfg) continue;
    const n = parseFloat(r.n_uavs);
    const yv = parseFloat(r[yKey]);
    if (!Number.isFinite(n) || !Number.isFinite(yv)) continue;
    if (!by.has(cfg)) by.set(cfg, { label: cfg, color: resolve(cfg), points: [] });
    by.get(cfg)!.points.push({ x: n, y: yv });
  }
  return [...by.values()];
}

export function ScalabilityScreen() {
  const { theme } = useTheme();
  const [fp, setFp] = useState<Row[]>([]);
  const [lat, setLat] = useState<Row[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all([fetchPaperResults("fig2_false_alerts"), fetchPaperResults("fig3_latency_data")])
      .then(([a, b]) => { if (!alive) return; setFp(a.rows ?? []); setLat(b.rows ?? []); })
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, []);

  const color = (name: string) => {
    const c = COLORS[norm(name)] ?? { light: "#8A8F98", dark: "#9AA6B8" };
    return theme === "dark" ? c.dark : c.light;
  };

  const fpSeries = useMemo(() => groupSeries(fp, "fp_mean", color), [fp, theme]); // eslint-disable-line react-hooks/exhaustive-deps
  const ldSeries = useMemo(() => groupSeries(lat, "ld_mean", color), [lat, theme]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-bold text-slate-100">Scalability</h2>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          How the pipeline scales with fleet size N (Figures 2 &amp; 3 of the manuscript,
          n = 20 seeds). Governance keeps false alerts low while detection latency falls
          with fleet size; the ranking is stable across N ∈ {"{5, 10, 20, 40}"}.
        </p>
      </header>
      {err && <div className="text-sm text-red-400">Failed to load: {err}</div>}
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <LineChart title="Figure 2 · False-alert rate vs fleet size" series={fpSeries}
                     xLabel="Fleet size N (UAVs)" yLabel="F_p" yUnit="%" />
        </div>
        <div>
          <LineChart title="Figure 3 · Detection latency vs fleet size" series={ldSeries}
                     xLabel="Fleet size N (UAVs)" yLabel="L_d" yUnit="steps" />
        </div>
      </div>
    </div>
  );
}
