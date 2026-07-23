"use client";
import { useEffect, useMemo, useState } from "react";
import { LineChart, type LineSeries } from "@/components/bench/LineChart";
import { useTheme } from "@/components/providers/ThemeProvider";
import { fetchPaperResults } from "@/lib/api";

type Row = Record<string, string>;

export function LearningScreen() {
  const { theme } = useTheme();
  const [rows, setRows] = useState<Row[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetchPaperResults("fig3_learning_curve")
      .then((d) => alive && setRows(d.rows ?? []))
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, []);

  const series: LineSeries[] = useMemo(() => {
    const c = theme === "dark" ? "#FF6B3D" : "#E4572E";
    const base = theme === "dark" ? "#9AA6B8" : "#8A8F98";
    const curve = rows
      .map((r) => ({ x: parseFloat(r.episode), y: parseFloat(r.ld_mean) }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    const baseline = rows
      .map((r) => ({ x: parseFloat(r.episode), y: parseFloat(r.greedy_baseline) }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    const out: LineSeries[] = [{ label: "PPO-GOMDP (validation L_d)", color: c, points: curve }];
    if (baseline.length) out.push({ label: "Greedy baseline", color: base, points: baseline });
    return out;
  }, [rows, theme]);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-bold text-slate-100">Learning curve</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          PPO-GOMDP validation detection latency over training episodes (Figure 3, frozen
          manuscript reference). The learned policy converges below the training-free greedy
          baseline. Marked <span className="text-sky-300">training-derived reference</span>.
        </p>
      </header>
      {err && <div className="text-sm text-red-400">Failed to load: {err}</div>}
      <div className="max-w-xl">
        <LineChart title="Figure 3 · Validation L_d vs training episode" series={series}
                   xLabel="Training episode" yLabel="L_d" yUnit="steps" />
      </div>
    </div>
  );
}
