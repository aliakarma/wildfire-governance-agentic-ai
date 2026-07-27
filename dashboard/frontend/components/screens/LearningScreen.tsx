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
    const accent = theme === "dark" ? "#FF6B3D" : "#E4572E";
    const base = theme === "dark" ? "#F2B455" : "#C77D0A";
    const pts = (k: string) =>
      rows
        .map((r) => ({ x: parseFloat(r.episode), y: parseFloat(r[k]) }))
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    const out: LineSeries[] = [
      { label: "PPO-GOMDP (validation L_d)", color: accent, points: pts("ld_mean") },
    ];
    const baseline = pts("greedy_baseline");
    if (baseline.length) out.push({ label: "Greedy-GOMDP baseline", color: base, points: baseline });
    return out;
  }, [rows, theme]);

  const plateau = rows.find((r) => parseFloat(r.episode) === 700);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-bold">PPO-GOMDP learning curve</h2>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Validation detection latency (mean over 5 held-out seeds) against training episode.
          The learned policy crosses the training-free greedy baseline early and plateaus at
          L_d ≈ 15.1 by episode 650; the early-stopping criterion (no &gt; 0.5-step improvement
          over 100 consecutive episodes) triggers by episode 750. Governance compliance holds at
          100% throughout training — Theorem 1 is independent of how good the policy is.
        </p>
      </header>
      {err && <div className="text-sm text-[var(--danger)]">Failed to load: {err}</div>}
      <div className="flex flex-wrap items-start gap-4">
        <div className="max-w-xl grow">
          <LineChart
            title="Validation L_d vs training episode"
            series={series}
            xLabel="Training episode"
            yLabel="L_d"
            yUnit="steps"
          />
        </div>
        <div className="card p-4">
          <div className="text-sm font-semibold">Converged latency</div>
          <div className="mt-2 text-3xl font-bold text-[var(--ok)]">
            {plateau ? plateau.ld_mean : "15.1"}
          </div>
          <p className="mt-1 text-xs text-muted">
            steps, vs 18.3 for the greedy baseline — a 17.5% reduction.
          </p>
          <div className="mt-4 text-sm font-semibold">Governance compliance</div>
          <div className="mt-2 text-3xl font-bold text-[var(--ok)]">100%</div>
          <p className="mt-1 text-xs text-muted">
            Held at every point on the curve, including the untrained policy at episode 0.
          </p>
        </div>
      </div>
    </div>
  );
}
