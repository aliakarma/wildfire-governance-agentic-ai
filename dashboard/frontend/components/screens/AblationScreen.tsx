"use client";
import { useEffect, useMemo, useState } from "react";
import { ComparisonChart, type Series } from "@/components/bench/ComparisonChart";
import { useTheme } from "@/components/providers/ThemeProvider";
import { fetchPaperResults } from "@/lib/api";

type Row = Record<string, string>;

export function AblationScreen() {
  const { theme } = useTheme();
  const [rows, setRows] = useState<Row[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetchPaperResults("table2_ablation")
      .then((d) => alive && setRows(d.rows ?? []))
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, []);

  const accent = theme === "dark" ? "#FF6B3D" : "#E4572E";
  const accent2 = theme === "dark" ? "#6FB1FF" : "#2D6BB0";

  const ldSeries: Series[] = useMemo(
    () => rows.map((r) => ({ label: r.config, value: parseFloat(r.ld_mean) || 0, color: accent })),
    [rows, accent],
  );
  const fpSeries: Series[] = useMemo(
    () => rows.map((r) => ({ label: r.config, value: parseFloat(r.fp_mean) || 0, color: accent2 })),
    [rows, accent2],
  );

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-bold">Component ablation</h2>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Knocking out each governance component from the full GOMDP stack (Table 2, n = 20 seeds).
          Signature verification alone defeats direct injection; only removing all authentication
          lets forged alerts through. Removing HITL authorisation is what costs F_p.
        </p>
      </header>
      {err && <div className="text-sm text-red-400">Failed to load: {err}</div>}
      <div className="grid gap-4 lg:grid-cols-2">
        <ComparisonChart title="Detection latency L_d (steps)" series={ldSeries} />
        <ComparisonChart title="False-alert rate F_p (%)" series={fpSeries} unit="%" />
      </div>
      <div className="card overflow-x-auto p-4">
        <div className="mb-2 text-sm font-semibold">Adversarial injections blocked</div>
        <table className="w-full text-sm">
          <thead><tr className="text-left text-muted">
            <th className="px-3 py-1.5">Configuration</th><th className="px-3 py-1.5">Blocked / total</th>
          </tr></thead>
          <tbody>
            {rows.map((r, i) => {
              const blocked = parseInt(r.injections_blocked || "0", 10);
              const total = parseInt(r.injections_total || "0", 10);
              const full = total > 0 && blocked === total;
              return (
                <tr key={i} className="odd:bg-white/[0.02]">
                  <td className="px-3 py-1.5">{r.config}</td>
                  <td className={`px-3 py-1.5 tabular-nums ${full ? "text-emerald-400" : "text-amber-400"}`}>
                    {blocked} / {total}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
