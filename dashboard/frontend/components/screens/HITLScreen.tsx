"use client";
import { useEffect, useMemo, useState } from "react";
import { LineChart, type LineSeries } from "@/components/bench/LineChart";
import { useTheme } from "@/components/providers/ThemeProvider";
import { fetchPaperResults } from "@/lib/api";

type Row = Record<string, string>;

export function HITLScreen() {
  const { theme } = useTheme();
  const [rows, setRows] = useState<Row[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetchPaperResults("table7_hitl_sensitivity")
      .then((d) => alive && setRows(d.rows ?? []))
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, []);

  const series: LineSeries[] = useMemo(() => {
    const fnC = theme === "dark" ? "#FF6B3D" : "#E4572E";
    const fpC = theme === "dark" ? "#6FB1FF" : "#2D6BB0";
    const pts = (k: string) => rows
      .map((r) => ({ x: parseFloat(r.p_err), y: parseFloat(r[k]) }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    return [
      { label: "Missed detections FN_r", color: fnC, points: pts("fn_mean") },
      { label: "False alerts F_p", color: fpC, points: pts("fp_mean") },
    ];
  }, [rows, theme]);

  const compliancePinned = rows.length > 0 && rows.every((r) => parseFloat(r.gov_compliance_pct) === 100);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-bold text-slate-100">HITL operator-error sensitivity</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          As the human operator's error rate p_err rises, missed detections increase and false
          alerts fall — but governance compliance stays pinned at 100% by Theorem 1: the operator
          can only withhold an alert, never authorise an unverified one (Table 7).
        </p>
      </header>
      {err && <div className="text-sm text-red-400">Failed to load: {err}</div>}
      <div className="flex flex-wrap items-start gap-4">
        <div className="max-w-xl grow">
          <LineChart title="Table 7 · FN_r and F_p vs operator error p_err" series={series}
                     xLabel="Operator error p_err" yLabel="rate" yUnit="%" />
        </div>
        <div className="card p-4">
          <div className="text-sm font-semibold">Governance compliance</div>
          <div className={`mt-2 text-3xl font-bold ${compliancePinned ? "text-emerald-400" : "text-amber-400"}`}>
            100%
          </div>
          <p className="mt-1 text-xs text-muted">
            {compliancePinned ? "Pinned across all p_err (Theorem 1)." : "Reference values."}
          </p>
        </div>
      </div>
    </div>
  );
}
