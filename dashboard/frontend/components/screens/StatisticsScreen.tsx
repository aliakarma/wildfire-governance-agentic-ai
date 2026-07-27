"use client";
import { useEffect, useState } from "react";
import { fetchPaperResults } from "@/lib/api";

type Row = Record<string, string>;

const MAIN_LABEL: Record<string, string> = {
  ppo_gomdp: "PPO-GOMDP",
  greedy_gomdp: "Greedy-GOMDP",
  ppo_cmdp: "PPO-CMDP",
  wcsac: "WCSAC",
  adaptive_ai: "Adaptive AI",
  static: "Static",
};

/** Manuscript "---": the metric is not defined for that configuration. */
const dash = (v?: string) => (v === undefined || v === "" ? "—" : v);
const pm = (m?: string, s?: string) =>
  m === undefined || m === "" ? "—" : s ? `${m} ± ${s}` : m;

export function StatisticsScreen() {
  const [tests, setTests] = useState<Row[]>([]);
  const [main, setMain] = useState<Row[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all([
      fetchPaperResults("statistical_tests"),
      fetchPaperResults("table1_rl_comparison_main"),
    ])
      .then(([a, b]) => {
        if (!alive) return;
        setTests(a.rows ?? []);
        setMain(b.rows ?? []);
      })
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-bold">Statistics &amp; full-metric comparison</h2>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Every significance claim in the manuscript, plus the full-metric table behind
          Table&nbsp;1. Seeds 0–19 are common random numbers across configurations, so all
          comparisons are paired; family-wise error is controlled with Holm–Bonferroni.
        </p>
      </header>
      {err && <div className="text-sm text-[var(--danger)]">Failed to load: {err}</div>}

      <section className="card overflow-x-auto p-4">
        <div className="mb-1 flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold">Significance and equivalence tests</span>
          <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
            Exact
          </span>
        </div>
        <p className="mb-3 text-[11px] text-muted">
          The PPO-GOMDP vs PPO-CMDP latency claim is an <em>equivalence</em> claim: TOST with an
          a-priori margin of one blockchain-commit delay (±1.2 steps).
        </p>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b text-start text-muted">
              <th className="px-2 py-2 text-start font-medium">Comparison</th>
              <th className="px-2 py-2 text-start font-medium">Metric</th>
              <th className="px-2 py-2 text-start font-medium">Test</th>
              <th className="px-2 py-2 text-end font-medium">Mean A</th>
              <th className="px-2 py-2 text-end font-medium">Mean B</th>
              <th className="px-2 py-2 text-end font-medium">Statistic</th>
              <th className="px-2 py-2 text-end font-medium">p</th>
              <th className="px-2 py-2 text-end font-medium">Cohen&apos;s d</th>
              <th className="px-2 py-2 text-start font-medium">Conclusion</th>
            </tr>
          </thead>
          <tbody>
            {tests.map((r, i) => (
              <tr key={i} className="border-b odd:bg-white/[0.02]">
                <td className="px-2 py-2 font-medium">{r.comparison}</td>
                <td className="px-2 py-2">{r.metric}</td>
                <td className="px-2 py-2 text-muted">{r.test}</td>
                <td className="tnum px-2 py-2 text-end">{r.mean_a}</td>
                <td className="tnum px-2 py-2 text-end">{r.mean_b}</td>
                <td className="tnum px-2 py-2 text-end">{r.statistic}</td>
                <td className="tnum px-2 py-2 text-end">{r.p_value}</td>
                <td className="tnum px-2 py-2 text-end">{dash(r.effect_size_d)}</td>
                <td
                  className="px-2 py-2"
                  style={{
                    color: (r.conclusion ?? "").startsWith("not")
                      ? "var(--warn)"
                      : "var(--ok)",
                  }}
                >
                  {r.conclusion}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card overflow-x-auto p-4">
        <div className="mb-1 flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold">Full-metric comparison (N = 20, 20 seeds)</span>
          <span className="rounded-full border border-[var(--warn)] px-2 py-0.5 text-[10px] font-semibold text-[var(--warn)]">
            Manuscript values
          </span>
        </div>
        <p className="mb-3 text-[11px] text-muted">
          Governance overhead is L_d relative to the ungoverned Adaptive AI baseline. Compliance is
          definitional for the governed rows (Theorem 1) and for the ungoverned rows (no gate).
        </p>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b text-muted">
              <th className="px-2 py-2 text-start font-medium">Configuration</th>
              <th className="px-2 py-2 text-end font-medium">L_d (steps)</th>
              <th className="px-2 py-2 text-end font-medium">F_p (%)</th>
              <th className="px-2 py-2 text-end font-medium">FN_r (%)</th>
              <th className="px-2 py-2 text-end font-medium">BC delay</th>
              <th className="px-2 py-2 text-end font-medium">Human rev.</th>
              <th className="px-2 py-2 text-end font-medium">Compliance</th>
              <th className="px-2 py-2 text-end font-medium">Gov. overhead</th>
            </tr>
          </thead>
          <tbody>
            {main.map((r, i) => (
              <tr key={i} className="border-b odd:bg-white/[0.02]">
                <td className="px-2 py-2 font-medium">{MAIN_LABEL[r.config] ?? r.config}</td>
                <td className="tnum px-2 py-2 text-end">{pm(r.ld_mean, r.ld_std)}</td>
                <td className="tnum px-2 py-2 text-end">{pm(r.fp_mean, r.fp_std)}</td>
                <td className="tnum px-2 py-2 text-end">{pm(r.fn_mean, r.fn_std)}</td>
                <td className="tnum px-2 py-2 text-end">{dash(r.bc_delay_mean)}</td>
                <td className="tnum px-2 py-2 text-end">{dash(r.human_review_mean)}</td>
                <td
                  className="tnum px-2 py-2 text-end"
                  style={{
                    color:
                      parseFloat(r.compliance_pct) >= 99.9
                        ? "var(--ok)"
                        : parseFloat(r.compliance_pct) === 0
                          ? "var(--danger)"
                          : "var(--warn)",
                  }}
                >
                  {r.compliance_pct}%
                </td>
                <td className="tnum px-2 py-2 text-end">
                  {r.gov_overhead_pct === "" ? "—" : `${r.gov_overhead_pct}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
