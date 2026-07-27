"use client";
import { useEffect, useState } from "react";
import { fetchPaperResults } from "@/lib/api";

type Row = Record<string, string>;

/** Closed-form consensus-security reference: validator compromise (Theorem 2,
 *  stochastic + deterministic), the validator-count sweep, and the m-of-n
 *  multisignature ablation. These reproduce the manuscript exactly. */
export function ConsensusReference() {
  const [ksweep, setKsweep] = useState<Row[]>([]);
  const [theory, setTheory] = useState<Row[]>([]);
  const [empirical, setEmpirical] = useState<Row[]>([]);
  const [multisig, setMultisig] = useState<Row[]>([]);

  useEffect(() => {
    let alive = true;
    const get = (id: string, set: (r: Row[]) => void) =>
      fetchPaperResults(id).then((d) => alive && set(d.rows ?? [])).catch(() => {});
    get("table6_ksweep", setKsweep);
    get("table5_byzantine_theory", setTheory);
    get("table5_byzantine_empirical", setEmpirical);
    get("table9_multisig", setMultisig);
    return () => { alive = false; };
  }, []);

  const mrow = multisig[0];

  return (
    <div className="card p-4">
      <div className="mb-1 flex items-center gap-2">
        <span className="text-sm font-semibold">Consensus security (closed form)</span>
        <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
          Exact
        </span>
      </div>

      <p className="mb-1.5 mt-2 text-[11px] text-muted">
        Theorem 2 breach probability at k = 7, f = 2, against a single signature-verifying
        server. Consensus wins below the p_c ≈ 0.26 crossover.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead><tr className="text-left text-muted">
            <th className="px-2 py-1">p_c</th>
            <th className="px-2 py-1">P(breach) GOMDP</th>
            <th className="px-2 py-1">single verifier</th>
          </tr></thead>
          <tbody>
            {theory.map((r, i) => (
              <tr key={i} className="odd:bg-white/[0.02]">
                <td className="px-2 py-1 tabular-nums">{r.p_c}</td>
                <td
                  className="px-2 py-1 tabular-nums"
                  style={{ color: parseFloat(r.p_break_gomdp) < parseFloat(r.p_break_sig) ? "var(--ok)" : "var(--warn)" }}
                >
                  {r.p_break_gomdp}
                </td>
                <td className="px-2 py-1 tabular-nums">{r.p_break_sig}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mb-1.5 mt-3 text-[11px] text-muted">
        Deterministic compromise: breach requires f_c ≥ f+1 = 3 validators.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead><tr className="text-left text-muted">
            <th className="px-2 py-1">f_c</th><th className="px-2 py-1">breach</th><th className="px-2 py-1">F_p (%)</th>
          </tr></thead>
          <tbody>
            {empirical.map((r, i) => {
              const tolerated = (r.breach ?? "").startsWith("0/");
              return (
                <tr key={i} className="odd:bg-white/[0.02]">
                  <td className="px-2 py-1 tabular-nums">{r.f_c}</td>
                  <td className="px-2 py-1 tabular-nums" style={{ color: tolerated ? "var(--ok)" : "var(--danger)" }}>
                    {r.breach}
                  </td>
                  <td className="px-2 py-1 tabular-nums">{r.fp_pct}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mb-1.5 mt-3 text-[11px] text-muted">
        Validator-count sweep at p_c = 0.10, f = ⌊(k−1)/3⌋.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead><tr className="text-left text-muted">
            <th className="px-2 py-1">k</th><th className="px-2 py-1">f</th>
            <th className="px-2 py-1">theory</th><th className="px-2 py-1">empirical</th>
          </tr></thead>
          <tbody>
            {ksweep.map((r, i) => (
              <tr key={i} className="odd:bg-white/[0.02]">
                <td className="px-2 py-1 tabular-nums">{r.k}</td>
                <td className="px-2 py-1 tabular-nums">{r.f}</td>
                <td className="px-2 py-1 tabular-nums">{r.p_break_gomdp_theory}</td>
                <td className="px-2 py-1 tabular-nums">{r.empirical}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {mrow && (
        <p className="mt-3 text-[11px] text-muted">
          m-of-n multisig: forged alerts blocked{" "}
          <span className="tabular-nums text-emerald-300">
            {mrow.injections_blocked}/{mrow.injections_total}
          </span>{" "}
          at L_d {mrow.ld_mean} and F_p {mrow.fp_mean}% — a forged alert carries zero of the
          m required signatures.
        </p>
      )}
    </div>
  );
}
