"use client";
import { useEffect, useState } from "react";
import { fetchPaperResults } from "@/lib/api";

type Row = Record<string, string>;

/** Closed-form consensus-security reference (Theorem 2): validator-count sweep
 *  (Table 6, ksweep), Byzantine compromise (Table 5), and m-of-n multisig
 *  injection-blocking (Table 9). These reproduce the paper EXACTLY. */
export function ConsensusReference() {
  const [ksweep, setKsweep] = useState<Row[]>([]);
  const [multisig, setMultisig] = useState<Row[]>([]);

  useEffect(() => {
    let alive = true;
    fetchPaperResults("table6_ksweep").then((d) => alive && setKsweep(d.rows ?? [])).catch(() => {});
    fetchPaperResults("table9_multisig").then((d) => alive && setMultisig(d.rows ?? [])).catch(() => {});
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
      <p className="mb-2 text-[11px] text-muted">
        Theorem 2 breach probability by validator count k (p_c = 0.10, f = ⌊(k−1)/3⌋).
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead><tr className="text-left text-muted">
            <th className="px-2 py-1">k</th><th className="px-2 py-1">f</th>
            <th className="px-2 py-1">P(breach) theory</th><th className="px-2 py-1">empirical</th>
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
        <p className="mt-2 text-[11px] text-muted">
          m-of-n multisig (Table 9): forged alerts blocked{" "}
          <span className="text-emerald-300 tabular-nums">
            {mrow.injections_blocked}/{mrow.injections_total}
          </span>{" "}
          — a forged alert carries zero of the m required signatures.
        </p>
      )}
    </div>
  );
}
