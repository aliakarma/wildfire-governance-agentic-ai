"use client";
import { useEffect, useState } from "react";
import { fetchPaperResults } from "@/lib/api";

type Row = Record<string, string>;

/** Manuscript adversarial-robustness table: F_p per attack for GOMDP, the
 *  single signature-verifying server, and the unauthenticated centralized
 *  system, plus the direct alert-injection outcome. */
export function AttackReference() {
  const [rows, setRows] = useState<Row[]>([]);

  useEffect(() => {
    let alive = true;
    fetchPaperResults("table3_adversarial")
      .then((d) => alive && setRows(d.rows ?? []))
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  const fp = rows.filter((r) => r.metric === "fp_pct");
  const inj = rows.find((r) => r.metric === "injection_ratio");

  return (
    <div className="card overflow-x-auto p-4">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold">Attack resistance</span>
        <span className="rounded-full border border-[var(--warn)] px-2 py-0.5 text-[10px] font-semibold text-[var(--warn)]">
          Manuscript values
        </span>
      </div>
      <p className="mb-3 text-[11px] text-muted">
        False public alert rate F_p (%) under each attack. Spoofing corrupts the belief state and
        bypasses governance entirely, so it raises F_p for every architecture — but the HITL gate
        bounds the damage. Compliance stays 100% throughout: spoofing cannot manufacture a valid
        authorization certificate.
      </p>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b text-muted">
            <th className="px-2 py-2 text-start font-medium">Attack / condition</th>
            <th className="px-2 py-2 text-start font-medium">Param.</th>
            <th className="px-2 py-2 text-end font-medium">GOMDP</th>
            <th className="px-2 py-2 text-end font-medium">Central+Sig</th>
            <th className="px-2 py-2 text-end font-medium">Central</th>
          </tr>
        </thead>
        <tbody>
          {fp.map((r, i) => (
            <tr key={i} className="border-b odd:bg-white/[0.02]">
              <td className="px-2 py-2 font-medium">{r.attack_type}</td>
              <td className="px-2 py-2 text-muted">{r.parameter}</td>
              <td className="tnum px-2 py-2 text-end text-[var(--ok)]">{r.gomdp}%</td>
              <td className="tnum px-2 py-2 text-end">{r.central_sig}%</td>
              <td className="tnum px-2 py-2 text-end text-[var(--danger)]">{r.central}%</td>
            </tr>
          ))}
          {inj && (
            <tr className="odd:bg-white/[0.02]">
              <td className="px-2 py-2 font-medium">Alert injection (success)</td>
              <td className="px-2 py-2 text-muted">{inj.parameter}</td>
              <td className="tnum px-2 py-2 text-end text-[var(--ok)]">{inj.gomdp}</td>
              <td className="tnum px-2 py-2 text-end text-[var(--ok)]">{inj.central_sig}</td>
              <td className="tnum px-2 py-2 text-end text-[var(--danger)]">{inj.central}</td>
            </tr>
          )}
        </tbody>
      </table>
      <p className="mt-3 text-[11px] text-muted">
        Injection is blocked completely by GOMDP <em>and</em> by Central+Sig, and succeeds
        completely against the unauthenticated system — so injection resistance is attributable to
        Ed25519 signature verification, not to the blockchain layer generally.
      </p>
    </div>
  );
}
