"use client";
import { useEffect, useState } from "react";
import { fetchPaperResults } from "@/lib/api";

type Row = Record<string, string>;

export function CNNScreen() {
  const [rows, setRows] = useState<Row[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetchPaperResults("table10_cnn_ablation")
      .then((d) => alive && setRows(d.rows ?? []))
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, []);

  const cols = rows.length ? Object.keys(rows[0]) : [];

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-bold text-slate-100">Architecture ablation (MLP vs CNN)</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Policy-network architecture comparison (Table 10). The CNN converges faster with far
          fewer parameters at comparable latency. These are{" "}
          <span className="text-sky-300">training-derived reference values</span>, not recomputed live.
        </p>
      </header>
      {err && <div className="text-sm text-red-400">Failed to load: {err}</div>}
      <div className="card max-w-2xl overflow-x-auto p-4">
        <table className="w-full text-sm">
          <thead><tr className="text-left text-muted">
            {cols.map((c) => <th key={c} className="px-3 py-2 whitespace-nowrap">{c}</th>)}
          </tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="odd:bg-white/[0.02]">
                {cols.map((c) => <td key={c} className="px-3 py-2 tabular-nums whitespace-nowrap">{r[c]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
