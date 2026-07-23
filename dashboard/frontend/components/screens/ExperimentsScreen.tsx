"use client";
import { useEffect, useState } from "react";
import { fetchArtifacts, fetchPaperResults, type ArtifactMeta } from "@/lib/api";

// Provenance -> badge styling + human label. supplementary is visibly flagged
// as NOT in the paper; calibration carries the documented-deviation note.
const PROV: Record<string, { label: string; cls: string }> = {
  exact: { label: "Exact reproduction", cls: "border-emerald-500/40 text-emerald-300 bg-emerald-500/10" },
  calibration: { label: "Calibrated (magnitudes are documented deviations)", cls: "border-amber-500/40 text-amber-300 bg-amber-500/10" },
  reference: { label: "Training-derived reference", cls: "border-sky-500/40 text-sky-300 bg-sky-500/10" },
  supplementary: { label: "Supplementary — not in the paper", cls: "border-fuchsia-500/40 text-fuchsia-300 bg-fuchsia-500/10" },
};

interface Row { [k: string]: string }

function DataTable({ id }: { id: string }) {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    fetchPaperResults(id)
      .then((r) => { if (alive) setRows(r.rows ?? []); })
      .catch((e) => { if (alive) setErr(String(e)); });
    return () => { alive = false; };
  }, [id]);

  if (err) return <p className="text-sm text-red-400">Failed to load: {err}</p>;
  if (!rows) return <p className="text-sm text-slate-400">Loading…</p>;
  if (rows.length === 0) return <p className="text-sm text-slate-400">No rows.</p>;
  const cols = Object.keys(rows[0]);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="text-left text-slate-400">
            {cols.map((c) => <th key={c} className="px-3 py-1.5 font-medium border-b border-slate-700 whitespace-nowrap">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="odd:bg-white/[0.02]">
              {cols.map((c) => <td key={c} className="px-3 py-1.5 border-b border-slate-800 whitespace-nowrap tabular-nums">{r[c]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ArtifactCard({ a }: { a: ArtifactMeta }) {
  const [open, setOpen] = useState(false);
  const prov = PROV[a.provenance] ?? PROV.reference;
  return (
    <div className="rounded-xl border border-slate-700/60 bg-white/[0.02] p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-500">{a.kind}</span>
            <span className="text-xs text-slate-400">· {a.paper_ref}</span>
          </div>
          <h3 className="text-base font-semibold text-slate-100">{a.title}</h3>
        </div>
        <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${prov.cls}`}>{prov.label}</span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-400">
        <span className="rounded bg-slate-700/40 px-1.5 py-0.5">route: {a.route}</span>
        {a.metrics.slice(0, 4).map((m) => <span key={m} className="rounded bg-slate-700/40 px-1.5 py-0.5">{m}</span>)}
        {!a.csv_present && <span className="text-red-400">⚠ CSV missing</span>}
      </div>
      <button
        onClick={() => setOpen((o) => !o)}
        className="mt-3 text-sm font-medium text-sky-300 hover:text-sky-200"
      >
        {open ? "Hide reference data ▲" : "Show reference data ▼"}
      </button>
      {open && a.csv_present && <div className="mt-3"><DataTable id={a.id} /></div>}
    </div>
  );
}

export function ExperimentsScreen() {
  const [arts, setArts] = useState<ArtifactMeta[] | null>(null);
  const [note, setNote] = useState("");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchArtifacts()
      .then((d) => { setArts(d.artifacts); setNote(d.note); })
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="p-6 text-red-400">Failed to load experiments: {err}</div>;
  if (!arts) return <div className="p-6 text-slate-400">Loading experiments…</div>;

  const order = ["exact", "calibration", "reference", "supplementary"];
  const groups = order
    .map((p) => ({ prov: p, items: arts.filter((a) => a.provenance === p) }))
    .filter((g) => g.items.length > 0);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-bold text-slate-100">All experiments</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Every table and figure in the manuscript, each backed by a committed CSV and a
          runnable script (see <code>results/paper/MANIFEST.yaml</code>). Values shown here are
          the frozen paper reference. {note}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          {arts.length} artifacts · {arts.filter((a) => a.provenance === "exact").length} exact ·
          {" "}{arts.filter((a) => a.provenance === "calibration").length} calibrated ·
          {" "}{arts.filter((a) => a.provenance === "supplementary").length} supplementary
        </p>
      </header>
      {groups.map((g) => (
        <section key={g.prov} className="space-y-3">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
            {(PROV[g.prov] ?? PROV.reference).label}
          </h3>
          <div className="grid gap-3 md:grid-cols-2">
            {g.items.map((a) => <ArtifactCard key={a.id} a={a} />)}
          </div>
        </section>
      ))}
    </div>
  );
}
