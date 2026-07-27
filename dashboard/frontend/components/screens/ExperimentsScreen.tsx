"use client";

import { useEffect, useState } from "react";
import { fetchArtifacts, fetchPaperResults, type ArtifactMeta } from "@/lib/api";

// Provenance -> badge styling + human label. Mirrors PROVENANCE.md; supplementary
// is visibly flagged as NOT in the manuscript.
const PROV: Record<string, { label: string; cls: string }> = {
  exact: {
    label: "Exact — closed-form / deterministic",
    cls: "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-300",
  },
  measured: {
    label: "Measured — aggregated over seeds 0–19",
    cls: "border-sky-300 bg-sky-50 text-sky-800 dark:border-sky-500/40 dark:bg-sky-500/10 dark:text-sky-300",
  },
  specification: {
    label: "Specification — configuration, not a measurement",
    cls: "border-slate-300 bg-slate-100 text-slate-800 dark:border-slate-500/40 dark:bg-slate-500/10 dark:text-slate-300",
  },
  "training-derived": {
    label: "Training-derived — from PPO training runs",
    cls: "border-indigo-300 bg-indigo-50 text-indigo-800 dark:border-indigo-500/40 dark:bg-indigo-500/10 dark:text-indigo-300",
  },
  supplementary: {
    label: "Supplementary — not in the manuscript",
    cls: "border-fuchsia-300 bg-fuchsia-50 text-fuchsia-800 dark:border-fuchsia-500/40 dark:bg-fuchsia-500/10 dark:text-fuchsia-300",
  },
};

interface Row {
  [k: string]: string;
}

function DataTable({ id }: { id: string }) {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    fetchPaperResults(id)
      .then((r) => {
        if (alive) setRows(r.rows ?? []);
      })
      .catch((e) => {
        if (alive) setErr(String(e));
      });
    return () => {
      alive = false;
    };
  }, [id]);

  if (err) return <p className="text-sm text-[var(--danger)] font-medium">Failed to load: {err}</p>;
  if (!rows) return <p className="text-sm text-muted">Loading…</p>;
  if (rows.length === 0) return <p className="text-sm text-muted">No rows.</p>;
  const cols = Object.keys(rows[0]);
  return (
    <div className="overflow-x-auto rounded-lg border border-[var(--border)] mt-2">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="text-left text-muted bg-[var(--surface-2)] font-semibold">
            {cols.map((c) => (
              <th key={c} className="px-3 py-2 border-b border-[var(--border)] whitespace-nowrap text-xs uppercase tracking-wider">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="odd:bg-[var(--surface-2)]/30 hover:bg-[var(--surface-2)]/70 transition-colors">
              {cols.map((c) => (
                <td key={c} className="px-3 py-1.5 border-b border-[var(--border)] whitespace-nowrap tabular-nums text-sm">
                  {r[c]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ArtifactCard({ a }: { a: ArtifactMeta }) {
  const [open, setOpen] = useState(false);
  const prov = PROV[a.provenance] ?? PROV.measured;
  return (
    <div className="card p-4 flex flex-col justify-between space-y-3 shadow-sm hover:shadow-md transition-shadow">
      <div>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted">{a.kind}</span>
              <span className="text-xs text-muted">· {a.paper_ref}</span>
            </div>
            <h3 className="mt-1 text-base font-bold text-[var(--text)]">{a.title}</h3>
          </div>
          <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${prov.cls}`}>
            {prov.label}
          </span>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs text-muted">
          <span className="rounded bg-[var(--surface-2)] border border-[var(--border)] px-2 py-0.5 font-mono text-[11px]">
            route: {a.route}
          </span>
          {a.metrics.slice(0, 4).map((m) => (
            <span key={m} className="rounded bg-[var(--surface-2)] border border-[var(--border)] px-2 py-0.5 font-mono text-[11px]">
              {m}
            </span>
          ))}
          {!a.csv_present && <span className="text-[var(--danger)] font-medium">⚠ CSV missing</span>}
        </div>
      </div>
      <div>
        <button
          onClick={() => setOpen((o) => !o)}
          className="mt-2 text-sm font-semibold text-[var(--accent)] hover:underline inline-flex items-center gap-1 transition-colors"
        >
          {open ? "Hide reference data ▲" : "Show reference data ▼"}
        </button>
        {open && a.csv_present && <DataTable id={a.id} />}
      </div>
    </div>
  );
}

export function ExperimentsScreen() {
  const [arts, setArts] = useState<ArtifactMeta[] | null>(null);
  const [note, setNote] = useState("");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchArtifacts()
      .then((d) => {
        setArts(d.artifacts);
        setNote(d.note);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="p-6 text-[var(--danger)] font-medium">Failed to load experiments: {err}</div>;
  if (!arts) return <div className="p-6 text-muted">Loading experiments…</div>;

  const order = ["exact", "measured", "training-derived", "specification", "supplementary"];
  const groups = order
    .map((p) => ({ prov: p, items: arts.filter((a) => a.provenance === p) }))
    .filter((g) => g.items.length > 0);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-xl font-bold text-[var(--text)]">All experiments</h2>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Every table and figure in the manuscript, each backed by a committed CSV and a
          runnable script (see <code className="rounded bg-[var(--surface-2)] border border-[var(--border)] px-1.5 py-0.5 text-xs font-mono text-[var(--text)]">results/paper/MANIFEST.yaml</code>). {note}
        </p>
        <p className="mt-1 text-xs text-muted">
          {arts.length} artifacts · {arts.filter((a) => a.provenance === "exact").length} exact ·
          {" "}{arts.filter((a) => a.provenance === "measured").length} measured ·
          {" "}{arts.filter((a) => a.provenance === "supplementary").length} supplementary
        </p>
      </header>
      {groups.map((g) => (
        <section key={g.prov} className="space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
            {(PROV[g.prov] ?? PROV.measured).label}
          </h3>
          <div className="grid gap-4 md:grid-cols-2">
            {g.items.map((a) => (
              <ArtifactCard key={a.id} a={a} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
