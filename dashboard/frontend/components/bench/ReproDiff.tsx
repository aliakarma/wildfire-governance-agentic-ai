"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import type { BenchmarkResponse, PaperResponse } from "@/lib/types";

interface Props {
  live: BenchmarkResponse | null;
  paper: PaperResponse | null;
}

function deviation(liveV: number, paperV: number): number {
  const denom = Math.abs(paperV) > 1e-9 ? Math.abs(paperV) : 1;
  return Math.abs(liveV - paperV) / denom;
}

function Cell({ live, paper }: { live: number; paper: number }) {
  const dev = deviation(live, paper);
  const within = dev < 0.05;
  return (
    <td className="px-3 py-2 text-end">
      <span className="tnum">{live.toFixed(1)}</span>
      <span className="tnum text-muted"> / {paper.toFixed(1)}</span>
      <span
        className="tnum ms-1 inline-block rounded px-1 text-[10px] font-semibold"
        style={{
          background: within ? "color-mix(in srgb, var(--ok) 18%, transparent)" : "color-mix(in srgb, var(--danger) 18%, transparent)",
          color: within ? "var(--ok)" : "var(--danger)",
        }}
      >
        {(dev * 100).toFixed(0)}%
      </span>
    </td>
  );
}

export function ReproDiff({ live, paper }: Props) {
  const { t } = useLang();

  const downloadCsv = () => {
    if (!live) return;
    const header = "method,seed,ld,fp_pct,compliance\n";
    const body = live.raw
      .map((r) => `${r.method},${r.seed},${r.ld ?? ""},${r.fp_pct},${r.compliance}`)
      .join("\n");
    const blob = new Blob([header + body], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "benchmark_raw_per_seed.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const paperByMethod = new Map<string, Record<string, string>>();
  (paper?.rows ?? []).forEach((r) => paperByMethod.set(r.method, r));

  const matched = (live?.rows ?? [])
    .map((lr) => ({ lr, pr: paperByMethod.get(lr.label) }))
    .filter((m) => m.pr) as Array<{ lr: BenchmarkResponse["rows"][0]; pr: Record<string, string> }>;

  return (
    <div className="card p-4">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold">{t("bench.repro")}</h2>
        <button
          onClick={downloadCsv}
          disabled={!live}
          className="ms-auto rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-[var(--surface-2)] disabled:opacity-40"
        >
          ⇩ {t("bench.download")}
        </button>
      </div>
      <p className="mb-3 text-xs text-muted">{t("bench.repro.desc")}</p>

      {!live ? (
        <p className="rounded-lg border border-dashed px-3 py-6 text-center text-xs text-muted">{t("bench.no_live")}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-xs">
            <thead>
              <tr className="border-b text-muted">
                <th className="px-3 py-2 text-start font-medium">{t("bench.methods")}</th>
                <th className="px-3 py-2 text-end font-medium">{t("bench.metric.ld")}</th>
                <th className="px-3 py-2 text-end font-medium">{t("bench.metric.fp")}</th>
                <th className="px-3 py-2 text-end font-medium">{t("bench.metric.comp")}</th>
              </tr>
              <tr className="text-[10px] text-muted">
                <th className="px-3 pb-1 text-start font-normal" />
                <th className="px-3 pb-1 text-end font-normal">live / paper</th>
                <th className="px-3 pb-1 text-end font-normal">live / paper</th>
                <th className="px-3 pb-1 text-end font-normal">live / paper</th>
              </tr>
            </thead>
            <tbody>
              {matched.map(({ lr, pr }) => (
                <tr key={lr.method} className="border-b">
                  <td className="px-3 py-2 font-medium">{lr.label}</td>
                  <Cell live={lr.ld_mean} paper={parseFloat(pr.ld_mean)} />
                  <Cell live={lr.fp_mean} paper={parseFloat(pr.fp_mean)} />
                  <Cell live={lr.compliance_pct} paper={parseFloat(pr.compliance_pct)} />
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-2 flex items-center gap-4 text-[10px] text-muted">
            <span className="inline-flex items-center gap-1">
              <span className="h-2 w-2 rounded" style={{ background: "var(--ok)" }} /> {t("bench.within")}
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2 w-2 rounded" style={{ background: "var(--danger)" }} /> {t("bench.beyond")}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
