"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import { useBench } from "@/components/providers/BenchmarkProvider";

const METHODS: Array<{ id: string; label: string }> = [
  { id: "greedy_gomdp", label: "Greedy-GOMDP" },
  { id: "central_sig", label: "Central+Sig" },
  { id: "ppo_cmdp", label: "PPO-CMDP" },
  { id: "adaptive_ai", label: "Adaptive AI" },
  { id: "static", label: "Static" },
];

export function BenchmarkControls() {
  const { t } = useLang();
  const b = useBench();

  return (
    <div className="card flex flex-col gap-4 p-4">
      <div className="flex flex-wrap items-start gap-x-8 gap-y-4">
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">{t("bench.methods")}</div>
          <div className="flex flex-wrap gap-2">
            {METHODS.map((m) => {
              const on = b.methods.includes(m.id);
              return (
                <button
                  key={m.id}
                  onClick={() => b.toggleMethod(m.id)}
                  className={`rounded-md border px-3 py-1.5 text-xs font-medium ${
                    on ? "border-[var(--accent)] bg-[var(--surface-2)] text-accent" : "text-muted"
                  }`}
                  aria-pressed={on}
                >
                  {m.label}
                </button>
              );
            })}
          </div>
        </div>

        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">{t("bench.source")}</div>
          <div className="inline-flex rounded-md border p-0.5">
            {(["live", "paper"] as const).map((s) => (
              <button
                key={s}
                onClick={() => b.setSource(s)}
                className={`rounded px-3 py-1.5 text-xs font-medium ${
                  b.source === s ? "bg-[var(--accent)] text-white" : "text-muted"
                }`}
              >
                {t(s === "live" ? "bench.live" : "bench.paper")}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-1 flex-wrap items-end gap-4">
          <Stepper label={t("param.seed")} value={b.config.n_seeds} min={2} max={10} onChange={(v) => b.setConfig({ n_seeds: v })} />
          <Stepper label={t("param.grid_size")} value={b.config.grid_size} min={30} max={100} step={10} onChange={(v) => b.setConfig({ grid_size: v })} />
          <Stepper label={t("param.n_timesteps")} value={b.config.n_timesteps} min={100} max={600} step={50} onChange={(v) => b.setConfig({ n_timesteps: v })} />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={b.runLive}
          disabled={b.loading || b.methods.length === 0}
          className="rounded-lg bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
        >
          {b.loading ? t("bench.running") : t("bench.run")}
        </button>
        {b.error && <span className="text-xs text-[var(--danger)]">{b.error}</span>}
        {b.methods.length === 0 && <span className="text-xs text-muted">{t("bench.empty_methods")}</span>}
      </div>
    </div>
  );
}

function Stepper({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="text-xs text-muted">
      <div className="mb-1">{label}</div>
      <span className="inline-flex items-center rounded-md border">
        <button className="px-2 py-1 hover:bg-[var(--surface-2)]" onClick={() => onChange(Math.max(min, value - step))}>
          −
        </button>
        <span className="tnum w-10 text-center text-text">{value}</span>
        <button className="px-2 py-1 hover:bg-[var(--surface-2)]" onClick={() => onChange(Math.min(max, value + step))}>
          +
        </button>
      </span>
    </label>
  );
}
