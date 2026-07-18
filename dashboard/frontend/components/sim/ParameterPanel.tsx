"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import type { SimParams } from "@/lib/types";
import type { StreamStatus } from "@/lib/useEpisodeStream";

const METHODS: Array<{ id: string; label: string; enf: string }> = [
  { id: "ppo_gomdp", label: "PPO-GOMDP", enf: "crypto" },
  { id: "greedy_gomdp", label: "Greedy-GOMDP", enf: "crypto" },
  { id: "central_sig", label: "Central+Sig", enf: "signature" },
  { id: "ppo_cmdp", label: "PPO-CMDP", enf: "lagrangian" },
  { id: "adaptive_ai", label: "Adaptive AI", enf: "none" },
  { id: "static", label: "Static", enf: "none" },
];

const ATTACKS = ["none", "spoofing", "spoofing_strategic", "injection", "byzantine"];

interface Props {
  params: SimParams;
  status: StreamStatus;
  onChange: (patch: Partial<SimParams>) => void;
  onRun: () => void;
  onStop: () => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-muted">{label}</span>
        <span className="tnum font-medium">
          {value}
          {unit ? ` ${unit}` : ""}
        </span>
      </div>
      <input
        dir="ltr"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full accent-[var(--accent)]"
      />
    </label>
  );
}

export function ParameterPanel({ params, status, onChange, onRun, onStop }: Props) {
  const { t } = useLang();
  const running = status === "running" || status === "connecting";

  return (
    <div className="flex flex-col gap-4">
      <div className="card p-4">
        <h2 className="mb-3 text-sm font-semibold">{t("panel.method")}</h2>
        <div className="grid grid-cols-2 gap-2">
          {METHODS.map((m) => {
            const active = params.method === m.id;
            return (
              <button
                key={m.id}
                onClick={() => onChange({ method: m.id })}
                className={`rounded-lg border px-2 py-2 text-start text-xs transition ${
                  active
                    ? "border-[var(--accent)] bg-[var(--surface-2)]"
                    : "hover:bg-[var(--surface-2)]/60"
                }`}
              >
                <div className="font-semibold">{m.label}</div>
                <div className="text-[10px] text-muted">{m.enf}</div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="card p-4">
        <h2 className="mb-3 text-sm font-semibold">{t("panel.parameters")}</h2>
        <div className="space-y-3">
          <Slider label={t("param.grid_size")} value={params.grid_size} min={20} max={200} step={10} unit="cells" onChange={(v) => onChange({ grid_size: v })} />
          <Slider label={t("param.n_uavs")} value={params.n_uavs} min={1} max={60} step={1} onChange={(v) => onChange({ n_uavs: v })} />
          <Slider label={t("param.n_sectors")} value={params.n_sectors} min={4} max={100} step={1} onChange={(v) => onChange({ n_sectors: v })} />
          <Slider label={t("param.n_timesteps")} value={params.n_timesteps} min={100} max={3000} step={100} onChange={(v) => onChange({ n_timesteps: v })} />
          <Slider label={t("param.tau")} value={params.tau} min={0.5} max={0.99} step={0.01} onChange={(v) => onChange({ tau: v })} />
          <Slider label={t("param.seed")} value={params.seed} min={0} max={9999} step={1} onChange={(v) => onChange({ seed: v })} />
        </div>
      </div>

      <div className="card p-4">
        <h2 className="mb-3 text-sm font-semibold">{t("panel.adversarial")}</h2>
        <label className="mb-3 block">
          <span className="mb-1 block text-xs text-muted">{t("param.attack")}</span>
          <select
            value={params.attack_type}
            onChange={(e) => onChange({ attack_type: e.target.value as SimParams["attack_type"] })}
            className="w-full rounded-md border bg-[var(--surface)] px-2 py-1.5 text-sm"
          >
            {ATTACKS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>
        {(params.attack_type === "spoofing" || params.attack_type === "spoofing_strategic") && (
          <Slider label={t("param.p_spoof")} value={params.p_spoof} min={0} max={0.5} step={0.01} onChange={(v) => onChange({ p_spoof: v })} />
        )}
        {params.attack_type === "byzantine" && (
          <Slider label={t("param.n_byzantine")} value={params.n_byzantine} min={0} max={3} step={1} onChange={(v) => onChange({ n_byzantine: v })} />
        )}
        <div className="mt-3 space-y-3">
          <Slider label={t("param.p_drop")} value={params.p_drop} min={0} max={0.3} step={0.01} onChange={(v) => onChange({ p_drop: v })} />
          <Slider label={t("param.sensor_fail")} value={params.sensor_failure_rate} min={0} max={0.4} step={0.05} onChange={(v) => onChange({ sensor_failure_rate: v })} />
        </div>
      </div>

      <button
        onClick={running ? onStop : onRun}
        className={`w-full rounded-lg px-4 py-3 font-semibold text-white ${
          running ? "bg-[var(--danger)]" : "bg-[var(--accent)]"
        }`}
      >
        {running ? t("cta.stop") : t("cta.run")}
      </button>
    </div>
  );
}
