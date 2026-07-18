"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useLang } from "@/components/providers/LanguageProvider";
import { GridCanvas } from "@/components/sim/GridCanvas";
import { MetricHUD } from "@/components/sim/MetricHUD";
import { PlaybackControls } from "@/components/sim/PlaybackControls";
import { DEFAULT_PARAMS } from "@/lib/api";
import type { SimParams, UavPoint } from "@/lib/types";
import { useEpisodeStream } from "@/lib/useEpisodeStream";

const BASE_FPS = 14;
const METHODS: Array<{ id: string; label: string; enf: string }> = [
  { id: "greedy_gomdp", label: "Greedy-GOMDP", enf: "crypto" },
  { id: "ppo_gomdp", label: "PPO-GOMDP", enf: "crypto" },
  { id: "central_sig", label: "Central+Sig", enf: "signature" },
  { id: "ppo_cmdp", label: "PPO-CMDP", enf: "lagrangian" },
  { id: "adaptive_ai", label: "Adaptive AI", enf: "none" },
  { id: "static", label: "Static", enf: "none" },
];

const CONFIG = { grid_size: 60, n_uavs: 16, n_timesteps: 400, tau: 0.72 };

function MethodSelect({ value, onChange, label }: { value: string; onChange: (v: string) => void; label: string }) {
  return (
    <label className="text-xs text-muted">
      <span className="mb-1 block">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border bg-[var(--surface)] px-2 py-1.5 text-sm text-text"
      >
        {METHODS.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function Panel({
  title,
  enf,
  frame,
  trail,
}: {
  title: string;
  enf: string;
  frame: Parameters<typeof MetricHUD>[0]["frame"];
  trail: UavPoint[][];
}) {
  const governed = enf === "crypto" || enf === "signature";
  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold">{title}</span>
        <span
          className="rounded-full px-2 py-0.5 text-[10px] font-semibold"
          style={{
            background: governed ? "color-mix(in srgb, var(--ok) 16%, transparent)" : "color-mix(in srgb, var(--warn) 16%, transparent)",
            color: governed ? "var(--ok)" : "var(--warn)",
          }}
        >
          {enf}
        </span>
      </div>
      <MetricHUD frame={frame} />
      <GridCanvas frame={frame} showFire showUavs showSectors={false} nSectors={25} trail={trail} />
    </section>
  );
}

export function CompareScreen() {
  const { t } = useLang();
  const [leftMethod, setLeftMethod] = useState("greedy_gomdp");
  const [rightMethod, setRightMethod] = useState("adaptive_ai");
  const [seed, setSeed] = useState(0);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const left = useEpisodeStream();
  const right = useEpisodeStream();
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef(0);
  const accRef = useRef(0);

  const running = left.status === "running" || right.status === "running" || left.status === "connecting";
  const minLen = Math.min(left.frames.length, right.frames.length);

  const runBoth = useCallback(() => {
    setIndex(0);
    setPlaying(true);
    const base: SimParams = { ...DEFAULT_PARAMS, ...CONFIG, seed };
    left.start({ ...base, method: leftMethod });
    right.start({ ...base, method: rightMethod });
  }, [leftMethod, rightMethod, seed, left, right]);

  // Follow the live edge while either stream is running.
  useEffect(() => {
    if (running && minLen > 0) setIndex(minLen - 1);
  }, [minLen, running]);

  // Replay loop once both are complete.
  useEffect(() => {
    if (!playing || running || minLen <= 1) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }
    lastTsRef.current = 0;
    accRef.current = 0;
    const tick = (ts: number) => {
      if (!lastTsRef.current) lastTsRef.current = ts;
      const dt = (ts - lastTsRef.current) / 1000;
      lastTsRef.current = ts;
      accRef.current += dt * speed * BASE_FPS;
      if (accRef.current >= 1) {
        const adv = Math.floor(accRef.current);
        accRef.current -= adv;
        setIndex((i) => {
          const next = i + adv;
          if (next >= minLen - 1) {
            setPlaying(false);
            return minLen - 1;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, speed, running, minLen]);

  const leftFrame = left.frames[Math.min(index, left.frames.length - 1)] ?? null;
  const rightFrame = right.frames[Math.min(index, right.frames.length - 1)] ?? null;
  const leftEnf = METHODS.find((m) => m.id === leftMethod)?.enf ?? "";
  const rightEnf = METHODS.find((m) => m.id === rightMethod)?.enf ?? "";

  return (
    <div className="flex flex-col gap-4">
      <div className="card flex flex-col gap-3 p-4">
        <div>
          <h1 className="text-sm font-semibold">{t("compare.title")}</h1>
          <p className="text-xs text-muted">{t("compare.desc")}</p>
        </div>
        <div className="flex flex-wrap items-end gap-4">
          <MethodSelect label={t("compare.left")} value={leftMethod} onChange={setLeftMethod} />
          <MethodSelect label={t("compare.right")} value={rightMethod} onChange={setRightMethod} />
          <label className="text-xs text-muted">
            <span className="mb-1 block">{t("param.seed")}</span>
            <span className="inline-flex items-center rounded-md border">
              <button className="px-2 py-1.5 hover:bg-[var(--surface-2)]" onClick={() => setSeed((s) => Math.max(0, s - 1))}>
                −
              </button>
              <span className="tnum w-8 text-center text-text">{seed}</span>
              <button className="px-2 py-1.5 hover:bg-[var(--surface-2)]" onClick={() => setSeed((s) => s + 1)}>
                +
              </button>
            </span>
          </label>
          <button
            onClick={runBoth}
            disabled={running}
            className="rounded-lg bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
          >
            {running ? t("bench.running") : t("compare.run")}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel
          title={METHODS.find((m) => m.id === leftMethod)?.label ?? leftMethod}
          enf={leftEnf}
          frame={leftFrame}
          trail={left.frames.slice(Math.max(0, index - 5), index).map((f) => f.uavs)}
        />
        <Panel
          title={METHODS.find((m) => m.id === rightMethod)?.label ?? rightMethod}
          enf={rightEnf}
          frame={rightFrame}
          trail={right.frames.slice(Math.max(0, index - 5), index).map((f) => f.uavs)}
        />
      </div>

      <PlaybackControls
        total={minLen}
        index={index}
        playing={playing}
        speed={speed}
        onSeek={(i) => {
          setPlaying(false);
          setIndex(i);
        }}
        onTogglePlay={() => setPlaying((p) => !p)}
        onSpeed={setSpeed}
      />
    </div>
  );
}
