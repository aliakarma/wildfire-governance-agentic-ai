"use client";
import { useEffect, useState } from "react";
import { useLang } from "@/components/providers/LanguageProvider";
import { useSim } from "@/components/providers/SimulationProvider";
import { GridCanvas } from "@/components/sim/GridCanvas";
import { MetricHUD } from "@/components/sim/MetricHUD";
import { PlaybackControls } from "@/components/sim/PlaybackControls";
import { SwarmStatus } from "@/components/sim/SwarmStatus";
import { fetchPaperResults } from "@/lib/api";
import type { PaperResponse } from "@/lib/types";

const REGIONS = [
  { id: "california", key: "viirs.california", match: "California", coords: "37.5°N · 122.0°W", seed: 20 },
  { id: "mediterranean", key: "viirs.mediterranean", match: "Mediterranean", coords: "38.0°N · 23.7°E", seed: 21 },
  { id: "nsw", key: "viirs.nsw", match: "NSW", coords: "33.9°S · 151.2°E", seed: 19 },
];

const REGION_PARAMS = { grid_size: 60, n_uavs: 20, n_sectors: 25, n_timesteps: 400, tau: 0.72, method: "greedy_gomdp", attack_type: "none" as const };

export function VIIRSScreen() {
  const { t } = useLang();
  const s = useSim();
  const [regionId, setRegionId] = useState("california");
  const [paper, setPaper] = useState<PaperResponse | null>(null);

  useEffect(() => {
    fetchPaperResults("table4_realworld_viirs").then(setPaper).catch(() => {});
  }, []);

  // Seed the shared params for the initial region on mount.
  useEffect(() => {
    s.setParam({ ...REGION_PARAMS, seed: 20 });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const region = REGIONS.find((r) => r.id === regionId)!;
  const selectRegion = (r: (typeof REGIONS)[number]) => {
    setRegionId(r.id);
    s.setParam({ ...REGION_PARAMS, seed: r.seed });
  };

  const rows = (paper?.rows ?? []).filter((row) => (row.event ?? "").startsWith(region.match));
  const running = s.status === "running" || s.status === "connecting";

  return (
    <div className="flex flex-col gap-4">
      <div className="card p-4">
        <h1 className="text-sm font-semibold">{t("viirs.title")}</h1>
        <p className="mt-1 text-xs text-muted">{t("viirs.desc")}</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {REGIONS.map((r) => {
            const active = r.id === regionId;
            return (
              <button
                key={r.id}
                onClick={() => selectRegion(r)}
                className={`rounded-md border px-3 py-1.5 text-xs font-medium transition ${
                  active ? "border-[var(--accent)] bg-[var(--surface-2)] text-accent" : "text-muted hover:bg-[var(--surface-2)]/60"
                }`}
              >
                {t(r.key)}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <section className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold">{t(region.key)}</span>
            <span dir="ltr" className="tnum rounded-full border px-2.5 py-1 text-[11px] text-muted">{region.coords}</span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-[var(--ok)] px-2.5 py-1 text-[11px] font-semibold text-[var(--ok)]">
              <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-[var(--ok)]" />
              {t("note.livebadge")}
            </span>
            <button
              onClick={running ? s.stop : s.run}
              className={`ms-auto rounded-lg px-4 py-2 text-sm font-semibold text-white ${running ? "bg-[var(--danger)]" : "bg-[var(--accent)]"}`}
            >
              {running ? t("cta.stop") : t("viirs.run")}
            </button>
          </div>

          <MetricHUD frame={s.currentFrame} />
          <GridCanvas
            frame={s.currentFrame}
            showFire
            showUavs
            showSectors={false}
            showComms={s.layers.comms}
            nSectors={s.params.n_sectors}
            trail={s.frames.slice(Math.max(0, s.index - 5), s.index).map((f) => f.uavs)}
          />
          <PlaybackControls
            total={s.frames.length}
            index={s.index}
            playing={s.playing}
            speed={s.speed}
            onSeek={s.seek}
            onTogglePlay={s.togglePlay}
            onSpeed={s.setSpeed}
          />
          <p className="text-[11px] text-muted">{t("viirs.source")}</p>
        </section>

        <aside className="flex flex-col gap-4">
        <SwarmStatus frame={s.currentFrame} />
        <div className="card flex flex-col p-4">
          <div className="mb-1 text-sm font-semibold">{t("viirs.ref")}</div>
          <span className="mb-3 inline-flex w-fit items-center gap-1.5 rounded-full border border-[var(--warn)] px-2 py-0.5 text-[10px] font-semibold text-[var(--warn)]">
            {t("bench.paper_badge")}
          </span>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted">
                  <th className="px-2 py-2 text-start font-medium">{t("bench.methods")}</th>
                  <th className="px-2 py-2 text-end font-medium">L_d</th>
                  <th className="px-2 py-2 text-end font-medium">F_p</th>
                  <th className="px-2 py-2 text-end font-medium">Gov.</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i} className="border-b">
                    <td className="px-2 py-2 font-medium">{r.method}</td>
                    <td className="tnum px-2 py-2 text-end">{r.ld_mean}</td>
                    <td className="tnum px-2 py-2 text-end">{r.fp_mean}%</td>
                    <td className="tnum px-2 py-2 text-end" style={{ color: parseFloat(r.gov_compliance_pct) >= 99.9 ? "var(--ok)" : parseFloat(r.gov_compliance_pct) === 0 ? "var(--danger)" : "var(--warn)" }}>
                      {r.gov_compliance_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        </aside>
      </div>
    </div>
  );
}
