"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import { useSim } from "@/components/providers/SimulationProvider";
import { EventFeed } from "@/components/sim/EventFeed";
import { GridCanvas } from "@/components/sim/GridCanvas";
import { MetricHUD } from "@/components/sim/MetricHUD";
import { ParameterPanel } from "@/components/sim/ParameterPanel";
import { PlaybackControls } from "@/components/sim/PlaybackControls";
import { SwarmStatus } from "@/components/sim/SwarmStatus";

function LayerToggle({ label, on, onClick }: { label: string; on: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md border px-2.5 py-1.5 text-xs ${on ? "border-[var(--accent)] text-accent" : "text-muted"}`}
      aria-pressed={on}
    >
      {label}
    </button>
  );
}

function SummaryStat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded-lg border bg-[var(--surface-2)]/40 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className="tnum text-base font-semibold" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

export function LiveScreen() {
  const { t } = useLang();
  const s = useSim();
  const running = s.status === "running" || s.status === "connecting";

  const statusLabel =
    ({ idle: t("status.idle"), connecting: t("status.connecting"), running: t("status.running"), done: t("status.done"), error: t("status.error") } as Record<string, string>)[s.status] ?? s.status;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[300px_minmax(0,1fr)_320px]">
      <aside>
        <ParameterPanel
          params={s.params}
          status={s.status}
          onChange={s.setParam}
          onRun={s.run}
          onStop={s.stop}
        />
      </aside>

      <section className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-[var(--ok)] px-2.5 py-1 text-[11px] font-semibold text-[var(--ok)]">
            <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-[var(--ok)]" />
            {t("note.livebadge")}
          </span>
          <span className="rounded-full border px-2.5 py-1 text-[11px] text-muted">{statusLabel}</span>
          <div className="ms-auto flex items-center gap-2">
            <LayerToggle label={t("layers.fire")} on={s.layers.fire} onClick={() => s.toggleLayer("fire")} />
            <LayerToggle label={t("layers.uavs")} on={s.layers.uavs} onClick={() => s.toggleLayer("uavs")} />
            <LayerToggle label={t("layers.comms")} on={s.layers.comms} onClick={() => s.toggleLayer("comms")} />
            <LayerToggle label={t("layers.sectors")} on={s.layers.sectors} onClick={() => s.toggleLayer("sectors")} />
            <button
              onClick={s.share}
              className="rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-[var(--surface-2)]"
            >
              {s.shared ? t("cta.shared") : t("cta.share")}
            </button>
            <button
              onClick={s.exportGif}
              disabled={s.frames.length === 0 || s.exporting}
              className="rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-[var(--surface-2)] disabled:opacity-40"
            >
              {s.exporting ? "…" : t("cta.export_gif")}
            </button>
          </div>
        </div>

        <MetricHUD frame={s.currentFrame} />

        <GridCanvas
          frame={s.currentFrame}
          showFire={s.layers.fire}
          showUavs={s.layers.uavs}
          showSectors={s.layers.sectors}
          showComms={s.layers.comms}
          nSectors={s.params.n_sectors}
          trail={s.frames.slice(Math.max(0, s.index - 14), s.index).map((f) => f.uavs)}
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

        {s.error && (
          <div className="rounded-card border border-[var(--danger)] bg-[var(--danger)]/10 px-3 py-2 text-sm text-[var(--danger)]">
            {s.error}
          </div>
        )}

        {s.summary && !running && (
          <div className="card p-4 text-sm">
            <div className="mb-2 font-semibold">{t("summary.title")}</div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <SummaryStat label={t("metric.ld")} value={`${s.summary.ld ?? "—"}`} />
              <SummaryStat label={t("metric.fp")} value={`${s.summary.fp_pct}%`} />
              <SummaryStat
                label={t("metric.compliance")}
                value={`${s.summary.compliance}%`}
                color={Number(s.summary.compliance) >= 99.999 ? "var(--ok)" : "var(--danger)"}
              />
              <SummaryStat label={t("metric.injections")} value={`${s.summary.n_injections_blocked}`} />
            </div>
            {Number(s.summary.compliance) >= 99.999 && (
              <p className="mt-3 text-xs text-muted">✓ {t("note.enforced")}</p>
            )}
            {s.meta?.policy_effective === "ppo_pretrained" && (
              <p className="mt-1 text-xs text-[var(--info)]">◆ {t("note.ppo_pretrained")}</p>
            )}
            {s.meta?.note ? <p className="mt-1 text-xs text-[var(--warn)]">⚠ {String(s.meta.note)}</p> : null}
          </div>
        )}
      </section>

      <aside className="flex flex-col gap-3">
        <SwarmStatus frame={s.currentFrame} />
        <div className="h-[420px] lg:min-h-[420px] lg:flex-1">
          <EventFeed frames={s.frames} index={s.index} />
        </div>
      </aside>
    </div>
  );
}
