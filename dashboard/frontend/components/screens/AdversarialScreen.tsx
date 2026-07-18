"use client";
import { AttackConsole } from "@/components/adv/AttackConsole";
import { BreachMeter } from "@/components/adv/BreachMeter";
import { SafetyVerdict } from "@/components/adv/SafetyVerdict";
import { useSim } from "@/components/providers/SimulationProvider";
import { EventFeed } from "@/components/sim/EventFeed";
import { GridCanvas } from "@/components/sim/GridCanvas";
import { MetricHUD } from "@/components/sim/MetricHUD";
import { PlaybackControls } from "@/components/sim/PlaybackControls";
import { SwarmStatus } from "@/components/sim/SwarmStatus";

export function AdversarialScreen() {
  const s = useSim();
  const nVal = (s.meta?.n_validators as number) ?? 7;
  const threshold = (s.meta?.byzantine_threshold as number) ?? Math.floor((nVal - 1) / 3);
  // The breach meter shows the design's guarantee at its BFT tolerance f=⌊(k-1)/3⌋
  // (the paper's operating point). Only a Byzantine attack raises f past that,
  // which the meter then reports as consensus-compromised.
  const nByz =
    s.params.attack_type === "byzantine"
      ? ((s.meta?.n_byzantine as number) ?? s.params.n_byzantine)
      : threshold;

  return (
    <div className="flex flex-col gap-4">
      <AttackConsole />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <section className="flex flex-col gap-3">
          <MetricHUD frame={s.currentFrame} />
          <GridCanvas
            frame={s.currentFrame}
            showFire={s.layers.fire}
            showUavs={s.layers.uavs}
            showSectors={s.layers.sectors}
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
          {s.error && (
            <div className="rounded-card border border-[var(--danger)] bg-[var(--danger)]/10 px-3 py-2 text-sm text-[var(--danger)]">
              {s.error}
            </div>
          )}
        </section>
        <aside className="flex flex-col gap-4">
          <SwarmStatus frame={s.currentFrame} />
          <SafetyVerdict />
          <BreachMeter nValidators={nVal} nByzantine={nByz} />
          <div className="h-[280px]">
            <EventFeed frames={s.frames} />
          </div>
        </aside>
      </div>
    </div>
  );
}
