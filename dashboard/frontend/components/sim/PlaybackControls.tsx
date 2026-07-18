"use client";
import { useLang } from "@/components/providers/LanguageProvider";

interface Props {
  total: number;
  index: number;
  playing: boolean;
  speed: number;
  onSeek: (i: number) => void;
  onTogglePlay: () => void;
  onSpeed: (s: number) => void;
}

const SPEEDS = [0.5, 1, 2, 4, 8];

export function PlaybackControls({
  total,
  index,
  playing,
  speed,
  onSeek,
  onTogglePlay,
  onSpeed,
}: Props) {
  const { t } = useLang();
  const disabled = total <= 1;
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-card border bg-[var(--surface)] px-3 py-2">
      <button
        onClick={onTogglePlay}
        disabled={disabled}
        className="grid h-9 w-9 place-items-center rounded-md bg-[var(--accent)] text-white disabled:opacity-40"
        aria-label={playing ? t("playback.pause") : t("playback.play")}
      >
        {playing ? "⏸" : "▶"}
      </button>
      <button
        onClick={() => onSeek(Math.max(0, index - 1))}
        disabled={disabled}
        className="grid h-9 w-9 place-items-center rounded-md border disabled:opacity-40"
        aria-label="Step back"
      >
        ⏮
      </button>
      <button
        onClick={() => onSeek(Math.min(total - 1, index + 1))}
        disabled={disabled}
        className="grid h-9 w-9 place-items-center rounded-md border disabled:opacity-40"
        aria-label="Step forward"
      >
        ⏭
      </button>

      <input
        dir="ltr"
        type="range"
        min={0}
        max={Math.max(0, total - 1)}
        value={index}
        onChange={(e) => onSeek(Number(e.target.value))}
        disabled={disabled}
        className="h-1.5 flex-1 min-w-[140px] accent-[var(--accent)]"
        aria-label="Timeline"
      />
      <span className="tnum text-xs text-muted">
        {total ? index + 1 : 0}/{total}
      </span>

      <div className="flex items-center gap-1">
        <span className="text-xs text-muted">{t("playback.speed")}</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => onSpeed(s)}
            className={`rounded-md px-2 py-1 text-xs ${
              speed === s ? "bg-[var(--surface-2)] font-semibold text-accent" : "text-muted"
            }`}
          >
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
