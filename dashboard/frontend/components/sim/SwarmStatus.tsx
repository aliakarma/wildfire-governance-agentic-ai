"use client";
import { useMemo } from "react";
import { useLang } from "@/components/providers/LanguageProvider";
import { useTheme } from "@/components/providers/ThemeProvider";
import { b64ToU8, roleColor } from "@/lib/colormap";
import type { Frame, SwarmPhase, UavRole } from "@/lib/types";

const PHASE_ROLE: Record<SwarmPhase, UavRole> = {
  search: "scout",
  verify: "verifier",
  track: "responder",
  static: "static",
};

const ROLE_ORDER: UavRole[] = ["scout", "verifier", "responder", "static"];

/** Compact narrator for the cooperative swarm: current phase, role mix, and how
 *  far the fire has spread — so the coordination story reads without guessing
 *  what the colours on the canvas mean. */
export function SwarmStatus({ frame }: { frame: Frame | null }) {
  const { t } = useLang();
  const { theme } = useTheme();

  const burning = useMemo(() => {
    if (!frame) return 0;
    const fire = b64ToU8(frame.fire_b64);
    let n = 0;
    for (let i = 0; i < fire.length; i++) if (fire[i] > 0) n++;
    return n;
  }, [frame]);

  const phase: SwarmPhase = frame?.phase ?? "search";
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const u of frame?.uavs ?? []) {
      const r = u.role ?? "scout";
      c[r] = (c[r] ?? 0) + 1;
    }
    return c;
  }, [frame]);

  const phaseColor = roleColor(PHASE_ROLE[phase], theme);
  const fires = frame?.fires?.length ?? 0;

  return (
    <div className="card flex flex-col gap-3 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold">{t("swarm.title")}</span>
        <span
          className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold"
          style={{ borderColor: phaseColor, color: phaseColor }}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${phase === "verify" ? "pulse-dot" : ""}`}
            style={{ background: phaseColor }}
          />
          {t(`swarm.phase.${phase}`)}
        </span>
      </div>

      <p className="text-xs text-muted">{t(`swarm.hint.${phase}`)}</p>

      <div className="flex flex-wrap gap-1.5">
        {ROLE_ORDER.filter((r) => (counts[r] ?? 0) > 0).map((r) => (
          <span
            key={r}
            className="inline-flex items-center gap-1.5 rounded-md border bg-[var(--surface-2)]/40 px-2 py-1 text-[11px]"
          >
            <span className="h-2 w-2 rounded-full" style={{ background: roleColor(r, theme) }} />
            <span className="text-muted">{t(`swarm.role.${r}`)}</span>
            <span className="tnum font-semibold">{counts[r]}</span>
          </span>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-lg border bg-[var(--surface-2)]/40 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted">{t("swarm.fires")}</div>
          <div className="tnum text-base font-semibold">{fires}</div>
        </div>
        <div className="rounded-lg border bg-[var(--surface-2)]/40 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-muted">{t("swarm.burning")}</div>
          <div className="tnum text-base font-semibold" style={{ color: "var(--warn)" }}>
            {burning.toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
}
