"use client";
import { useEffect, useState } from "react";
import { useLang } from "@/components/providers/LanguageProvider";
import { useTheme } from "@/components/providers/ThemeProvider";
import { fetchBreach } from "@/lib/api";
import type { BreachResponse } from "@/lib/types";

interface Props {
  nValidators: number;
  nByzantine: number;
}

/** Line chart of Theorem-2 breach probability: GOMDP vs centralized over p_c. */
export function BreachMeter({ nValidators, nByzantine }: Props) {
  const { t } = useLang();
  const { theme } = useTheme();
  const [data, setData] = useState<BreachResponse | null>(null);

  useEffect(() => {
    let alive = true;
    fetchBreach(nValidators, nByzantine)
      .then((d) => alive && setData(d))
      .catch(() => alive && setData(null));
    return () => {
      alive = false;
    };
  }, [nValidators, nByzantine]);

  const W = 300;
  const H = 160;
  const pad = { l: 34, r: 12, t: 12, b: 26 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;

  const gomdpC = theme === "dark" ? "#35D0A5" : "#0F8A6A";
  const centralC = theme === "dark" ? "#FF5C72" : "#C7263E";
  const axis = theme === "dark" ? "#263041" : "#DAD6CE";

  const x = (pc: number) => pad.l + (pc / 0.5) * iw;
  const y = (v: number) => pad.t + (1 - v) * ih;

  const path = (key: "gomdp" | "central") =>
    (data?.points ?? [])
      .map((p, i) => `${i === 0 ? "M" : "L"}${x(p.p_c).toFixed(1)},${y(p[key]).toFixed(1)}`)
      .join(" ");

  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-semibold">{t("adv.breach")}</div>
      <p className="mb-2 text-xs text-muted">{t("adv.pc")}</p>
      <div dir="ltr">
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Breach probability chart">
          {[0, 0.25, 0.5, 0.75, 1].map((g) => (
            <g key={g}>
              <line x1={pad.l} y1={y(g)} x2={W - pad.r} y2={y(g)} stroke={axis} strokeWidth={0.5} />
              <text x={pad.l - 6} y={y(g) + 3} textAnchor="end" fontSize={9} fill="var(--text-muted)">
                {g}
              </text>
            </g>
          ))}
          {[0, 0.25, 0.5].map((g) => (
            <text key={g} x={x(g)} y={H - 8} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
              {g}
            </text>
          ))}
          {data && (
            <>
              <path d={path("central")} fill="none" stroke={centralC} strokeWidth={2} />
              <path d={path("gomdp")} fill="none" stroke={gomdpC} strokeWidth={2} />
            </>
          )}
        </svg>
      </div>
      <div className="mt-2 flex items-center gap-4 text-xs">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded" style={{ background: gomdpC }} /> {t("adv.breach.gomdp")}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded" style={{ background: centralC }} /> {t("adv.breach.central")}
        </span>
      </div>
      {data && !data.bft_safe && (
        <p className="mt-2 text-xs text-[var(--danger)]">⚠ f = {data.max_byzantine} ≥ ⌊k/3⌋ = {data.threshold}</p>
      )}
    </div>
  );
}
