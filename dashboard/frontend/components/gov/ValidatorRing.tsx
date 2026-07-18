"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import { useTheme } from "@/components/providers/ThemeProvider";

interface Props {
  nValidators: number;
  nByzantine: number;
  threshold: number;
}

/** SVG ring of PBFT validators; the first `nByzantine` are marked malicious. */
export function ValidatorRing({ nValidators, nByzantine, threshold }: Props) {
  const { t } = useLang();
  const { theme } = useTheme();
  const size = 220;
  const cx = size / 2;
  const cy = size / 2;
  const radius = 78;
  const safe = nByzantine <= threshold;

  const honest = theme === "dark" ? "#35D0A5" : "#0F8A6A";
  const byz = theme === "dark" ? "#FF5C72" : "#C7263E";
  const line = theme === "dark" ? "#263041" : "#DAD6CE";

  const nodes = Array.from({ length: nValidators }, (_, i) => {
    const angle = (i / nValidators) * Math.PI * 2 - Math.PI / 2;
    return {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      byzantine: i < nByzantine,
      id: i,
    };
  });

  return (
    <div className="card p-4">
      <div className="mb-3 text-sm font-semibold">{t("gov.validators")}</div>
      <div dir="ltr" className="flex flex-col items-center">
        <svg width={size} height={size} role="img" aria-label={`${nValidators} validators, ${nByzantine} Byzantine`}>
          {nodes.map((n) => (
            <line key={`l-${n.id}`} x1={cx} y1={cy} x2={n.x} y2={n.y} stroke={line} strokeWidth={1} />
          ))}
          <circle cx={cx} cy={cy} r={16} fill="none" stroke={line} strokeWidth={1.5} />
          <text x={cx} y={cy + 4} textAnchor="middle" fontSize={11} fill="var(--text-muted)">PBFT</text>
          {nodes.map((n) => (
            <g key={`n-${n.id}`}>
              <circle cx={n.x} cy={n.y} r={13} fill={n.byzantine ? byz : honest} />
              <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize={11} fill="#fff" fontWeight={600}>
                {n.byzantine ? "✕" : n.id + 1}
              </text>
            </g>
          ))}
        </svg>
        <div className="mt-3 flex items-center gap-4 text-xs">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: honest }} /> {t("gov.validator.honest")}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: byz }} /> {t("gov.validator.byzantine")}
          </span>
        </div>
        <div
          className="mt-3 w-full rounded-lg px-3 py-2 text-center text-xs font-semibold"
          style={{
            background: safe ? "color-mix(in srgb, var(--ok) 14%, transparent)" : "color-mix(in srgb, var(--danger) 14%, transparent)",
            color: safe ? "var(--ok)" : "var(--danger)",
          }}
        >
          f = {nByzantine} / k = {nValidators} · {safe ? t("gov.bft.safe") : t("gov.bft.unsafe")}
        </div>
      </div>
    </div>
  );
}
