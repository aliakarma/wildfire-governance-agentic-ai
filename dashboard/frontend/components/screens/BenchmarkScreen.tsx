"use client";
import { useEffect, useMemo } from "react";
import { BenchmarkControls } from "@/components/bench/BenchmarkControls";
import { ComparisonChart, type Series } from "@/components/bench/ComparisonChart";
import { ReproDiff } from "@/components/bench/ReproDiff";
import { TradeoffFrontier } from "@/components/bench/TradeoffFrontier";
import { useBench } from "@/components/providers/BenchmarkProvider";
import { useLang } from "@/components/providers/LanguageProvider";
import { useTheme } from "@/components/providers/ThemeProvider";

const COLORS: Record<string, { light: string; dark: string }> = {
  "PPO-GOMDP": { light: "#E4572E", dark: "#FF6B3D" },
  "Greedy-GOMDP": { light: "#C77D0A", dark: "#F2B455" },
  "Central+Sig": { light: "#2D6BB0", dark: "#6FB1FF" },
  "Shield-PPO": { light: "#2D6BB0", dark: "#6FB1FF" },
  "SafeLayer": { light: "#0F8A6A", dark: "#35D0A5" },
  "PPO-CMDP": { light: "#7A5AF8", dark: "#A78BFA" },
  "WCSAC": { light: "#0F8A6A", dark: "#35D0A5" },
  "Adaptive AI": { light: "#8A8F98", dark: "#9AA6B8" },
  "Static": { light: "#5B616E", dark: "#6B7280" },
};

interface NormRow {
  label: string;
  ld_mean: number;
  ld_ci95: number;
  fp_mean: number;
  fp_ci95: number;
  compliance_pct: number;
}

export function BenchmarkScreen() {
  const { t } = useLang();
  const { theme } = useTheme();
  const b = useBench();

  useEffect(() => {
    b.loadPaper();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const color = (label: string) => {
    const c = COLORS[label] ?? { light: "#8A8F98", dark: "#9AA6B8" };
    return theme === "dark" ? c.dark : c.light;
  };

  const rows: NormRow[] = useMemo(() => {
    if (b.source === "live") {
      return (b.live?.rows ?? []).map((r) => ({
        label: r.label,
        ld_mean: r.ld_mean,
        ld_ci95: r.ld_ci95,
        fp_mean: r.fp_mean,
        fp_ci95: r.fp_ci95,
        compliance_pct: r.compliance_pct,
      }));
    }
    return (b.paper?.rows ?? []).map((r) => ({
      label: r.method,
      ld_mean: parseFloat(r.ld_mean),
      ld_ci95: 0,
      fp_mean: parseFloat(r.fp_mean),
      fp_ci95: 0,
      compliance_pct: parseFloat(r.compliance_pct),
    }));
  }, [b.source, b.live, b.paper]);

  const ser = (key: "ld_mean" | "fp_mean" | "compliance_pct", ciKey?: "ld_ci95" | "fp_ci95"): Series[] =>
    rows.map((r) => ({ label: r.label, value: r[key], ci: ciKey ? r[ciKey] : undefined, color: color(r.label) }));

  return (
    <div className="flex flex-col gap-4">
      <BenchmarkControls />

      <div
        className="inline-flex w-fit items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold"
        style={{
          borderColor: b.source === "live" ? "var(--ok)" : "var(--warn)",
          color: b.source === "live" ? "var(--ok)" : "var(--warn)",
        }}
      >
        <span
          className={`h-1.5 w-1.5 rounded-full ${b.source === "live" ? "pulse-dot" : ""}`}
          style={{ background: b.source === "live" ? "var(--ok)" : "var(--warn)" }}
        />
        {t(b.source === "live" ? "bench.live_badge" : "bench.paper_badge")}
      </div>

      {rows.length === 0 ? (
        <div className="card p-8 text-center text-sm text-muted">{t("bench.no_live")}</div>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <ComparisonChart title={t("bench.metric.ld")} series={ser("ld_mean", "ld_ci95")} unit="steps" />
          <ComparisonChart title={t("bench.metric.fp")} series={ser("fp_mean", "fp_ci95")} unit="%" />
          <ComparisonChart title={t("bench.metric.comp")} series={ser("compliance_pct")} unit="%" max={100} />
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[380px_minmax(0,1fr)]">
        <TradeoffFrontier />
        <ReproDiff live={b.live} paper={b.paper} />
      </div>
    </div>
  );
}
