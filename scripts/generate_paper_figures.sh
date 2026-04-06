#!/usr/bin/env bash
# =============================================================================
# Regenerate all paper figures from pre-committed results/paper/ CSVs.
# Figures are saved to results/figures/.
#
# Usage:
#   bash scripts/generate_paper_figures.sh
# =============================================================================

set -euo pipefail
mkdir -p results/figures

echo "Generating paper figures from results/paper/ CSVs..."

python - << 'EOF'
from pathlib import Path
import pandas as pd
from wildfire_governance.utils.visualisation import (
    plot_latency_vs_uavs,
    plot_false_alert_bar,
    plot_tradeoff_frontier,
)

PAPER = Path("results/paper")
OUT   = Path("results/figures")
OUT.mkdir(exist_ok=True)

# Figure 3 — Detection latency vs. fleet size
fig3 = PAPER / "fig3_latency_data.csv"
if fig3.exists():
    df = pd.read_csv(fig3)
    plot_latency_vs_uavs(df, output_path=OUT / "fig3_latency.pdf")
    print(f"  Saved: {OUT}/fig3_latency.pdf")

# Figure 4 — False alert rate
fig4 = PAPER / "table3_main_comparison.csv"
if fig4.exists():
    df = pd.read_csv(fig4)[["config", "fp_mean", "fp_std"]].dropna()
    plot_false_alert_bar(df, output_path=OUT / "fig4_false_alerts.pdf")
    print(f"  Saved: {OUT}/fig4_false_alerts.pdf")

# Figure 5 — Tradeoff frontier (N=40)
fig5 = PAPER / "fig5_tradeoff_data.csv" if (PAPER / "fig5_tradeoff_data.csv").exists() else None
if fig5 is None:
    # Fall back to table3
    df5 = pd.read_csv(PAPER / "table3_main_comparison.csv").dropna(subset=["ld_mean", "fp_mean"])
else:
    df5 = pd.read_csv(fig5)
plot_tradeoff_frontier(df5, output_path=OUT / "fig5_tradeoff.pdf")
print(f"  Saved: {OUT}/fig5_tradeoff.pdf")

print("\nAll figures regenerated in results/figures/")
EOF
