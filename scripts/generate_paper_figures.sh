#!/usr/bin/env bash
# =============================================================================
# Regenerate all paper figures from latest live run outputs in results/runs/.
# Figures are saved to results/figures/.
#
# Usage:
#   bash scripts/generate_paper_figures.sh
# =============================================================================

set -euo pipefail
mkdir -p results/figures

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "[FAIL] Python interpreter not found in PATH (tried python3, python)."
        exit 1
    fi
fi

echo "Generating paper figures from latest live run outputs..."

"$PYTHON_BIN" - << 'EOF'
from pathlib import Path
import pandas as pd
import sys
from wildfire_governance.utils.visualisation import (
    plot_latency_vs_uavs,
    plot_false_alert_bar,
    plot_tradeoff_frontier,
)

RUNS = Path("results/runs")
OUT   = Path("results/figures")
OUT.mkdir(exist_ok=True)

if not RUNS.exists():
    print("[FAIL] results/runs/ does not exist. Run experiments first.")
    sys.exit(1)

run_dirs = [p for p in RUNS.iterdir() if p.is_dir()]
if not run_dirs:
    print("[FAIL] No run directories found under results/runs/.")
    sys.exit(1)

latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
print(f"Using latest run: {latest_run}")

required_files = [
    "fig3_latency_data.csv",
    "fig4_false_alerts_data.csv",
    "fig5_tradeoff_data.csv",
]
missing = [name for name in required_files if not (latest_run / name).exists()]
if missing:
    print("[FAIL] Latest run is missing required figure inputs:")
    for name in missing:
        print(f"  - {latest_run / name}")
    print("Run full reproduction to generate all figure inputs.")
    sys.exit(1)

# Figure 3 â€” Detection latency vs. fleet size
fig3 = latest_run / "fig3_latency_data.csv"
df = pd.read_csv(fig3)
plot_latency_vs_uavs(df, output_path=OUT / "fig3_latency.pdf")
print(f"  Saved: {OUT}/fig3_latency.pdf")

# Figure 4 â€” False alert rate
fig4 = latest_run / "fig4_false_alerts_data.csv"
df = pd.read_csv(fig4)[["config", "fp_mean", "fp_std"]].dropna()
plot_false_alert_bar(df, output_path=OUT / "fig4_false_alerts.pdf")
print(f"  Saved: {OUT}/fig4_false_alerts.pdf")

# Figure 5 â€” Tradeoff frontier (N=40)
fig5 = latest_run / "fig5_tradeoff_data.csv"
df5 = pd.read_csv(fig5)
plot_tradeoff_frontier(df5, output_path=OUT / "fig5_tradeoff.pdf")
print(f"  Saved: {OUT}/fig5_tradeoff.pdf")

print("\nAll figures regenerated in results/figures/")
EOF
