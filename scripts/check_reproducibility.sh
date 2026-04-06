#!/usr/bin/env bash
# =============================================================================
# Check that newly computed results match pre-committed paper values
# within 5% relative tolerance.
#
# Usage:
#   bash scripts/check_reproducibility.sh
#
# Windows (PowerShell): python scripts/check_reproducibility.py
# =============================================================================

set -euo pipefail

echo "=== Reproducibility Check ==="
echo "Tolerance: 5% relative"
echo ""

python - << 'EOF'
import sys
from pathlib import Path
import pandas as pd
import numpy as np

TOLERANCE = 0.05
PAPER_DIR = Path("results/paper")
RUNS_DIR = Path("results/runs")

def find_latest_run(filename: str) -> Path | None:
    """Find the most recently generated version of a result file."""
    candidates = sorted(RUNS_DIR.glob(f"**/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

checks = {
    "table3_main_comparison.csv": {
        "key_col": "config_name",
        "metrics": ["ld_mean", "fp_mean"],
    },
    "table2_rl_comparison.csv": {
        "key_col": "method",
        "metrics": ["ld_mean", "fp_mean"],
    },
}

all_pass = True

for filename, spec in checks.items():
    paper_path = PAPER_DIR / filename
    if not paper_path.exists():
        print(f"[SKIP] {filename} — paper CSV not found")
        continue

    run_path = find_latest_run(filename)
    if run_path is None:
        print(f"[SKIP] {filename} — no run results found (run: make reproduce first)")
        continue

    paper_df = pd.read_csv(paper_path)
    run_df = pd.read_csv(run_path)
    key = spec["key_col"]

    print(f"Checking {filename}:")
    for _, paper_row in paper_df.iterrows():
        key_val = paper_row.get(key)
        if pd.isna(key_val):
            continue
        run_rows = run_df[run_df[key] == key_val]
        if run_rows.empty:
            print(f"  [WARN] {key_val}: not found in run results")
            continue
        run_row = run_rows.iloc[0]
        for metric in spec["metrics"]:
            if metric not in paper_row.index or metric not in run_row.index:
                continue
            pv = float(paper_row[metric]) if not pd.isna(paper_row[metric]) else None
            rv = float(run_row[metric]) if not pd.isna(run_row[metric]) else None
            if pv is None or rv is None:
                continue
            denom = abs(pv) if abs(pv) > 1e-9 else 1.0
            err = abs(pv - rv) / denom
            status = "PASS" if err < TOLERANCE else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  [{status}] {key_val}/{metric}: paper={pv:.2f}, run={rv:.2f}, err={err:.1%}")
    print()

if all_pass:
    print("All checks PASSED — results reproduce within 5% tolerance.")
    sys.exit(0)
else:
    print("Some checks FAILED — results exceed 5% tolerance.")
    print("This may be due to platform-specific floating-point differences (MKL vs OpenBLAS).")
    print("Values within 10% are typically acceptable for stochastic simulations.")
    sys.exit(1)
EOF
