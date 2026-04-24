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

echo "=== Reproducibility Check ==="
echo "Tolerance: 5% relative"
echo ""

"$PYTHON_BIN" - << 'EOF'
import sys
from pathlib import Path
import pandas as pd
import numpy as np

TOLERANCE = 0.05
PAPER_DIR = Path("results/paper")
RUNS_DIR = Path("results/runs")

def get_latest_run_dir() -> Path | None:
    """Return latest run directory under results/runs/."""
    if not RUNS_DIR.exists():
        return None
    run_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def get_join_key(run_df: pd.DataFrame, paper_df: pd.DataFrame, preferred: list[str]) -> str | None:
    for key in preferred:
        if key in run_df.columns and key in paper_df.columns:
            return key
    common = [c for c in run_df.columns if c in paper_df.columns and run_df[c].dtype == object]
    return common[0] if common else None


def compare_file(run_path: Path, paper_path: Path, preferred_keys: list[str]) -> tuple[bool, int, int]:
    run_df = pd.read_csv(run_path)
    paper_df = pd.read_csv(paper_path)

    key = get_join_key(run_df, paper_df, preferred_keys)
    if key is not None:
        merged = paper_df.merge(run_df, on=key, suffixes=("_paper", "_run"), how="left")
    else:
        if len(run_df) != len(paper_df):
            print(f"  [FAIL] row_count: paper={len(paper_df)}, run={len(run_df)}")
            return False, 0, 1
        merged = pd.concat(
            [paper_df.add_suffix("_paper"), run_df.add_suffix("_run")], axis=1
        )

    numeric_cols = []
    for col in paper_df.columns:
        if key is not None and col == key:
            continue
        if key is None and f"{col}_paper" not in merged.columns:
            continue
        left = f"{col}_paper"
        right = f"{col}_run"
        if left in merged.columns and right in merged.columns:
            if pd.api.types.is_numeric_dtype(merged[left]) and pd.api.types.is_numeric_dtype(merged[right]):
                numeric_cols.append(col)

    if not numeric_cols:
        print("  [WARN] No comparable numeric columns found.")
        return True, 0, 0

    failed = 0
    passed = 0
    for col in numeric_cols:
        left = merged[f"{col}_paper"]
        right = merged[f"{col}_run"]
        if right.isna().all():
            print(f"  [FAIL] {col}: missing in run output")
            failed += 1
            continue

        diffs = []
        for paper_val, run_val in zip(left, right):
            if pd.isna(paper_val) or pd.isna(run_val):
                continue
            denom = abs(float(paper_val)) if abs(float(paper_val)) > 1e-9 else 1.0
            diffs.append(abs(float(paper_val) - float(run_val)) / denom)

        if not diffs:
            print(f"  [WARN] {col}: no overlapping non-null values")
            continue

        mean_dev = float(np.mean(diffs))
        max_dev = float(np.max(diffs))
        status = "PASS" if max_dev < TOLERANCE else "FAIL"
        print(
            f"  [{status}] {col}: mean_dev={mean_dev:.2%}, max_dev={max_dev:.2%}, "
            f"tol={TOLERANCE:.0%}"
        )
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    return failed == 0, passed, failed

checks = {
    "table2_rl_comparison.csv": ["method", "config", "config_name"],
    "table3_main_comparison.csv": ["config", "config_name", "method"],
    "table4_ablation.csv": ["config", "config_name"],
    "table5_adversarial.csv": ["attack_type", "parameter"],
    "fig3_latency_data.csv": ["config", "n_uavs"],
    "fig5_tradeoff_data.csv": ["config", "n_uavs"],
}

all_pass = True
latest_run = get_latest_run_dir()
if latest_run is None:
    print("[FAIL] No run directories found under results/runs/. Run experiments first.")
    sys.exit(1)

print(f"Latest run: {latest_run}")
print("")

required_run_files = list(checks.keys())
missing_run = [name for name in required_run_files if not (latest_run / name).exists()]
if missing_run:
    print("[FAIL] Latest run is missing required reproducibility files:")
    for name in missing_run:
        print(f"  - {latest_run / name}")
    print("Run full reproduction first (e.g., bash experiments/run_all.sh --skip_training).")
    sys.exit(1)

total_passed = 0
total_failed = 0

for filename, preferred_keys in checks.items():
    paper_path = PAPER_DIR / filename
    if not paper_path.exists():
        print(f"[SKIP] {filename} â€” paper CSV not found")
        continue

    run_path = latest_run / filename

    print(f"Checking {filename}:")
    file_ok, passed, failed = compare_file(run_path, paper_path, preferred_keys)
    total_passed += passed
    total_failed += failed
    if not file_ok:
        all_pass = False
    print()

if all_pass:
    print(
        f"All checks PASSED â€” deviations within {TOLERANCE:.0%} tolerance "
        f"(metrics passed: {total_passed})."
    )
    sys.exit(0)
else:
    print(
        f"Some checks FAILED â€” {total_failed} metric deviations exceed "
        f"{TOLERANCE:.0%} tolerance."
    )
    sys.exit(1)
EOF
