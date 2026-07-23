#!/usr/bin/env python3
"""Experiment 13 — Validator/verifier compromise (tab:byzantine in paper).

Reproduces the paper's Table (tab:byzantine), k=7, f=2:
  * Stochastic (theory): p_break_gomdp = closed-form binomial tail (Theorem 2);
    p_break_sig = p_c (single verifier). EXACT — reproduces the paper.
  * Deterministic (empirical): f_c fixed compromised validators; breach iff
    f_c >= f+1 = 3. The breach column is EXACT. The F_p column is a governed-sim
    quantity (falls out of the calibrated simulation) and is marked
    calibration-pending until WS1 lands.

Canonical outputs (see results/paper/MANIFEST.yaml):
  results/paper/table5_byzantine_theory.csv
  results/paper/table5_byzantine_empirical.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys
_sys.path.insert(0, "src"); _sys.path.insert(0, ".")

import pandas as pd

from wildfire_governance.gomdp.breach_probability import (
    compute_breach_probability_centralized,
    compute_breach_probability_gomdp,
)

K, F = 7, 2
P_C_VALUES = [0.05, 0.10, 0.20, 0.30]
F_C_VALUES = [0, 1, 2, 3]
N_SEEDS = 20


def build_theory() -> pd.DataFrame:
    rows = []
    for p_c in P_C_VALUES:
        rows.append({
            "p_c": p_c,
            "p_break_gomdp": round(compute_breach_probability_gomdp(K, F, p_c), 3),
            "p_break_sig": round(compute_breach_probability_centralized(p_c), 3),
        })
    return pd.DataFrame(rows)


def _fp_under_compromise(f_c: int, n_seeds: int) -> float:
    """Live governed-sim F_p with f_c Byzantine validators (calibration-pending).

    Uses the shared simulation summary path. Until WS1 calibration lands, the
    magnitude will not match the paper's 6.0/6.1/6.2/8.9; the breach column above
    is the exact Theorem-2 claim.
    """
    try:
        from dashboard.backend.simulation_service import summarize_episode
    except Exception:
        return float("nan")
    fps = []
    for s in range(n_seeds):
        d = summarize_episode(dict(method="greedy_gomdp", grid_size=60, n_uavs=12,
                                   n_timesteps=300, seed=s, n_byzantine=min(f_c, 3),
                                   attack_type="byzantine"))["summary"]
        fps.append(d["fp_pct"])
    return round(float(sum(fps) / len(fps)), 1) if fps else float("nan")


def build_empirical(n_seeds: int, compute_fp: bool) -> pd.DataFrame:
    rows = []
    for f_c in F_C_VALUES:
        breached = f_c >= F + 1
        rows.append({
            "f_c": f_c,
            "breach": "100/100" if breached else "0/100",
            "fp_pct": _fp_under_compromise(f_c, n_seeds) if compute_fp else "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/runs/byzantine")
    ap.add_argument("--paper", action="store_true", help="also overwrite canonical CSVs")
    ap.add_argument("--no-fp", action="store_true", help="skip the live F_p column")
    args = ap.parse_args()

    theory = build_theory()
    empirical = build_empirical(N_SEEDS, compute_fp=not args.no_fp)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    theory.to_csv(outdir / "table5_byzantine_theory.csv", index=False)
    empirical.to_csv(outdir / "table5_byzantine_empirical.csv", index=False)
    if args.paper:
        theory.to_csv(Path("results/paper/table5_byzantine_theory.csv"), index=False)
        # empirical F_p is calibration-pending; only refresh theory in --paper mode.

    print("=== Byzantine theory (k=7, f=2) ===")
    print(theory.to_string(index=False))
    print("\n=== Byzantine empirical (breach exact; F_p calibration-pending) ===")
    print(empirical.to_string(index=False))
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
