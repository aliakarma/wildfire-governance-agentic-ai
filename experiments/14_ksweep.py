#!/usr/bin/env python3
"""Experiment 14 — Validator-count sweep (tab:ksweep in paper).

Reproduces the paper's Table (tab:ksweep) EXACTLY from real computation — the
theory column is the closed-form binomial tail of Theorem 2
(src/wildfire_governance/gomdp/breach_probability.py) and the empirical column is
a Monte-Carlo of the same compromise process. No calibration or back-fill.

Canonical output: results/paper/table6_ksweep.csv  (see results/paper/MANIFEST.yaml)
Paper reference: Table (tab:ksweep), p_c = 0.10, f = floor((k-1)/3).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys
_sys.path.insert(0, "src"); _sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from wildfire_governance.gomdp.breach_probability import compute_breach_probability_gomdp

P_C = 0.10
K_VALUES = [4, 7, 10, 13]
# Monte-Carlo settings are fixed here and mirrored in
# scripts/generate_all_paper_results.py so the empirical column is identical
# whichever entry point produces it.
N_TRIALS = 10_000
SEED = 1


def simulate_empirical(k: int, f: int, p_c: float, n_trials: int, seed: int) -> float:
    """Monte-Carlo breach frequency: breach iff > f of k validators compromised."""
    rng = np.random.default_rng(seed)
    compromised = rng.random((n_trials, k)) < p_c
    return float(np.mean(compromised.sum(axis=1) > f))


def build_table() -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        f = (k - 1) // 3
        theory = compute_breach_probability_gomdp(k, f, P_C)
        empirical = simulate_empirical(k, f, P_C, N_TRIALS, SEED)
        rows.append({
            "k": k, "f": f,
            "p_break_gomdp_theory": round(theory, 3),
            "empirical": round(empirical, 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/runs/ksweep/table6_ksweep.csv")
    ap.add_argument("--paper", action="store_true",
                    help="also write the canonical results/paper/table6_ksweep.csv")
    args = ap.parse_args()

    df = build_table()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if args.paper:
        df.to_csv(Path("results/paper/table6_ksweep.csv"), index=False)
    print("=== Table ksweep (p_c=0.10) ===")
    print(df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
