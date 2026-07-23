#!/usr/bin/env python3
"""Experiment 06b — HITL operator-error sensitivity (tab:hitl_sensitivity in paper).

Sweeps the human-operator error probability p_err (the chance the operator
rejects an already-verified, high-confidence *true* alert) for the full
Greedy-GOMDP configuration and reports how the missed-detection rate FN_r and the
false-alert rate F_p respond. Governance compliance stays pinned at 100% by
Theorem 1 regardless of p_err — the operator can only *withhold* an alert, never
authorise an unverified one — which is the point of the table.

This runs entirely on the shared simulation core (experiments/utils/runner.py),
which now threads p_err through to the HITL oracle and counts FN directly. The
FN_r / F_p magnitudes are calibration-pending until WS1; the qualitative claim
(FN_r rises and F_p falls with p_err, compliance fixed at 100%) is reproduced by
construction.

Canonical output: results/paper/table7_hitl_sensitivity.csv  (see results/paper/MANIFEST.yaml)
Paper reference: Table (tab:hitl_sensitivity), PPO/Greedy-GOMDP p_err sweep.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys
_sys.path.insert(0, "src"); _sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode

P_ERR_GRID = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 20
GRID = 100
N_TIMESTEPS = 3000
N_UAVS = 20


def build_per_seed(p_errs, n_seeds, grid, n_timesteps, n_uavs) -> pd.DataFrame:
    rows = []
    for p_err in p_errs:
        for seed in range(n_seeds):
            r = run_episode(
                seed=seed, config_name="greedy_gomdp", grid_size=grid,
                n_timesteps=n_timesteps, n_uavs=n_uavs,
                enable_governance=True, enable_hitl=True, enable_blockchain=True,
                enable_verification=True, enable_coordination=True,
                hitl_rejection_rate=p_err,
            )
            rows.append({
                "p_err": p_err, "seed": seed,
                "fn_pct": round(r.fn_pct, 2),
                "fp_pct": round(r.fp_pct, 2),
                # GOMDP: compliance is definitional (Theorem 1). Report measured.
                "gov_compliance_pct": round(100.0 * float(r.governance_compliant), 1),
            })
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p_err in sorted(per_seed["p_err"].unique()):
        sub = per_seed[per_seed["p_err"] == p_err]
        fn = sub["fn_pct"].to_numpy(dtype=float)
        fp = sub["fp_pct"].to_numpy(dtype=float)
        comp = sub["gov_compliance_pct"].to_numpy(dtype=float)
        rows.append({
            "p_err": p_err,
            "fn_mean": round(float(np.mean(fn)), 1),
            "fn_std": round(float(np.std(fn, ddof=1)), 1) if len(fn) > 1 else 0.0,
            "fp_mean": round(float(np.mean(fp)), 1),
            "fp_std": round(float(np.std(fp, ddof=1)), 1) if len(fp) > 1 else 0.0,
            "gov_compliance_pct": round(float(np.mean(comp)), 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/runs/hitl_sensitivity")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds, grid, n_ts, n_uavs = (N_SEEDS, GRID, N_TIMESTEPS, N_UAVS)
    if args.smoke:
        n_seeds, grid, n_ts, n_uavs = 3, 30, 300, 8

    per_seed = build_per_seed(P_ERR_GRID, n_seeds, grid, n_ts, n_uavs)
    agg = aggregate(per_seed)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(outdir / "table7_hitl_sensitivity_per_seed.csv", index=False)
    agg.to_csv(outdir / "table7_hitl_sensitivity.csv", index=False)

    print("=== HITL operator-error sensitivity (Greedy-GOMDP) ===")
    print(agg.to_string(index=False))
    print("\n(compliance pinned at 100% by Theorem 1; FN/FP magnitude calibration-pending)")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
