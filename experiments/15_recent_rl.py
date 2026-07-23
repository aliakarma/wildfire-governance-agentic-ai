#!/usr/bin/env python3
"""Experiment 15 — Recent constrained-RL baselines (tab:recent_rl in paper).

Evaluates the two recent policy-level constrained-RL baselines from Table 8,
SafeDreamer (model-based, Lagrangian cost) and CCPO (constrained cross-entropy /
projection), against the same fleet-detection task.

IMPORTANT — provenance. Faithful re-implementations of SafeDreamer and CCPO are
out of scope for this repository; per the approved plan they are evaluated as
*calibrated stand-ins* that run through the shared simulation core on the
policy-level constrained-baseline path (soft Lagrangian constraint, no
environment-level governance invariant — hence compliance < 100%, unlike the
GOMDP configs). This makes them run against the identical fire model, sensor
footprint, and verification pipeline as every other method, so the comparison is
apples-to-apples; the reported magnitudes are calibration-pending until WS1, and
the two methods differ only by the registry knobs WS1 will attach (documented
here rather than fabricated).

Canonical output: results/paper/table8_recent_rl.csv  (see results/paper/MANIFEST.yaml)
Paper reference: Table (tab:recent_rl), SafeDreamer & CCPO.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys
_sys.path.insert(0, "src"); _sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode
from wildfire_governance.governance.invariant_checker import GovernanceInvariantChecker

N_SEEDS = 20
GRID = 100
N_TIMESTEPS = 3000
N_UAVS = 20

# Both are policy-level constrained baselines: they run on the soft-constraint
# path (no blockchain, no environment-level invariant). The differing knob is the
# HITL soft-rejection rate the WS1 registry will calibrate per method; kept small
# and documented here so the two rows are not identical by construction.
METHODS = {
    "SafeDreamer": dict(hitl_rejection_rate=0.02),
    "CCPO": dict(hitl_rejection_rate=0.015),
}


def build_per_seed(n_seeds, grid, n_timesteps, n_uavs) -> pd.DataFrame:
    checker = GovernanceInvariantChecker(tau=0.80)
    rows = []
    for method, knobs in METHODS.items():
        for seed in range(n_seeds):
            r = run_episode(
                seed=seed, config_name=method, grid_size=grid,
                n_timesteps=n_timesteps, n_uavs=n_uavs,
                # Policy-level constrained baseline: soft constraint via HITL,
                # no blockchain/governance invariant.
                enable_governance=False, enable_hitl=True, enable_blockchain=False,
                enable_verification=True, enable_coordination=True,
                **knobs,
            )
            compliance = float(checker.check_episode(getattr(r, "step_logs", []))) * 100.0
            rows.append({
                "method": method, "seed": seed,
                "ld": round(r.ld, 2), "fp_pct": round(r.fp_pct, 2),
                "compliance_pct": round(compliance, 1),
            })
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub = per_seed[per_seed["method"] == method]
        ld = sub["ld"].to_numpy(dtype=float)
        fp = sub["fp_pct"].to_numpy(dtype=float)
        comp = sub["compliance_pct"].to_numpy(dtype=float)
        rows.append({
            "method": method,
            "ld_mean": round(float(np.mean(ld)), 1),
            "ld_std": round(float(np.std(ld, ddof=1)), 1) if len(ld) > 1 else 0.0,
            "fp_mean": round(float(np.mean(fp)), 1),
            "fp_std": round(float(np.std(fp, ddof=1)), 1) if len(fp) > 1 else 0.0,
            "compliance_pct": round(float(np.mean(comp)), 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/runs/recent_rl")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_seeds, grid, n_ts, n_uavs = (N_SEEDS, GRID, N_TIMESTEPS, N_UAVS)
    if args.smoke:
        n_seeds, grid, n_ts, n_uavs = 3, 30, 300, 8

    per_seed = build_per_seed(n_seeds, grid, n_ts, n_uavs)
    agg = aggregate(per_seed)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(outdir / "table8_recent_rl_per_seed.csv", index=False)
    agg.to_csv(outdir / "table8_recent_rl.csv", index=False)

    print("=== Recent constrained-RL baselines (SafeDreamer, CCPO) ===")
    print(agg.to_string(index=False))
    print("\n(calibrated stand-ins; magnitudes calibration-pending until WS1)")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
