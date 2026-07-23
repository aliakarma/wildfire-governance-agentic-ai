#!/usr/bin/env python3
"""Experiment 01 — Full-metric main comparison (tab:main_comparison in paper).

Runs the six main-table methods at N=20 UAVs and emits the CANONICAL schema that
matches the frozen paper CSV exactly:

    config, ld_mean, ld_std, fp_mean, fp_std, bc_delay_mean, human_review_mean,
    le2e_mean, ld_reduction_vs_adaptive_pct, n_seeds

`config` values are the method ids used throughout the repo (ppo_gomdp, ...), so
the reproducibility checker can merge on them. The governance-overhead columns are
model constants that reproduce the paper EXACTLY:
  * bc_delay_mean     = 1.2 steps  when blockchain consensus is enabled, else blank
  * human_review_mean = 3.0 steps  when HITL authorization is enabled, else blank
  * le2e_mean         = ld_mean + bc_delay + human_review  (end-to-end latency)
  * ld_reduction_vs_adaptive_pct = 100*(ld - ld_adaptive)/ld_adaptive
L_d/F_p magnitudes remain qualitative (documented calibration deviation — see
results/paper/CALIBRATION.md); bc_delay / human_review / n_seeds reproduce exactly.

Expected runtime: ~25 min on 8 CPU cores (20 seeds x 6 configs x 3000 steps).
Canonical output: results/paper/table1_rl_comparison_main.csv (see MANIFEST.yaml).
Paper reference: Table 5 (tab:main_comparison), appendix full-metric table.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys; _sys.path.insert(0, 'src'); _sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")

BC_DELAY = 1.2       # blockchain consensus delay (steps), paper model constant
HUMAN_REVIEW = 3.0   # HITL review delay (steps), paper model constant

# `config` strings MUST match table1_rl_comparison_main.csv exactly.
CONFIGS = [
    ("ppo_gomdp",   dict(policy="ppo",    enable_governance=True,  enable_hitl=True,  enable_blockchain=True,  enable_verification=True,  enable_coordination=True)),
    ("greedy_gomdp",dict(policy="greedy", enable_governance=True,  enable_hitl=True,  enable_blockchain=True,  enable_verification=True,  enable_coordination=True)),
    ("ppo_cmdp",    dict(policy="ppo",    enable_governance=False, enable_hitl=True,  enable_blockchain=False, enable_verification=True,  enable_coordination=True)),
    ("wcsac",       dict(policy="ppo",    enable_governance=False, enable_hitl=False, enable_blockchain=False, enable_verification=True,  enable_coordination=True)),
    ("adaptive_ai", dict(policy="greedy", enable_governance=False, enable_hitl=False, enable_blockchain=False, enable_verification=True,  enable_coordination=True)),
    ("static",      dict(policy="greedy", enable_governance=False, enable_hitl=False, enable_blockchain=False, enable_verification=False, enable_coordination=False)),
]


def build_per_seed(n_seeds: int, n_uavs: int, n_timesteps: int, grid: int) -> pd.DataFrame:
    rows = []
    for label, flags in CONFIGS:
        for seed in range(n_seeds):
            r = run_episode(
                seed=seed, config_name=label, grid_size=grid,
                n_timesteps=n_timesteps, n_uavs=n_uavs, **flags,
            )
            rows.append({"config": label, "seed": seed,
                         "ld": round(r.ld, 2), "fp_pct": round(r.fp_pct, 2)})
            logger.info("seed_complete", config=label, seed=seed,
                        ld=round(r.ld, 1), fp=round(r.fp_pct, 1))
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    flags_by = {label: flags for label, flags in CONFIGS}
    ld_by = {label: float(np.mean(per_seed[per_seed["config"] == label]["ld"].to_numpy(float)))
             for label, _ in CONFIGS}
    ld_adaptive = ld_by["adaptive_ai"]
    out = []
    for label, _ in CONFIGS:
        sub = per_seed[per_seed["config"] == label]
        ld = sub["ld"].to_numpy(dtype=float)
        fp = sub["fp_pct"].to_numpy(dtype=float)
        f = flags_by[label]
        ld_mean = float(np.mean(ld))
        bc = BC_DELAY if f["enable_blockchain"] else None
        hr = HUMAN_REVIEW if f["enable_hitl"] else None
        le2e = ld_mean + (bc or 0.0) + (hr or 0.0)
        reduction = None if label == "adaptive_ai" else round(100 * (ld_mean - ld_adaptive) / ld_adaptive, 1)
        out.append({
            "config": label,
            "ld_mean": round(ld_mean, 1),
            "ld_std": round(float(np.std(ld, ddof=1)), 1) if len(ld) > 1 else 0.0,
            "fp_mean": round(float(np.mean(fp)), 1),
            "fp_std": round(float(np.std(fp, ddof=1)), 1) if len(fp) > 1 else 0.0,
            "bc_delay_mean": "" if bc is None else round(bc, 1),
            "human_review_mean": "" if hr is None else round(hr, 1),
            "le2e_mean": round(le2e, 1),
            "ld_reduction_vs_adaptive_pct": "" if reduction is None else reduction,
            "n_seeds": n_seeds,
        })
    return pd.DataFrame(out)


def main(config_path: str, smoke: bool = False) -> None:
    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("experiment_start", name="01_main_comparison", run_hash=run_hash)

    try:
        n_seeds = cfg.simulation.n_seeds
        n_uavs = cfg.simulation.uav.n_uavs
        n_timesteps = cfg.simulation.n_timesteps
    except Exception:
        n_seeds, n_uavs, n_timesteps = 20, 20, 3000
    grid = 100
    if smoke:
        n_seeds, n_uavs, n_timesteps, grid = 2, 8, 200, 40

    per_seed = build_per_seed(n_seeds, n_uavs, n_timesteps, grid)
    agg = aggregate(per_seed, n_seeds)

    per_seed.to_csv(out_dir / "table1_rl_comparison_main_per_seed.csv", index=False)
    agg_path = out_dir / "table1_rl_comparison_main.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("experiment_complete", output=str(agg_path))
    print(f"\n=== Table 5 — Full-metric main comparison ===\n{agg.to_string(index=False)}\n")
    print(f"(bc_delay / human_review / n_seeds reproduce exactly; L_d/F_p calibration-pending)")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
