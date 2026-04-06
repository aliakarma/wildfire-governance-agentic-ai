#!/usr/bin/env python3
"""Experiment 01 — Main comparison (Table III in paper).

Runs Greedy-GOMDP, Adaptive AI (no governance), and Static Monitoring
for n_seeds episodes at N=20 UAVs.

Expected runtime: ~25 minutes on 8 CPU cores (20 seeds × 3 configs × 3000 steps).
Output: results/runs/<hash>/table3_main_comparison.csv
Paper reference: Table III, Section VI-B.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys; _sys.path.insert(0, 'src'); _sys.path.insert(0, '.')

import pandas as pd

from experiments.utils.io_utils import aggregate_to_table, save_results
from experiments.utils.runner import run_episode
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")
PAPER_CSV = Path("results/paper/table3_main_comparison.csv")


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

    if smoke:
        n_seeds, n_uavs, n_timesteps = 2, 5, 100

    configs = [
        dict(config_name="greedy_gomdp", enable_governance=True, enable_hitl=True,
             enable_blockchain=True, enable_verification=True, enable_coordination=True),
        dict(config_name="adaptive_ai", enable_governance=False, enable_hitl=False,
             enable_blockchain=False, enable_verification=True, enable_coordination=True),
        dict(config_name="static", enable_governance=False, enable_hitl=False,
             enable_blockchain=False, enable_verification=False, enable_coordination=False),
    ]

    all_results = []
    for cfg_kwargs in configs:
        name = cfg_kwargs["config_name"]
        for seed in range(n_seeds):
            result = run_episode(
                seed=seed, n_uavs=n_uavs, n_timesteps=n_timesteps, **cfg_kwargs
            )
            all_results.append(result)
            logger.info("seed_complete", config=name, seed=seed,
                        ld=round(result.ld, 1), fp=round(result.fp_pct, 1))

    csv_path = save_results(all_results, out_dir, "table3_raw.csv")
    agg = aggregate_to_table(all_results)
    agg_path = out_dir / "table3_main_comparison.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("experiment_complete", output=str(agg_path))
    print(f"\n=== Table III Results ===\n{agg.to_string(index=False)}\n")
    print(f"Full results saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
