#!/usr/bin/env python3
"""Experiment 02 — Ablation study (Table IV in paper).

Isolates the contribution of each architectural component by removing
one layer at a time from the full PPO-GOMDP system.

Paper reference: Table IV, Section VI-C (Ablation Study).
Output: results/runs/<hash>/table4_ablation.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys; _sys.path.insert(0, 'src'); _sys.path.insert(0, '.')

from experiments.utils.io_utils import aggregate_to_table, save_results
from experiments.utils.runner import run_episode
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")


def main(config_path: str, smoke: bool = False) -> None:
    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("experiment_start", name="02_ablation_study", run_hash=run_hash)

    try:
        n_seeds = cfg.simulation.n_seeds
        n_uavs = cfg.simulation.uav.n_uavs
        n_timesteps = cfg.simulation.n_timesteps
    except Exception:
        n_seeds, n_uavs, n_timesteps = 20, 20, 3000

    if smoke:
        n_seeds, n_uavs, n_timesteps = 2, 5, 100

    ablations = [
        dict(config_name="full_greedy_gomdp", enable_governance=True, enable_hitl=True,
             enable_blockchain=True, enable_verification=True, enable_coordination=True),
        dict(config_name="minus_coordination", enable_governance=True, enable_hitl=True,
             enable_blockchain=True, enable_verification=True, enable_coordination=False),
        dict(config_name="minus_hitl", enable_governance=True, enable_hitl=False,
             enable_blockchain=True, enable_verification=True, enable_coordination=True),
        dict(config_name="minus_blockchain", enable_governance=True, enable_hitl=True,
             enable_blockchain=False, enable_verification=True, enable_coordination=True),
        dict(config_name="minus_verification", enable_governance=True, enable_hitl=True,
             enable_blockchain=True, enable_verification=False, enable_coordination=True),
    ]

    all_results = []
    for abl in ablations:
        name = abl["config_name"]
        for seed in range(n_seeds):
            result = run_episode(seed=seed, n_uavs=n_uavs, n_timesteps=n_timesteps, **abl)
            all_results.append(result)
        logger.info("ablation_complete", config=name)

    save_results(all_results, out_dir, "table4_raw.csv")
    agg = aggregate_to_table(all_results)
    agg_path = out_dir / "table4_ablation.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("experiment_complete", output=str(agg_path))
    print(f"\n=== Table IV Ablation Results ===\n{agg.to_string(index=False)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
