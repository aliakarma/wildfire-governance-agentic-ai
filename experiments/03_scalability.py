#!/usr/bin/env python3
"""Experiment 03 — Scalability: Detection latency vs. UAV fleet size (Figure 3).

Evaluates Ld for N ∈ {5, 10, 20, 40} UAVs across all three configurations.
Also validates Proposition 1 latency bound: E[Ld] <= A/(vN) + Delta.

Paper reference: Figure 3, Section VI-B (Detection Latency).
Output: results/runs/<hash>/fig3_latency_data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys; _sys.path.insert(0, 'src'); _sys.path.insert(0, '.')
from typing import List

import numpy as np
import pandas as pd

from experiments.utils.io_utils import save_results
from experiments.utils.runner import EpisodeResult, run_episode
from wildfire_governance.decision.cpomdp import WildfireCPOMDP
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")
UAV_FLEET_SIZES = [5, 10, 20, 40]


def main(config_path: str, smoke: bool = False) -> None:
    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        n_seeds = cfg.simulation.n_seeds
        n_timesteps = cfg.simulation.n_timesteps
    except Exception:
        n_seeds, n_timesteps = 20, 3000

    if smoke:
        n_seeds, n_timesteps = 2, 100

    fleet_sizes = [5, 20] if smoke else UAV_FLEET_SIZES
    configs_map = {
        "greedy_gomdp": dict(enable_governance=True, enable_hitl=True,
                             enable_blockchain=True, enable_verification=True,
                             enable_coordination=True),
        "adaptive_ai": dict(enable_governance=False, enable_hitl=False,
                            enable_blockchain=False, enable_verification=True,
                            enable_coordination=True),
        "static": dict(enable_governance=False, enable_hitl=False,
                       enable_blockchain=False, enable_verification=False,
                       enable_coordination=False),
    }

    rows = []
    cpomdp = WildfireCPOMDP()
    grid_area = 100.0 * 100.0
    velocity = 1.0
    delta = 4.2  # BC + HV overhead

    for n_uavs in fleet_sizes:
        for config_name, kwargs in configs_map.items():
            lds = []
            for seed in range(n_seeds):
                result = run_episode(
                    seed=seed, config_name=config_name,
                    n_uavs=n_uavs, n_timesteps=n_timesteps, **kwargs
                )
                if result.ld < float("inf"):
                    lds.append(result.ld)

            ld_mean = float(np.mean(lds)) if lds else float(n_timesteps)
            ld_std = float(np.std(lds)) if lds else 0.0
            bound = cpomdp.latency_bound(grid_area, velocity, n_uavs, delta)
            rows.append({
                "config": config_name,
                "n_uavs": n_uavs,
                "ld_mean": round(ld_mean, 2),
                "ld_std": round(ld_std, 2),
                "proposition1_bound": round(bound, 2),
            })
            logger.info("scalability_point", config=config_name,
                        n_uavs=n_uavs, ld_mean=round(ld_mean, 2))

    out_df = pd.DataFrame(rows)
    agg_path = out_dir / "fig3_latency_data.csv"
    out_df.to_csv(agg_path, index=False)
    logger.info("experiment_complete", output=str(agg_path))
    print(f"\n=== Figure 3 Data ===\n{out_df.to_string(index=False)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/scalability_uav_fleet.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
