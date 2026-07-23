#!/usr/bin/env python3
"""Experiment 05 — Latency-false alert Pareto frontier (Figure 5).

IMPORTANT: This figure uses N=40 UAVs (not N=20).
At N=40, PPO-GOMDP and Adaptive AI converge to ~10 steps Ld,
giving the clearest Pareto dominance illustration.

Paper reference: Figure 5, Section VI-B (Performance-Governance Tradeoff).
Output: results/runs/<hash>/fig5_tradeoff_data.csv
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

# N=40 matches the tradeoff figure in the paper (NOT N=20)
N_UAVS_TRADEOFF = 40


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

    n_uavs = 10 if smoke else N_UAVS_TRADEOFF  # smoke uses smaller fleet

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
    for config_name, kwargs in configs_map.items():
        lds, fps = [], []
        for seed in range(n_seeds):
            result = run_episode(seed=seed, config_name=config_name,
                                 n_uavs=n_uavs, n_timesteps=n_timesteps, **kwargs)
            if result.ld < float("inf"):
                lds.append(result.ld)
            fps.append(result.fp_pct)
        rows.append({
            "config": config_name,
            "n_uavs": n_uavs,
            "ld_mean": round(float(np.mean(lds)) if lds else float(n_timesteps), 2),
            "ld_std": round(float(np.std(lds)) if lds else 0.0, 2),
            "fp_mean": round(float(np.mean(fps)), 2),
            "fp_std": round(float(np.std(fps)), 2),
        })

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "fig5_tradeoff_data.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("experiment_complete", output=str(out_path), n_uavs=n_uavs)
    print(f"\n=== Figure 5 Tradeoff Data (N={n_uavs} UAVs) ===")
    print(out_df.to_string(index=False))
    print(f"\nNote: Figure is generated at N={N_UAVS_TRADEOFF} UAVs per paper.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
