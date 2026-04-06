#!/usr/bin/env python3
"""Experiment 10 — Stress testing (Figure 6 in paper).

Three stress conditions:
(a) Sensor failure cascade: k% of UAV sensors offline.
(b) Communication disruption: packet drop probability p_drop.
(c) High-burst anomaly frequency: up to 5x nominal injection rate.

Paper reference: Figure 6, Section VI-D (Stress Testing).
Output: results/runs/<hash>/fig6_stress_test_data.csv
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


def main(config_path: str, smoke: bool = False) -> None:
    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = 3 if smoke else 20
    n_uavs = 5 if smoke else 20
    n_timesteps = 100 if smoke else 3000

    rows = []

    # (a) Sensor failure cascade
    failure_rates = [0.0, 0.10, 0.20, 0.30, 0.40]
    if smoke:
        failure_rates = [0.0, 0.20]
    for policy in ["greedy_gomdp", "adaptive_ai"]:
        is_gov = policy == "greedy_gomdp"
        for rate in failure_rates:
            lds = []
            for seed in range(n_seeds):
                r = run_episode(
                    seed=seed, config_name=policy,
                    n_uavs=n_uavs, n_timesteps=n_timesteps,
                    enable_governance=is_gov, enable_hitl=is_gov,
                    enable_blockchain=is_gov, enable_verification=is_gov,
                    enable_coordination=True,
                    sensor_failure_rate=rate,
                )
                if r.ld < float("inf"):
                    lds.append(r.ld)
            rows.append({
                "stress_type": "sensor_failure",
                "config": policy,
                "parameter": rate,
                "ld_mean": round(float(np.mean(lds)) if lds else float(n_timesteps), 2),
                "ld_std": round(float(np.std(lds)) if lds else 0.0, 2),
            })

    # (b) Communication disruption
    drop_rates = [0.0, 0.05, 0.10, 0.20]
    if smoke:
        drop_rates = [0.0, 0.10]
    for policy in ["greedy_gomdp", "adaptive_ai"]:
        is_gov = policy == "greedy_gomdp"
        for drop in drop_rates:
            lds = []
            for seed in range(n_seeds):
                r = run_episode(
                    seed=seed, config_name=policy,
                    n_uavs=n_uavs, n_timesteps=n_timesteps,
                    enable_governance=is_gov, enable_hitl=is_gov,
                    enable_blockchain=is_gov, enable_verification=is_gov,
                    enable_coordination=True,
                    p_drop=drop,
                )
                if r.ld < float("inf"):
                    lds.append(r.ld)
            rows.append({
                "stress_type": "comm_disruption",
                "config": policy,
                "parameter": drop,
                "ld_mean": round(float(np.mean(lds)) if lds else float(n_timesteps), 2),
                "ld_std": round(float(np.std(lds)) if lds else 0.0, 2),
            })

    # (c) Burst anomaly frequency (measured via fp_pct under burst)
    burst_multipliers = [1, 2, 3, 5]
    if smoke:
        burst_multipliers = [1, 2]
    for policy in ["greedy_gomdp", "adaptive_ai"]:
        is_gov = policy == "greedy_gomdp"
        for mult in burst_multipliers:
            fps = []
            for seed in range(n_seeds):
                r = run_episode(
                    seed=seed, config_name=policy,
                    n_uavs=n_uavs, n_timesteps=n_timesteps,
                    enable_governance=is_gov, enable_hitl=is_gov,
                    enable_blockchain=is_gov, enable_verification=is_gov,
                    enable_coordination=True,
                    burst_mode=(mult > 1),
                    p_spoof=min(0.02 * mult, 0.15),
                )
                fps.append(r.fp_pct)
            rows.append({
                "stress_type": "burst_anomaly",
                "config": policy,
                "parameter": float(mult),
                "fp_mean": round(float(np.mean(fps)), 2),
                "fp_std": round(float(np.std(fps)), 2),
            })

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "fig6_stress_test_data.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("experiment_complete", output=str(out_path))
    print(f"\n=== Figure 6 Stress Test Data ===\n{out_df.to_string(index=False)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/stress_testing.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
