#!/usr/bin/env python3
"""Experiment 13: Sensor Noise vs. Topology Density Ablation.
Gradually increase `noise_std` in `ThermalUAVSensor` and test the drop-off 
in detection speed against UAV fleet density (N=5 to N=50).
"""

import sys
import numpy as np
import polars as pl
import joblib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.utils.io_utils import save_results
from experiments.utils.runner import run_episode
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")
PAPER_CSV = Path("results/paper/table_sensor_noise_ablation.csv")

def main(smoke: bool = False):
    run_hash = generate_run_hash()
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("experiment_start", experiment="sensor_noise_ablation", out_dir=str(out_dir))

    n_seeds = 2 if smoke else 10
    n_timesteps = 100 if smoke else 1000

    uav_densities = [5, 20, 50]
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5] if not smoke else [0.01, 0.1]

    all_results = []
    
    # We will use joblib to parallelize across seeds for each config
    for n_uavs in uav_densities:
        for noise in noise_levels:
            logger.info("evaluating", n_uavs=n_uavs, noise=noise)
            
            def run_seed(seed):
                res = run_episode(
                    seed=seed,
                    config_name=f"greedy_uav_{n_uavs}_noise_{noise}",
                    grid_size=100,
                    n_timesteps=n_timesteps,
                    n_uavs=n_uavs,
                    uav_noise_std=noise,
                    enable_governance=True,
                    enable_hitl=True,
                    enable_blockchain=True,
                    enable_verification=True,
                    enable_coordination=True
                )
                return res

            results = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(run_seed)(seed) for seed in range(n_seeds)
            )
            
            for s, res in enumerate(results):
                all_results.append({
                    "n_uavs": n_uavs,
                    "noise_std": noise,
                    "seed": s,
                    "latency_detect": res.ld,
                    "false_positive_rate": res.fp_pct,
                    "theorem1_satisfied": res.theorem1_satisfied
                })

    df = pl.DataFrame(all_results)
    
    # Save the raw results
    csv_path = out_dir / "sensor_noise_ablation.csv"
    df.write_csv(csv_path)

    # Save to paper path
    df.write_csv(PAPER_CSV)

    logger.info("experiment_complete", output=str(PAPER_CSV))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()
    main(smoke=args.smoke)
