#!/usr/bin/env python3
"""Experiment 06 — Threshold sensitivity analysis (Section VI-C7).

Grid search over tau1 ∈ {0.50,0.55,0.60,0.65,0.70} and
tau2 ∈ {0.75,0.80,0.85,0.90} with tau2 > tau1 enforced.

Paper reference: Section VI-C7, threshold sensitivity paragraph.
Output: results/runs/<hash>/threshold_sensitivity.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_governance.simulation.grid_environment import (
    EnvironmentConfig,
    WildfireGridEnvironment,
)
from wildfire_governance.verification.confidence_scorer import TwoStageConfidenceScorer
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash, set_global_seed

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")

TAU1_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]
TAU2_GRID = [0.75, 0.80, 0.85, 0.90]


def main(config_path: str, smoke: bool = False) -> None:
    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = 5 if smoke else 20
    n_timesteps = 100 if smoke else 500  # shorter for grid search

    rows = []
    for tau1 in TAU1_GRID:
        for tau2 in TAU2_GRID:
            if tau2 <= tau1:
                continue
            fps, lds = [], []
            for seed in range(n_seeds):
                set_global_seed(seed)
                rng = np.random.default_rng(seed)
                env = WildfireGridEnvironment(EnvironmentConfig(
                    grid_size=20 if smoke else 100, n_timesteps=n_timesteps
                ))
                env.reset(seed=seed)
                scorer = TwoStageConfidenceScorer(tau1=tau1, tau2=tau2)
                n_alerts, n_false, first_det = 0, 0, None
                for t in range(n_timesteps):
                    obs, done, _ = env.step([])
                    heat = float(obs["heat_map"].max())
                    if heat > tau1 and first_det is None:
                        first_det = t
                    wx = float(np.clip(
                        obs["wind_field"].mean() - obs["humidity_field"].mean() + 0.5,
                        0.0, 1.0,
                    ))
                    result = scorer.score(min(heat, 1.0), wx, heat > tau2)
                    if result.escalated_to_hitl:
                        is_fire = bool(obs["fire_mask"].max() > 0.5)
                        n_alerts += 1
                        if not is_fire:
                            n_false += 1
                    if done:
                        break
                fps.append((n_false / max(1, n_alerts)) * 100.0)
                lds.append(float(first_det) if first_det is not None else float(n_timesteps))

            rows.append({
                "tau1": tau1, "tau2": tau2,
                "fp_mean": round(float(np.mean(fps)), 2),
                "ld_mean": round(float(np.mean(lds)), 2),
            })

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "threshold_sensitivity.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("experiment_complete", output=str(out_path))
    print(f"\n=== Threshold Sensitivity (tau1=0.60, tau2=0.80 is paper default) ===")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/sensitivity_thresholds.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
