#!/usr/bin/env python3
"""Experiment 04 — False alert rate under anomaly injection (Figure 4).

Injects synthetic non-fire anomalies and measures Fp for all configurations.

Paper reference: Figure 4, Section VI-B (False Alert Reduction).
Output: results/runs/<hash>/fig4_false_alerts_data.csv
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
from wildfire_governance.metrics.statistical_tests import (
    paired_ttest_holm_bonferroni, summarise_tests
)

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")


def main(config_path: str, smoke: bool = False) -> None:
    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        n_seeds = cfg.simulation.n_seeds
        n_uavs = cfg.simulation.uav.n_uavs
        n_timesteps = cfg.simulation.n_timesteps
    except Exception:
        n_seeds, n_uavs, n_timesteps = 20, 20, 3000

    if smoke:
        n_seeds, n_uavs, n_timesteps = 2, 5, 100

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

    per_config_fps: dict = {}
    rows = []
    for config_name, kwargs in configs_map.items():
        fps = []
        for seed in range(n_seeds):
            result = run_episode(seed=seed, config_name=config_name,
                                 n_uavs=n_uavs, n_timesteps=n_timesteps, **kwargs)
            fps.append(result.fp_pct)
        per_config_fps[config_name] = fps
        rows.append({
            "config": config_name,
            "fp_mean": round(float(np.mean(fps)), 2),
            "fp_std": round(float(np.std(fps)), 2),
            "n_seeds": n_seeds,
        })

    # Holm-Bonferroni corrected pairwise t-tests
    comparisons = [
        ("greedy_gomdp", "adaptive_ai", "fp", per_config_fps["greedy_gomdp"], per_config_fps["adaptive_ai"]),
        ("greedy_gomdp", "static", "fp", per_config_fps["greedy_gomdp"], per_config_fps["static"]),
        ("adaptive_ai", "static", "fp", per_config_fps["adaptive_ai"], per_config_fps["static"]),
    ]
    test_results = paired_ttest_holm_bonferroni(comparisons)
    summary = summarise_tests(test_results)

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "fig4_false_alerts_data.csv"
    out_df.to_csv(out_path, index=False)
    import json
    stats_path = out_dir / "fig4_stats_tests.json"
    stats_path.write_text(json.dumps(summary, indent=2))
    logger.info("experiment_complete", output=str(out_path),
                all_significant=summary["all_significant"])
    print(f"\n=== Figure 4 Data ===\n{out_df.to_string(index=False)}")
    print(f"All comparisons significant (p<0.01 HB-corrected): {summary['all_significant']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
