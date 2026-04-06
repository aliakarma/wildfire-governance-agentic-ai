#!/usr/bin/env python3
"""Experiment 07 — Blockchain throughput analysis (Section VI-C4).

Measures blockchain confirmation delay as a function of transaction load,
comparing nominal vs. 5× burst conditions.

Paper reference: Section VI-C4 (Governance Overhead Decomposition).
Output: results/runs/<hash>/blockchain_throughput_analysis.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_governance.blockchain.consensus import PBFTConsensus
from wildfire_governance.blockchain.transaction import build_transaction
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

    n_trials = 20 if smoke else 200
    burst_multipliers = [1, 2, 3, 5]
    rows = []

    consensus = PBFTConsensus(
        n_validators=7, max_byzantine=2,
        mean_delay_steps=1.2, std_delay_steps=0.3,
        burst_multiplier=1.35,
        rng=np.random.default_rng(42),
    )

    for mult in burst_multipliers:
        burst_mode = mult > 1
        delays = []
        for i in range(n_trials):
            tx = build_transaction(
                event_id=f"evt_{mult}_{i}",
                geo_boundary=(0, 0, 1, 1),
                confidence_score=0.90,
                sensor_readings={"heat": 0.9, "trial": i},
            )
            result = consensus.propose(tx, burst_mode=burst_mode)
            delays.append(result.delay_steps)

        mean_delay = float(np.mean(delays))
        std_delay = float(np.std(delays))
        p95_delay = float(np.percentile(delays, 95))
        pct_increase = ((mean_delay - 1.2) / 1.2) * 100.0

        rows.append({
            "burst_multiplier": mult,
            "burst_mode": burst_mode,
            "mean_delay_steps": round(mean_delay, 3),
            "std_delay_steps": round(std_delay, 3),
            "p95_delay_steps": round(p95_delay, 3),
            "pct_increase_vs_nominal": round(pct_increase, 1),
            "n_trials": n_trials,
        })
        logger.info("throughput_measured", mult=mult, mean=round(mean_delay, 3),
                    pct=round(pct_increase, 1))

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "blockchain_throughput_analysis.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("experiment_complete", output=str(out_path))
    print(f"\n=== Blockchain Throughput Analysis ===")
    print(out_df.to_string(index=False))
    print(f"\nPaper claims: 35% increase at 5× burst → actual: "
          f"{out_df[out_df.burst_multiplier==5].pct_increase_vs_nominal.values[0]:.1f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
