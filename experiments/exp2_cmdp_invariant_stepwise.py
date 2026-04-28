#!/usr/bin/env python3
"""Experiment (Missing #2): Stepwise invariant-checker behavior without blockchain.

Runs CMDP/no-blockchain episodes via the shared runner and logs per-step
invariant status using GovernanceInvariantChecker.check_trajectory(..., include_stepwise=True).

Outputs (under results/runs/<hash>/):
  - cmdp_stepwise_invariant.csv (one row per step)
  - cmdp_episode_invariant_summary.csv (one row per episode/seed)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.utils.runner import run_episode
from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")


def main(config_path: str, n_seeds: int = 20, smoke: bool = False) -> None:
    cfg = load_config(config_path)
    run_hash = generate_run_hash(
        {
            "experiment": "exp2_cmdp_invariant_stepwise",
            "config": str(config_path),
            "n_seeds": int(n_seeds),
            "smoke": bool(smoke),
        }
    )
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep defaults aligned with Table II evaluation unless smoke.
    try:
        n_uavs = int(cfg.simulation.uav.n_uavs)
        n_timesteps = int(cfg.simulation.n_timesteps)
    except Exception:  # noqa: BLE001
        n_uavs, n_timesteps = 20, 3000

    if smoke:
        n_seeds = min(int(n_seeds), 2)
        n_uavs, n_timesteps = 5, 100

    checker = GovernanceInvariantChecker(tau=0.80)
    step_rows: list[dict] = []
    episode_rows: list[dict] = []

    for seed in range(int(n_seeds)):
        # CMDP/no-blockchain trajectory: disable governance + blockchain.
        result = run_episode(
            seed=seed,
            config_name="cmdp_surrogate",
            n_uavs=n_uavs,
            n_timesteps=n_timesteps,
            enable_governance=False,
            enable_hitl=True,
            enable_blockchain=False,
            enable_verification=True,
            enable_coordination=True,
        )

        report = checker.check_trajectory(result.step_logs or [], include_stepwise=True)

        episode_rows.append(
            {
                "seed": seed,
                "n_timesteps": report.n_timesteps,
                "n_alert_attempts": report.n_alert_attempts,
                "n_violations": report.n_violations,
                "compliance_rate": report.compliance_rate,
                "theorem1_satisfied": bool(report.theorem1_satisfied),
                # runner-level context
                "n_alerts_broadcast": int(result.n_alerts),
                "fp_pct": float(result.fp_pct),
                "ld": float(result.ld),
            }
        )

        for row in report.step_evaluations:
            step_rows.append(
                {
                    "seed": seed,
                    **row,
                }
            )

    step_df = pd.DataFrame(step_rows)
    ep_df = pd.DataFrame(episode_rows)

    step_path = out_dir / "cmdp_stepwise_invariant.csv"
    ep_path = out_dir / "cmdp_episode_invariant_summary.csv"
    step_df.to_csv(step_path, index=False)
    ep_df.to_csv(ep_path, index=False)

    logger.info(
        "exp2_complete",
        stepwise_csv=str(step_path),
        episode_csv=str(ep_path),
        mean_episode_theorem1_satisfied=float(ep_df["theorem1_satisfied"].mean()) if len(ep_df) else None,
        mean_alert_level_compliance=float(ep_df["compliance_rate"].mean()) if len(ep_df) else None,
    )

    print("\n=== CMDP Invariant Checker (Stepwise) ===")
    print(f"Stepwise CSV:  {step_path}")
    print(f"Episode CSV:   {ep_path}")
    if len(ep_df):
        print(f"Episode-level theorem1_satisfied mean: {ep_df['theorem1_satisfied'].mean():.3f}")
        print(f"Alert-level compliance_rate mean:      {ep_df['compliance_rate'].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, n_seeds=args.n_seeds, smoke=args.smoke)

