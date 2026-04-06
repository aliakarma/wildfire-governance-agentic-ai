#!/usr/bin/env python3
"""Experiment 11 — Train PPO-GOMDP agent.

Trains the PPO-GOMDP policy for 1000 episodes inside the GOMDP environment.
The governance constraint is NOT in the reward — it is enforced by the GOMDP
environment, confirming Theorem 1 (Policy-Agnostic Safety).

Paper reference: Table II, Section V-B (PPO-GOMDP).
Output: src/wildfire_governance/rl/checkpoints/ppo_gomdp_best.pt
        results/runs/<hash>/ppo_learning_curve.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wildfire_governance.rl.trainer import train
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")


def main(config_path: str, smoke: bool = False, use_pretrained: bool = False) -> None:
    from pathlib import Path as P
    ckpt = P("src/wildfire_governance/rl/checkpoints/ppo_gomdp_best.pt")
    if use_pretrained and ckpt.exists() and not smoke:
        logger.info("pretrained_checkpoint_found", path=str(ckpt))
        print(f"Pre-trained checkpoint found at {ckpt}. Skipping training.")
        print("To re-train from scratch, run without --use_pretrained.")
        return

    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        n_episodes = 1000
        n_uavs = cfg.simulation.uav.n_uavs
    except Exception:
        n_episodes, n_uavs = 1000, 20

    logger.info("ppo_training_start", n_episodes=n_episodes, use_pretrained=use_pretrained)
    history = train(n_episodes=n_episodes, n_uavs=n_uavs, smoke=smoke)

    # Save learning curve
    curve_path = out_dir / "ppo_learning_curve.csv"
    df = pd.DataFrame({
        "episode": list(range(len(history["episode_rewards"]))),
        "reward": history["episode_rewards"],
        "ld": history["episode_lds"],
        "compliance": history["compliance_rates"],
        "loss": history["policy_losses"],
    })
    df.to_csv(curve_path, index=False)
    logger.info("training_complete", curve_path=str(curve_path))
    print(f"Learning curve saved to {curve_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/ppo_training.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--use_pretrained", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke, args.use_pretrained)
