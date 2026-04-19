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
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode
from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker
from wildfire_governance.rl.gomdp_env import GOMMDPGymEnv
from wildfire_governance.rl.ppo_agent import PPOGOMDPAgent
from wildfire_governance.simulation.grid_environment import EnvironmentConfig
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash, set_global_seed

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")
CHECKPOINT_PATH = Path("src/wildfire_governance/rl/checkpoints/ppo_gomdp_best.pt")
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_training_episode(
    agent: PPOGOMDPAgent,
    env: GOMMDPGymEnv,
    checker: GovernanceInvariantChecker,
    n_uavs: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Run one PPO training episode and return scalar metrics."""
    obs, _ = env.reset(seed=seed)
    ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []
    done = False
    total_reward = 0.0

    while not done:
        action_dict = agent.select_actions(obs, env._fleet)
        action_arr = np.array([action_dict.get(i, 0) for i in range(n_uavs)])
        next_obs, reward, terminated, truncated, info = env.step(action_arr)
        ep_obs.append(obs.copy())
        ep_actions.append(action_dict)
        ep_rewards.append(reward)
        ep_dones.append(terminated or truncated)
        obs = next_obs
        total_reward += reward
        done = terminated or truncated

    loss = agent.update(ep_obs, ep_actions, ep_rewards, ep_dones)
    trajectory = env.get_trajectory()
    report = checker.check_trajectory(trajectory)
    ep_ld = info.get("episode_ld", float("inf"))
    compliance = report.compliance_rate
    return float(total_reward), float(ep_ld), float(compliance), float(loss)


def main(config_path: str, smoke: bool = False, use_pretrained: bool = False) -> None:
    if use_pretrained and CHECKPOINT_PATH.exists():
        logger.info("pretrained_checkpoint_found", path=str(CHECKPOINT_PATH))
        print(f"Pre-trained checkpoint found at {CHECKPOINT_PATH}. Skipping training.")
        print("To re-train from scratch, run without --use_pretrained.")
        return

    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        num_episodes = int(cfg.ppo.n_episodes)
        n_uavs = cfg.simulation.uav.n_uavs
        n_timesteps = cfg.simulation.n_timesteps
        lr = float(cfg.ppo.lr)
        clip_ratio = float(cfg.ppo.clip_ratio)
        entropy_coeff = float(cfg.ppo.entropy_coeff)
        gamma = float(cfg.ppo.gamma)
        n_epochs = int(cfg.ppo.n_epochs)
    except Exception:
        num_episodes, n_uavs, n_timesteps = 1000, 20, 3000
        lr, clip_ratio, entropy_coeff, gamma, n_epochs = 3e-4, 0.2, 0.01, 0.99, 4

    try:
        grid_size = int(cfg.simulation.grid_size)
    except Exception:
        grid_size = 100

    try:
        seed = int(cfg.seed)
    except Exception:
        seed = 42

    if smoke:
        num_episodes = min(num_episodes, 2)
        grid_size = 10
        n_timesteps = 100
    else:
        num_episodes = max(num_episodes, 500)

    set_global_seed(seed)
    env_config = EnvironmentConfig(grid_size=grid_size, n_timesteps=n_timesteps)
    env = GOMMDPGymEnv(config=env_config, n_uavs=n_uavs, enable_governance=True)
    agent = PPOGOMDPAgent(
        grid_size=grid_size,
        n_uavs=n_uavs,
        lr=lr,
        clip_ratio=clip_ratio,
        entropy_coeff=entropy_coeff,
        gamma=gamma,
        n_epochs=n_epochs,
    )
    checker = GovernanceInvariantChecker(tau=0.80)

    best_reward = -float("inf")
    reward_history = []

    history = {
        "episode_rewards": [],
        "episode_lds": [],
        "compliance_rates": [],
        "policy_losses": [],
    }

    logger.info("ppo_training_start", n_episodes=num_episodes, use_pretrained=use_pretrained)

    for ep in range(num_episodes):
        reward, ep_ld, compliance, loss = run_training_episode(
            agent=agent,
            env=env,
            checker=checker,
            n_uavs=n_uavs,
            seed=seed + ep,
        )

        reward_history.append(reward)
        history["episode_rewards"].append(reward)
        history["episode_lds"].append(ep_ld)
        history["compliance_rates"].append(compliance)
        history["policy_losses"].append(loss)

        if reward > best_reward:
            best_reward = reward
            agent.save_checkpoint(str(CHECKPOINT_PATH))
            print(f"[INFO] New best model saved with reward: {reward:.4f}")

        if ep % 50 == 0:
            logger.info(
                "ppo_training_episode",
                episode=ep,
                reward=round(reward, 3),
                ld=round(ep_ld, 1) if ep_ld < float("inf") else "inf",
                compliance=f"{compliance:.1%}",
                loss=round(loss, 4),
            )

    print("\n=== TRAINING SUMMARY ===")
    print(f"Best reward: {best_reward:.4f}")
    print(f"Mean reward: {np.mean(reward_history):.4f}")
    print(f"Std reward: {np.std(reward_history):.4f}")

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError("Checkpoint was not created. Training failed.")

    print(f"Checkpoint saved at: {os.path.abspath(str(CHECKPOINT_PATH))}")

    test_result = run_episode(
        seed=0,
        config_name="gomdp",
        enable_governance=True,
        enable_hitl=True,
        enable_blockchain=True,
        enable_verification=True,
        enable_coordination=True,
    )

    print("\n=== QUICK EVAL ===")
    print(f"Detection latency: {getattr(test_result, 'latency', 'N/A')}")
    print(f"False positives: {getattr(test_result, 'fp_pct', 'N/A')}")

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
