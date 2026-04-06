"""PPO-GOMDP training loop — 1000-episode training on the GOMDP environment.

Usage:
    python -m wildfire_governance.rl.trainer  (or via make train-ppo)

The pre-trained checkpoint at src/wildfire_governance/rl/checkpoints/ppo_gomdp_best.pt
was produced with the default configuration below and achieves Ld=15.1 ± 1.1 steps
with 100% governance compliance (Table II in paper).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker
from wildfire_governance.rl.gomdp_env import GOMMDPGymEnv
from wildfire_governance.rl.ppo_agent import PPOGOMDPAgent
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import set_global_seed

logger = get_structured_logger(__name__)

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def train(
    n_episodes: int = 1000,
    n_uavs: int = 20,
    grid_size: int = 100,
    seed: int = 42,
    checkpoint_every: int = 50,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Train a PPO-GOMDP agent for *n_episodes* episodes.

    Args:
        n_episodes: Training episodes (paper default: 1000).
        n_uavs: UAV fleet size (paper default: 20).
        grid_size: Grid side length (paper default: 100).
        seed: Master random seed.
        checkpoint_every: Save checkpoint every N episodes.
        smoke: If True, use 2 episodes × 100 steps for quick sanity check.

    Returns:
        Dict with training history: episode_rewards, episode_lds, compliance_rates.
    """
    set_global_seed(seed)
    if smoke:
        n_episodes = 2
        grid_size = 10

    from wildfire_governance.simulation.grid_environment import EnvironmentConfig
    env_config = EnvironmentConfig(
        grid_size=grid_size,
        n_timesteps=100 if smoke else 3000,
    )
    env = GOMMDPGymEnv(config=env_config, n_uavs=n_uavs, enable_governance=True)
    agent = PPOGOMDPAgent(grid_size=grid_size, n_uavs=n_uavs)
    checker = GovernanceInvariantChecker(tau=0.80)

    history: Dict[str, List] = {
        "episode_rewards": [],
        "episode_lds": [],
        "compliance_rates": [],
        "policy_losses": [],
    }
    best_ld = float("inf")

    logger.info("ppo_training_start", n_episodes=n_episodes, n_uavs=n_uavs, seed=seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
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

        # PPO update
        loss = agent.update(ep_obs, ep_actions, ep_rewards, ep_dones)

        # Governance compliance
        trajectory = env.get_trajectory()
        report = checker.check_trajectory(trajectory)
        compliance = report.compliance_rate

        ep_ld = info.get("episode_ld", float("inf"))
        history["episode_rewards"].append(total_reward)
        history["episode_lds"].append(ep_ld)
        history["compliance_rates"].append(compliance)
        history["policy_losses"].append(loss)

        if ep % 50 == 0:
            logger.info(
                "ppo_training_episode",
                episode=ep,
                reward=round(total_reward, 3),
                ld=round(ep_ld, 1) if ep_ld < float("inf") else "inf",
                compliance=f"{compliance:.1%}",
                loss=round(loss, 4),
            )

        # Save best checkpoint
        if ep_ld < best_ld and not smoke:
            best_ld = ep_ld
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            agent.save_checkpoint(CHECKPOINT_DIR / "ppo_gomdp_best.pt")

        if (ep + 1) % checkpoint_every == 0 and not smoke:
            agent.save_checkpoint(CHECKPOINT_DIR / "ppo_gomdp_final.pt")

    if not smoke:
        agent.save_checkpoint(CHECKPOINT_DIR / "ppo_gomdp_final.pt")

    logger.info(
        "ppo_training_complete",
        mean_ld=round(float(np.mean([x for x in history["episode_lds"] if x < float("inf")])), 2),
        mean_compliance=round(float(np.mean(history["compliance_rates"])) * 100, 1),
    )
    return history


def main() -> None:
    """CLI entry-point for PPO-GOMDP training."""
    parser = argparse.ArgumentParser(description="Train PPO-GOMDP agent")
    parser.add_argument("--config", type=str, default="configs/experiments/ppo_training.yaml")
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Quick 2-episode sanity check")
    parser.add_argument("--use_pretrained", action="store_true", help="Skip training; use checkpoint")
    args = parser.parse_args()

    if args.use_pretrained:
        ckpt = CHECKPOINT_DIR / "ppo_gomdp_best.pt"
        if ckpt.exists():
            print(f"Pre-trained checkpoint found at {ckpt}. Skipping training.")
            print("Run: make eval-ppo to evaluate the pre-trained agent.")
            return
        print("No pre-trained checkpoint found. Starting training...")

    history = train(n_episodes=args.n_episodes, seed=args.seed, smoke=args.smoke)
    out = Path("results/runs") / f"ppo_training_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "training_history.json", "w") as fh:
        json.dump(
            {k: [float(v) for v in vals if v is not None] for k, vals in history.items()},
            fh, indent=2,
        )
    print(f"Training history saved to {out}/training_history.json")


if __name__ == "__main__":
    main()
