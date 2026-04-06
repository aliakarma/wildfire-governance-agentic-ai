"""PPO-GOMDP training loop — 1000-episode training on the GOMDP environment.

FIX M6: Changed best-checkpoint criterion from ep_ld < best_ld to:
    compliance == 1.0  AND  ep_ld < best_compliant_ld
  A policy achieving low Ld via governance violations would previously be
  saved as "best" — now governance compliance is a hard prerequisite.

FIX Issue 1 (training loop): The training loop now uses the new buffer API
  (store_transition / update_from_buffer) so that log_probs_old are the
  actual log-probs of the actions taken, not freshly sampled random ones.
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

    # FIX M6: track best COMPLIANT Ld separately
    best_compliant_ld = float("inf")

    logger.info("ppo_training_start", n_episodes=n_episodes, n_uavs=n_uavs, seed=seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        agent.clear_buffer()  # FIX Issue 1: start with empty rollout buffer
        done = False
        total_reward = 0.0

        while not done:
            # FIX Issue 1: select_actions now returns (allocation, log_probs, value)
            action_dict, log_probs, value = agent.select_actions(obs, env._fleet)
            action_arr = np.array([action_dict.get(i, 0) for i in range(n_uavs)])
            next_obs, reward, terminated, truncated, info = env.step(action_arr)

            # FIX Issue 1: store actual log-probs (not re-sampled random ones)
            action_list = [action_dict.get(i, 0) for i in range(n_uavs)]
            agent.store_transition(
                obs=obs.copy(),
                actions=action_list,
                log_probs=log_probs,
                value=value,
                reward=reward,
                done=(terminated or truncated),
            )

            obs = next_obs
            total_reward += reward
            done = terminated or truncated

        # PPO update using rollout buffer with correct log-probs
        loss = agent.update_from_buffer()

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

        # FIX M6: save best checkpoint only when compliance == 1.0 AND Ld improves
        if not smoke:
            is_compliant = (compliance == 1.0)
            if is_compliant and ep_ld < best_compliant_ld:
                best_compliant_ld = ep_ld
                CHECKPOINT_DIR.mkdir(exist_ok=True)
                agent.save_checkpoint(CHECKPOINT_DIR / "ppo_gomdp_best.pt")
                logger.info(
                    "checkpoint_saved_best",
                    episode=ep,
                    ld=round(ep_ld, 2),
                    compliance="100%",
                )

        if (ep + 1) % checkpoint_every == 0 and not smoke:
            agent.save_checkpoint(CHECKPOINT_DIR / "ppo_gomdp_final.pt")

    if not smoke:
        agent.save_checkpoint(CHECKPOINT_DIR / "ppo_gomdp_final.pt")

    compliant_lds = [
        x for x in history["episode_lds"] if x < float("inf")
    ]
    logger.info(
        "ppo_training_complete",
        mean_ld=round(float(np.mean(compliant_lds)), 2) if compliant_lds else "inf",
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
