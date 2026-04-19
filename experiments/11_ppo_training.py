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
import copy
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# FIX [HIGH]: run_episode import moved inside a lazy helper so a missing
# experiments.utils.runner module does NOT crash the entire script at import time.
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
# FIX [CRITICAL]: mkdir() moved into main() so it only runs when the script is
# actually executed, not at import time (which breaks read-only environments and
# test collection when CWD != project root).


def _copy_obs(obs):
    """Return a safe independent copy of an observation.

    FIX [HIGH]: env.reset() in a multi-UAV GOMDP may return a dict whose values
    are numpy arrays.  Plain dict.copy() is a *shallow* copy — the arrays would
    be shared across timesteps, silently corrupting the replay buffer.
    numpy arrays get np.copy(); everything else falls back to copy.deepcopy().
    """
    if isinstance(obs, np.ndarray):
        return obs.copy()
    if isinstance(obs, dict):
        return {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v))
                for k, v in obs.items()}
    return copy.deepcopy(obs)


def _format_ld(ep_ld: float) -> str:
    """Format detection-latency for logging, guarding against inf *and* NaN.

    FIX [MEDIUM]: the original round(...) call raises ValueError on NaN because
    math.isfinite() covers both cases cleanly.
    """
    if math.isfinite(ep_ld):
        return str(round(ep_ld, 1))
    return "inf" if math.isinf(ep_ld) else "nan"


def _run_quick_eval() -> None:
    """Run a single post-training evaluation episode and print the result.

    FIX [HIGH]: the import is deferred so a missing experiments.utils.runner
    module does not abort the script before training begins.  The call is also
    wrapped in a try/except so an unexpected signature mismatch or runtime error
    in the eval step does not destroy an otherwise successful training run.
    """
    try:
        from experiments.utils.runner import run_episode  # noqa: PLC0415
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
        print(f"False positives:   {getattr(test_result, 'fp_pct', 'N/A')}")
    except ImportError:
        logger.warning("quick_eval_skipped", reason="experiments.utils.runner not found")
        print("\n[WARN] Quick eval skipped — experiments.utils.runner not importable.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("quick_eval_failed", error=str(exc))
        print(f"\n[WARN] Quick eval failed: {exc}")


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
        # action_arr: env expects an array; ep_actions stores dicts for agent.update().
        # FIX [LOW]: documented explicitly so the dict/array duality is not confusing.
        action_arr = np.array([action_dict.get(i, 0) for i in range(n_uavs)])
        next_obs, reward, terminated, truncated, info = env.step(action_arr)

        # FIX [HIGH]: use _copy_obs() instead of obs.copy() to handle dict obs safely.
        ep_obs.append(_copy_obs(obs))
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


def _load_hyperparams(cfg, smoke: bool) -> dict:
    """Extract hyperparameters from config with explicit fallback logging.

    FIX [HIGH]: the original broad ``except Exception`` silently swallowed any
    config error.  We now log a structured warning whenever we fall back to
    defaults so the researcher knows the config was not read correctly.
    """
    defaults = dict(
        num_episodes=1000,
        n_uavs=20,
        n_timesteps=3000,
        lr=3e-4,
        clip_ratio=0.2,
        entropy_coeff=0.01,
        gamma=0.99,
        n_epochs=4,
        grid_size=100,
        seed=42,
    )

    params = {}
    fields = {
        "num_episodes":  lambda: int(cfg.ppo.n_episodes),
        "n_uavs":        lambda: int(cfg.simulation.uav.n_uavs),
        "n_timesteps":   lambda: int(cfg.simulation.n_timesteps),
        "lr":            lambda: float(cfg.ppo.lr),
        "clip_ratio":    lambda: float(cfg.ppo.clip_ratio),
        "entropy_coeff": lambda: float(cfg.ppo.entropy_coeff),
        "gamma":         lambda: float(cfg.ppo.gamma),
        "n_epochs":      lambda: int(cfg.ppo.n_epochs),
        "grid_size":     lambda: int(cfg.simulation.grid_size),
        "seed":          lambda: int(cfg.seed),
    }

    for key, getter in fields.items():
        try:
            params[key] = getter()
        except Exception as exc:  # noqa: BLE001
            params[key] = defaults[key]
            logger.warning(
                "config_field_fallback",
                field=key,
                default=defaults[key],
                reason=str(exc),
            )

    # Smoke overrides
    if smoke:
        params["num_episodes"] = min(params["num_episodes"], 2)
        params["grid_size"] = 10
        params["n_timesteps"] = 100
    else:
        original = params["num_episodes"]
        params["num_episodes"] = max(params["num_episodes"], 500)
        # FIX [MEDIUM]: warn if the config value was silently overridden.
        if params["num_episodes"] != original:
            logger.warning(
                "num_episodes_override",
                config_value=original,
                effective_value=params["num_episodes"],
                reason="minimum 500 episodes enforced for non-smoke runs",
            )

    return params


def main(config_path: str, smoke: bool = False, use_pretrained: bool = False) -> None:
    # FIX [CRITICAL]: mkdir() moved here — only runs when main() is called.
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if use_pretrained and CHECKPOINT_PATH.exists():
        logger.info("pretrained_checkpoint_found", path=str(CHECKPOINT_PATH))
        print(f"Pre-trained checkpoint found at {CHECKPOINT_PATH}. Skipping training.")
        print("To re-train from scratch, run without --use_pretrained.")
        return

    cfg = load_config(config_path)

    # FIX [MEDIUM]: extract params *before* computing the run hash so that smoke
    # overrides (grid_size, n_timesteps, num_episodes) are reflected in the hash.
    params = _load_hyperparams(cfg, smoke)
    run_hash = generate_run_hash({**vars(cfg), **params})
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(params["seed"])
    env_config = EnvironmentConfig(
        grid_size=params["grid_size"],
        n_timesteps=params["n_timesteps"],
    )
    env = GOMMDPGymEnv(
        config=env_config,
        n_uavs=params["n_uavs"],
        enable_governance=True,
    )
    agent = PPOGOMDPAgent(
        grid_size=params["grid_size"],
        n_uavs=params["n_uavs"],
        lr=params["lr"],
        clip_ratio=params["clip_ratio"],
        entropy_coeff=params["entropy_coeff"],
        gamma=params["gamma"],
        n_epochs=params["n_epochs"],
    )
    checker = GovernanceInvariantChecker(tau=0.80)

    best_reward = -float("inf")
    history = {
        "episode_rewards":  [],
        "episode_lds":      [],
        "compliance_rates": [],
        "policy_losses":    [],
    }

    logger.info(
        "ppo_training_start",
        n_episodes=params["num_episodes"],
        use_pretrained=use_pretrained,
    )

    # FIX [CRITICAL]: env.close() guaranteed via try/finally so resources are
    # always released even if an episode raises an unhandled exception.
    try:
        for ep in range(params["num_episodes"]):
            reward, ep_ld, compliance, loss = run_training_episode(
                agent=agent,
                env=env,
                checker=checker,
                n_uavs=params["n_uavs"],
                seed=params["seed"] + ep,
            )

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
                    # FIX [MEDIUM]: _format_ld() guards against NaN as well as inf.
                    ld=_format_ld(ep_ld),
                    compliance=f"{compliance:.1%}",
                    loss=round(loss, 4),
                )
    finally:
        # FIX [CRITICAL]: always close the environment.
        env.close()

    # FIX [CRITICAL]: removed redundant reward_history list; use
    # history["episode_rewards"] directly (identical data, no duplication).
    rewards = history["episode_rewards"]
    print("\n=== TRAINING SUMMARY ===")
    print(f"Best reward: {best_reward:.4f}")
    print(f"Mean reward: {np.mean(rewards):.4f}")
    print(f"Std reward:  {np.std(rewards):.4f}")

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError("Checkpoint was not created. Training failed.")

    print(f"Checkpoint saved at: {os.path.abspath(str(CHECKPOINT_PATH))}")

    # FIX [HIGH]: eval wrapped in _run_quick_eval() — import errors and signature
    # mismatches are handled gracefully without crashing the training summary.
    _run_quick_eval()

    # Save learning curve
    curve_path = out_dir / "ppo_learning_curve.csv"
    df = pd.DataFrame({
        "episode":    list(range(len(rewards))),
        "reward":     rewards,
        "ld":         history["episode_lds"],
        "compliance": history["compliance_rates"],
        "loss":       history["policy_losses"],
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