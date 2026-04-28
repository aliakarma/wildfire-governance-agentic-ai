"""PPO-GOMDP evaluation script — measures Ld, Fp, governance compliance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker
from wildfire_governance.metrics.detection_metrics import aggregate_metrics, EpisodeMetrics
from wildfire_governance.rl.gomdp_env import GOMMDPGymEnv
from wildfire_governance.rl.ppo_agent import PPOGOMDPAgent
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import set_global_seed

logger = get_structured_logger(__name__)
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def _load_checkpoint_if_compatible(
    agent: PPOGOMDPAgent,
    checkpoint_path: Path,
) -> bool:
    """Load a checkpoint only when its tensor shapes match the current agent.

    Returns:
        True when the checkpoint was loaded, False when a mismatch or load
        failure was detected and the agent was left at its random init.
    """
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "checkpoint_load_failed",
            path=str(checkpoint_path),
            reason=str(exc),
        )
        return False

    if not isinstance(checkpoint, dict) or "policy_state_dict" not in checkpoint:
        logger.warning(
            "checkpoint_format_unexpected",
            path=str(checkpoint_path),
        )
        return False

    current_policy_state = agent.policy.state_dict()
    current_value_state = agent.value_net.state_dict()
    loaded_policy_state = checkpoint["policy_state_dict"]
    loaded_value_state = checkpoint.get("value_state_dict", {})

    for key, current_tensor in current_policy_state.items():
        loaded_tensor = loaded_policy_state.get(key)
        if loaded_tensor is None or tuple(loaded_tensor.shape) != tuple(current_tensor.shape):
            logger.warning(
                "checkpoint_shape_mismatch",
                path=str(checkpoint_path),
                component="policy",
                key=key,
                expected_shape=tuple(current_tensor.shape),
                loaded_shape=None if loaded_tensor is None else tuple(loaded_tensor.shape),
            )
            return False

    for key, current_tensor in current_value_state.items():
        loaded_tensor = loaded_value_state.get(key)
        if loaded_tensor is None or tuple(loaded_tensor.shape) != tuple(current_tensor.shape):
            logger.warning(
                "checkpoint_shape_mismatch",
                path=str(checkpoint_path),
                component="value",
                key=key,
                expected_shape=tuple(current_tensor.shape),
                loaded_shape=None if loaded_tensor is None else tuple(loaded_tensor.shape),
            )
            return False

    try:
        agent.load_state_dict(checkpoint)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "checkpoint_load_failed",
            path=str(checkpoint_path),
            reason=str(exc),
        )
        return False

    logger.info("checkpoint_loaded", path=str(checkpoint_path))
    return True


def evaluate(
    n_seeds: int = 20,
    n_uavs: int = 20,
    grid_size: int = 100,
    use_pretrained: bool = True,
    enable_governance: bool = True,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Evaluate PPO-GOMDP over n_seeds episodes and report aggregated metrics.

    Args:
        n_seeds: Number of evaluation seeds (paper default: 20).
        n_uavs: UAV fleet size.
        grid_size: Grid side length.
        use_pretrained: If True, load the pre-trained checkpoint.
        enable_governance: If False, runs without GOMDP (CMDP comparison).
        smoke: If True, use 2 seeds × 100 steps.

    Returns:
        Dict with ld_mean, ld_std, fp_mean, fp_std, governance_compliance_pct.
    """
    if smoke:
        n_seeds = 2
        grid_size = 10

    from wildfire_governance.simulation.grid_environment import EnvironmentConfig
    env_config = EnvironmentConfig(
        grid_size=grid_size,
        n_timesteps=100 if smoke else 3000,
    )

    agent = PPOGOMDPAgent(grid_size=grid_size, n_uavs=n_uavs)
    if use_pretrained:
        ckpt = CHECKPOINT_DIR / "ppo_gomdp_best.pt"
        try:
            if not _load_checkpoint_if_compatible(agent, ckpt):
                logger.warning(
                    "checkpoint_compatibility_fallback",
                    path=str(ckpt),
                    reason="shape mismatch or load failure; using random init",
                )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"PPO checkpoint not found at {ckpt}. "
                "Download or provide checkpoint before evaluation."
            ) from exc

    checker = GovernanceInvariantChecker(tau=0.80)
    episode_metrics_list: List[EpisodeMetrics] = []
    all_trajectories = []

    for seed in range(n_seeds):
        set_global_seed(seed)
        env = GOMMDPGymEnv(
            config=env_config, n_uavs=n_uavs, enable_governance=enable_governance
        )
        obs, _ = env.reset(seed=seed)
        done = False

        while not done:
            action_dict = agent.select_actions(obs, env._fleet)
            action_arr = np.array([action_dict.get(i, 0) for i in range(n_uavs)])
            obs, _, terminated, truncated, info = env.step(action_arr)
            done = terminated or truncated

        trajectory = env.get_trajectory()
        all_trajectories.append(trajectory)

        ep_ld = info.get("episode_ld", float("inf"))
        ep_fp = info.get("episode_fp_pct", 0.0)
        compliance = checker.check_trajectory(trajectory)

        episode_metrics_list.append(
            EpisodeMetrics(
                detection_latency=ep_ld,
                false_alert_rate=ep_fp,
                governance_compliant=compliance.theorem1_satisfied,
            )
        )

    agg = aggregate_metrics(episode_metrics_list)
    return {
        "ld_mean": round(agg.ld_mean, 2),
        "ld_std": round(agg.ld_std, 2),
        "fp_mean": round(agg.fp_mean, 2),
        "fp_std": round(agg.fp_std, 2),
        "governance_compliance_pct": round(agg.governance_compliance_pct, 1),
        "n_seeds": n_seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO-GOMDP agent")
    parser.add_argument("--config", type=str, default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--use_pretrained", action="store_true", default=True)
    parser.add_argument("--no_governance", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    results = evaluate(
        n_seeds=args.n_seeds,
        use_pretrained=args.use_pretrained,
        enable_governance=not args.no_governance,
        smoke=args.smoke,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
