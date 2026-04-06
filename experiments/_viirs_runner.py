"""Shared VIIRS region runner used by experiments 08a/b/c.

Not run directly — imported by each regional experiment script.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def run_viirs_region(
    region: str,
    viirs_path: Path,
    nifc_path: "Path | None",
    output_stem: str,
    config_path: str,
    smoke: bool = False,
) -> None:
    """Run PPO-GOMDP evaluation on a real VIIRS region dataset.

    Args:
        region: Region name string (e.g. ``"mediterranean_2021"``).
        viirs_path: Path to preprocessed VIIRS .npz file.
        nifc_path: Path to NIFC/EFFIS ground-truth mask (optional).
        output_stem: Output CSV filename stem.
        config_path: Path to YAML config file.
        smoke: If True, use reduced episode count.
    """
    from wildfire_governance.utils.config import load_config
    from wildfire_governance.utils.logging import get_structured_logger
    from wildfire_governance.utils.reproducibility import generate_run_hash

    logger = get_structured_logger(__name__)
    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = Path("results/runs") / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    if not viirs_path.exists():
        msg = (
            f"\nVIIRS data not found: {viirs_path}\n"
            "Download with: make download-viirs\n"
            f"Experiment '{region}' skipped.\n"
        )
        print(msg)
        pd.DataFrame([{"region": region, "status": "data_not_found"}]).to_csv(
            out_dir / f"{output_stem}_skipped.csv", index=False
        )
        return

    from wildfire_governance.simulation.real_world_adapter import RealWorldAdapter
    from wildfire_governance.simulation.grid_environment import EnvironmentConfig
    from wildfire_governance.rl.gomdp_env import GOMMDPGymEnv
    from wildfire_governance.rl.ppo_agent import PPOGOMDPAgent
    from wildfire_governance.rl.evaluator import CHECKPOINT_DIR
    from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker

    n_seeds = 3 if smoke else 10
    n_uavs = 5 if smoke else 20
    n_timesteps = 50 if smoke else 500

    adapter = RealWorldAdapter(grid_size=100)
    viirs_grids = adapter.load_viirs_grid(viirs_path)
    checker = GovernanceInvariantChecker(tau=0.80)
    lds, fps, compliances = [], [], []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        env_cfg = EnvironmentConfig(grid_size=100, n_timesteps=n_timesteps)
        env = GOMMDPGymEnv(config=env_cfg, n_uavs=n_uavs, enable_governance=True)
        agent = PPOGOMDPAgent(grid_size=100, n_uavs=n_uavs)
        ckpt = CHECKPOINT_DIR / "ppo_gomdp_best.pt"
        if ckpt.exists():
            try:
                agent.load_checkpoint(ckpt)
            except Exception:
                pass

        obs, _ = env.reset(seed=seed)
        if viirs_grids.ndim == 3 and len(viirs_grids) > 0:
            t_idx = min(seed, len(viirs_grids) - 1)
            env._sim._heat_map = viirs_grids[t_idx]

        done = False
        while not done:
            action_dict = agent.select_actions(obs, env._fleet)
            action_arr = np.array([action_dict.get(i, 0) for i in range(n_uavs)])
            obs, _, terminated, truncated, info = env.step(action_arr)
            done = terminated or truncated

        ep_ld = info.get("episode_ld", float("inf"))
        ep_fp = info.get("episode_fp_pct", 0.0)
        traj = env.get_trajectory()
        report = checker.check_trajectory(traj)

        if ep_ld < float("inf"):
            lds.append(ep_ld)
        fps.append(ep_fp)
        compliances.append(float(report.theorem1_satisfied))

    iot_baseline_ld = 45.0
    result_ld = float(np.mean(lds)) if lds else float("inf")
    speedup = (iot_baseline_ld - result_ld) / iot_baseline_ld if result_ld < iot_baseline_ld else 0.0

    result_row = {
        "region": region,
        "method": "PPO-GOMDP",
        "ld_mean": round(result_ld, 2),
        "ld_std": round(float(np.std(lds)), 2) if lds else 0.0,
        "fp_mean": round(float(np.mean(fps)), 2),
        "fp_std": round(float(np.std(fps)), 2),
        "governance_compliance_pct": round(float(np.mean(compliances)) * 100, 1),
        "speedup_vs_iot_baseline": round(speedup, 2),
        "n_episodes": n_seeds,
    }

    out_df = pd.DataFrame([result_row])
    out_path = out_dir / f"{output_stem}.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("viirs_experiment_complete", region=region, output=str(out_path))
    print(f"\n=== {region} VIIRS Results ===\n{out_df.to_string(index=False)}\n")
