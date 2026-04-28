#!/usr/bin/env python3
"""Experiment (Missing #1): Multi-seed PPO training reproducibility.

Runs PPO-GOMDP training from scratch for seeds in [seed_start, seed_end]
and aggregates learning curves + final-episode reward distribution.

Usage:
  python experiments/exp1_multiseed_ppo_training.py --config configs/experiments/ppo_training.yaml --seed_start 0 --seed_end 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)


def _find_latest_curve_csv(results_root: Path) -> Path:
    # `11_ppo_training.py` writes to results/runs/<hash>/ppo_learning_curve.csv
    # We select the most recently modified such file under results/runs.
    candidates = sorted(
        results_root.glob("*/ppo_learning_curve.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No learning curves found under {results_root}")
    return candidates[0]


def main(
    config_path: str,
    seed_start: int = 0,
    seed_end: int = 4,
    out_dir: str | None = None,
    smoke: bool = False,
) -> None:
    results_root = Path("results/runs")
    run_hash = generate_run_hash(
        {
            "experiment": "exp1_multiseed_ppo_training",
            "config": str(config_path),
            "seed_start": int(seed_start),
            "seed_end": int(seed_end),
            "smoke": bool(smoke),
        }
    )
    exp_dir = Path(out_dir) if out_dir else (Path("results/experiments") / run_hash)
    exp_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(int(seed_start), int(seed_end) + 1))
    per_seed_curves: list[pd.DataFrame] = []
    final_rewards: list[dict] = []

    for seed in seeds:
        ckpt_path = exp_dir / f"checkpoints/ppo_gomdp_seed{seed}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("training_seed_start", seed=seed, checkpoint=str(ckpt_path))
        cmd = [
            sys.executable,
            "experiments/11_ppo_training.py",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--checkpoint_path",
            str(ckpt_path),
        ]
        if smoke:
            cmd.append("--smoke")
        # Train once; curve CSV will be written under results/runs/<hash>/...
        subprocess.run(cmd, check=True)

        curve_csv = _find_latest_curve_csv(results_root)
        df = pd.read_csv(curve_csv)
        df["train_seed"] = seed
        df["curve_source"] = str(curve_csv.as_posix())
        per_seed_curves.append(df)

        final_rewards.append(
            {
                "train_seed": seed,
                "final_episode": int(df["episode"].iloc[-1]),
                "final_reward": float(df["reward"].iloc[-1]),
                "best_reward": float(df["reward"].max()),
                "mean_reward": float(df["reward"].mean()),
                "std_reward": float(df["reward"].std(ddof=1)) if len(df) > 1 else 0.0,
                "curve_csv": str(curve_csv.as_posix()),
                "checkpoint": str(ckpt_path.as_posix()),
            }
        )

    curves = pd.concat(per_seed_curves, ignore_index=True) if per_seed_curves else pd.DataFrame()
    finals = pd.DataFrame(final_rewards)

    curves_path = exp_dir / "ppo_multiseed_learning_curves.csv"
    finals_path = exp_dir / "ppo_multiseed_final_rewards.csv"
    curves.to_csv(curves_path, index=False)
    finals.to_csv(finals_path, index=False)

    # Variance summary by episode (mean/std across seeds).
    if len(curves):
        grouped = curves.groupby("episode", as_index=False)["reward"].agg(["mean", "std"]).reset_index()
        grouped.columns = ["episode", "reward_mean", "reward_std"]
        summary_path = exp_dir / "ppo_multiseed_reward_mean_std_by_episode.csv"
        grouped.to_csv(summary_path, index=False)

        # Episode 500 check (if present).
        ep_target = 500
        if (curves["episode"] == ep_target).any():
            ep500 = curves[curves["episode"] == ep_target]
            mu = float(ep500["reward"].mean())
            sigma = float(ep500["reward"].std(ddof=1)) if len(ep500) > 1 else 0.0
            max_dev = float(np.max(np.abs(ep500["reward"] - mu))) if len(ep500) else float("nan")
            (exp_dir / "ppo_multiseed_episode500_check.txt").write_text(
                f"episode={ep_target}\nmean={mu}\nstd={sigma}\nmax_abs_dev={max_dev}\ncriterion=max_abs_dev<=2*std => {max_dev <= 2*sigma if sigma>0 else 'n/a'}\n"
            )

    print("\n=== Multi-seed PPO Training (Reproducibility) ===")
    print(f"Outputs in: {exp_dir}")
    print(f"- Curves (all seeds): {curves_path}")
    print(f"- Final rewards:      {finals_path}")
    if len(curves):
        print(f"- Mean±std by episode:{exp_dir / 'ppo_multiseed_reward_mean_std_by_episode.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/ppo_training.yaml")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.seed_start, args.seed_end, args.out_dir, args.smoke)

