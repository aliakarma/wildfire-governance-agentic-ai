#!/usr/bin/env python3
"""Experiment (Missing #3): PPO training-seed ablation vs Greedy baseline.

Assumes you have trained multiple PPO-GOMDP checkpoints (e.g., via Exp1).
For each PPO checkpoint (training seed), evaluate PPO-GOMDP and Greedy-GOMDP
across evaluation seeds and report mean±std of Ld improvement (%).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode
from wildfire_governance.rl.evaluator import evaluate as eval_ppo
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)


def _eval_greedy(
    n_seeds: int,
    *,
    n_uavs: int,
    n_timesteps: int,
) -> dict:
    lds = []
    for seed in range(int(n_seeds)):
        r = run_episode(
            seed=seed,
            config_name="greedy_gomdp",
            n_uavs=n_uavs,
            n_timesteps=n_timesteps,
            enable_governance=True,
            enable_hitl=True,
            enable_blockchain=True,
            enable_verification=True,
            enable_coordination=True,
        )
        if r.ld < float("inf"):
            lds.append(float(r.ld))
    return {"ld_mean": float(np.mean(lds)) if lds else float("inf"), "ld_values": lds}


def main(
    checkpoints_dir: str,
    n_eval_seeds: int = 20,
    n_uavs: int = 20,
    n_timesteps: int = 3000,
    out_dir: str | None = None,
    smoke: bool = False,
) -> None:
    ckpt_dir = Path(checkpoints_dir)
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No .pt checkpoints found in {ckpt_dir}")

    if smoke:
        n_eval_seeds = min(int(n_eval_seeds), 2)
        n_uavs = 5
        n_timesteps = 100

    run_hash = generate_run_hash(
        {
            "experiment": "exp3_multiseed_rl_ablation",
            "checkpoints_dir": str(ckpt_dir),
            "n_eval_seeds": int(n_eval_seeds),
            "n_uavs": int(n_uavs),
            "n_timesteps": int(n_timesteps),
            "smoke": bool(smoke),
        }
    )
    out = Path(out_dir) if out_dir else (Path("results/experiments") / run_hash)
    out.mkdir(parents=True, exist_ok=True)

    greedy = _eval_greedy(n_eval_seeds, n_uavs=n_uavs, n_timesteps=n_timesteps)
    greedy_ld = greedy["ld_mean"]

    rows = []
    improvements = []

    for ckpt in ckpts:
        logger.info("evaluating_checkpoint", checkpoint=str(ckpt))
        ppo_metrics = eval_ppo(
            n_seeds=int(n_eval_seeds),
            n_uavs=int(n_uavs),
            grid_size=10 if smoke else 100,
            use_pretrained=True,
            checkpoint_path=str(ckpt),
            enable_governance=True,
            smoke=smoke,
        )
        ppo_ld = float(ppo_metrics["ld_mean"])
        if np.isfinite(greedy_ld) and greedy_ld > 0:
            improvement_pct = (greedy_ld - ppo_ld) / greedy_ld * 100.0
        else:
            improvement_pct = float("nan")
        improvements.append(improvement_pct)
        rows.append(
            {
                "checkpoint": str(ckpt.as_posix()),
                "ppo_ld_mean": ppo_ld,
                "greedy_ld_mean": greedy_ld,
                "ld_improvement_pct": improvement_pct,
                "n_eval_seeds": int(n_eval_seeds),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = out / "exp3_multiseed_rl_ablation.csv"
    df.to_csv(out_csv, index=False)

    imp_arr = np.array(improvements, dtype=float)
    mean_imp = float(np.nanmean(imp_arr)) if len(imp_arr) else float("nan")
    std_imp = float(np.nanstd(imp_arr, ddof=1)) if np.sum(np.isfinite(imp_arr)) > 1 else 0.0

    (out / "exp3_summary.txt").write_text(
        f"n_checkpoints={len(ckpts)}\n"
        f"n_eval_seeds={int(n_eval_seeds)}\n"
        f"greedy_ld_mean={greedy_ld}\n"
        f"ld_improvement_pct_mean={mean_imp}\n"
        f"ld_improvement_pct_std={std_imp}\n"
    )

    print("\n=== Multi-seed RL Ablation (PPO vs Greedy) ===")
    print(f"Outputs in: {out}")
    print(f"Per-checkpoint CSV: {out_csv}")
    print(f"Improvement mean±std: {mean_imp:.2f}% ± {std_imp:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", required=True, help="Directory containing PPO .pt checkpoints.")
    parser.add_argument("--n_eval_seeds", type=int, default=20)
    parser.add_argument("--n_uavs", type=int, default=20)
    parser.add_argument("--n_timesteps", type=int, default=3000)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(
        checkpoints_dir=args.checkpoints_dir,
        n_eval_seeds=args.n_eval_seeds,
        n_uavs=args.n_uavs,
        n_timesteps=args.n_timesteps,
        out_dir=args.out_dir,
        smoke=args.smoke,
    )

