#!/usr/bin/env python3
"""Experiment 11b — RL policy comparison (Table II in paper).

Compares PPO-GOMDP, Greedy-GOMDP, PPO-CMDP, Adaptive AI, and Static Monitoring.
The governance compliance column confirms Theorem 1: GOMDP configs achieve 100%.

Paper reference: Table II, Section VI-B (Simulation Benchmarking).
Output: results/runs/<hash>/table2_rl_comparison.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys; _sys.path.insert(0, 'src'); _sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode
from wildfire_governance.governance.invariant_checker import GovernanceInvariantChecker
from wildfire_governance.rl.evaluator import evaluate
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")


def main(config_path: str, smoke: bool = False, use_pretrained: bool = True) -> None:
    cfg = load_config(config_path)
    run_hash = generate_run_hash(cfg)
    out_dir = RESULTS_BASE / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        n_seeds = cfg.simulation.n_seeds
        n_uavs = cfg.simulation.uav.n_uavs
        n_timesteps = cfg.simulation.n_timesteps
    except Exception:
        n_seeds, n_uavs, n_timesteps = 20, 20, 3000

    if smoke:
        n_seeds, n_uavs, n_timesteps = 2, 5, 100

    rows = []

    # 1. PPO-GOMDP (load checkpoint)
    logger.info("evaluating", method="PPO-GOMDP")
    ppo_results = evaluate(
        n_seeds=n_seeds, n_uavs=n_uavs,
        use_pretrained=use_pretrained, enable_governance=True, smoke=smoke
    )
    rows.append({
        "method": "PPO-GOMDP", "framework": "GOMDP",
        **ppo_results,
    })

    # 2. Greedy-GOMDP
    logger.info("evaluating", method="Greedy-GOMDP")
    greedy_fps, greedy_lds, greedy_comps = [], [], []
    for seed in range(n_seeds):
        r = run_episode(seed=seed, config_name="greedy_gomdp",
                        n_uavs=n_uavs, n_timesteps=n_timesteps,
                        enable_governance=True, enable_hitl=True,
                        enable_blockchain=True, enable_verification=True,
                        enable_coordination=True)
        greedy_fps.append(r.fp_pct)
        if r.ld < float("inf"):
            greedy_lds.append(r.ld)
        greedy_comps.append(float(getattr(r, "governance_compliant", False)))
    rows.append({
        "method": "Greedy-GOMDP", "framework": "GOMDP",
        "ld_mean": round(float(np.mean(greedy_lds)), 2),
        "ld_std": round(float(np.std(greedy_lds)), 2),
        "fp_mean": round(float(np.mean(greedy_fps)), 2),
        "fp_std": round(float(np.std(greedy_fps)), 2),
        "governance_compliance_pct": round(float(np.mean(greedy_comps)) * 100, 1),
        "n_seeds": n_seeds,
    })

    # 3. PPO-CMDP (no blockchain, Lagrangian constraint)
    logger.info("evaluating", method="PPO-CMDP")
    ppo_cmdp = evaluate(
        n_seeds=n_seeds, n_uavs=n_uavs,
        use_pretrained=False, enable_governance=False, smoke=smoke
    )
    checker = GovernanceInvariantChecker(tau=0.80)
    cmdp_compliances = []
    for seed in range(n_seeds):
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
        compliance = checker.check_episode(getattr(result, "step_logs", []))
        cmdp_compliances.append(float(compliance))

    ppo_cmdp["governance_compliance_pct"] = round(
        float(np.mean(cmdp_compliances)) * 100, 1
    )
    rows.append({"method": "PPO-CMDP", "framework": "CMDP", **ppo_cmdp})

    # 4. Adaptive AI (no governance)
    logger.info("evaluating", method="Adaptive-AI")
    ai_fps, ai_lds = [], []
    for seed in range(n_seeds):
        r = run_episode(seed=seed, config_name="adaptive_ai",
                        n_uavs=n_uavs, n_timesteps=n_timesteps,
                        enable_governance=False, enable_hitl=False,
                        enable_blockchain=False, enable_verification=True,
                        enable_coordination=True)
        ai_fps.append(r.fp_pct)
        if r.ld < float("inf"):
            ai_lds.append(r.ld)
    rows.append({
        "method": "Adaptive-AI", "framework": "None",
        "ld_mean": round(float(np.mean(ai_lds)), 2),
        "ld_std": round(float(np.std(ai_lds)), 2),
        "fp_mean": round(float(np.mean(ai_fps)), 2),
        "fp_std": round(float(np.std(ai_fps)), 2),
        "governance_compliance_pct": 0.0,
        "n_seeds": n_seeds,
    })

    # 5. Static
    logger.info("evaluating", method="Static")
    s_fps, s_lds = [], []
    for seed in range(n_seeds):
        r = run_episode(seed=seed, config_name="static",
                        n_uavs=n_uavs, n_timesteps=n_timesteps,
                        enable_governance=False, enable_hitl=False,
                        enable_blockchain=False, enable_verification=False,
                        enable_coordination=False)
        s_fps.append(r.fp_pct)
        if r.ld < float("inf"):
            s_lds.append(r.ld)
    rows.append({
        "method": "Static", "framework": "None",
        "ld_mean": round(float(np.mean(s_lds)), 2),
        "ld_std": round(float(np.std(s_lds)), 2),
        "fp_mean": round(float(np.mean(s_fps)), 2),
        "fp_std": round(float(np.std(s_fps)), 2),
        "governance_compliance_pct": 0.0,
        "n_seeds": n_seeds,
    })

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "table2_rl_comparison.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("experiment_complete", output=str(out_path))
    print(f"\n=== Table II RL Comparison ===\n{out_df.to_string(index=False)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--use_pretrained", action="store_true", default=True)
    args = parser.parse_args()
    main(args.config, args.smoke, args.use_pretrained)
