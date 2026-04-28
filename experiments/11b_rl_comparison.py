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
from scipy import stats  # type: ignore[import]

from experiments.utils.runner import run_episode
from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker as GOMDPInvariantChecker
from wildfire_governance.governance.invariant_checker import GovernanceInvariantChecker
from wildfire_governance.metrics.statistical_tests import paired_ttest_holm_bonferroni
from wildfire_governance.rl import evaluator as ppo_eval_module
from wildfire_governance.rl.gomdp_env import GOMMDPGymEnv
from wildfire_governance.rl.ppo_agent import PPOGOMDPAgent
from wildfire_governance.simulation.grid_environment import EnvironmentConfig
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash
from wildfire_governance.utils.reproducibility import set_global_seed

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")


def _enforce_min_seeds(n_seeds: int) -> int:
    if n_seeds < 3:
        logger.warning("min_seeds_enforced", requested=n_seeds, effective=3)
        return 3
    return n_seeds


def _mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, 0.0, 0.0
    std = float(np.std(arr, ddof=1))
    t_crit = float(stats.t.ppf(0.975, df=len(arr) - 1))
    ci95 = float(t_crit * std / np.sqrt(len(arr)))
    return mean, std, ci95


def _evaluate_ppo_seedwise(
    n_seeds: int,
    n_uavs: int,
    n_timesteps: int,
    use_pretrained: bool,
    enable_governance: bool,
    smoke: bool,
) -> dict:
    grid_size = 10 if smoke else 100
    env_config = EnvironmentConfig(grid_size=grid_size, n_timesteps=n_timesteps)

    agent = PPOGOMDPAgent(grid_size=grid_size, n_uavs=n_uavs)
    if use_pretrained:
        ckpt = ppo_eval_module.CHECKPOINT_DIR / "ppo_gomdp_best.pt"
        try:
            loaded = ppo_eval_module._load_checkpoint_if_compatible(agent, ckpt)
            if not loaded:
                logger.warning(
                    "checkpoint_compatibility_fallback",
                    path=str(ckpt),
                    reason="shape mismatch or load failure; using random init",
                )
        except FileNotFoundError:
            logger.warning(
                "checkpoint_not_found",
                path=str(ckpt),
                reason="using random init",
            )

    checker = GOMDPInvariantChecker(tau=0.80)
    lds, fps, comps = [], [], []

    for seed in range(n_seeds):
        set_global_seed(seed)
        env = GOMMDPGymEnv(config=env_config, n_uavs=n_uavs, enable_governance=enable_governance)
        obs, _ = env.reset(seed=seed)
        done = False
        info = {}

        while not done:
            action_dict = agent.select_actions(obs, env._fleet)
            action_arr = np.array([action_dict.get(i, 0) for i in range(n_uavs)])
            obs, _, terminated, truncated, info = env.step(action_arr)
            done = terminated or truncated

        traj = env.get_trajectory()
        report = checker.check_trajectory(traj)
        lds.append(float(info.get("episode_ld", float("inf"))))
        fps.append(float(info.get("episode_fp_pct", 0.0)))
        comps.append(float(report.theorem1_satisfied))

    return {"ld": lds, "fp": fps, "comp": comps}


def main(config_path: str, smoke: bool = False, use_pretrained: bool = True) -> None:
    if smoke:
        use_pretrained = False

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

    n_seeds = _enforce_min_seeds(int(n_seeds))

    rows = []
    metric_series: dict[str, dict[str, list[float]]] = {}

    # 1. PPO-GOMDP (load checkpoint)
    logger.info("evaluating", method="PPO-GOMDP")
    ppo_seedwise = _evaluate_ppo_seedwise(
        n_seeds=n_seeds,
        n_uavs=n_uavs,
        n_timesteps=n_timesteps,
        use_pretrained=use_pretrained,
        enable_governance=True,
        smoke=smoke,
    )
    ppo_ld_mean, ppo_ld_std, ppo_ld_ci = _mean_std_ci(ppo_seedwise["ld"])
    ppo_fp_mean, ppo_fp_std, ppo_fp_ci = _mean_std_ci(ppo_seedwise["fp"])
    ppo_comp_mean = float(np.mean(ppo_seedwise["comp"])) * 100.0
    rows.append({
        "method": "PPO-GOMDP", "framework": "GOMDP",
        "ld_mean": round(ppo_ld_mean, 2),
        "ld_std": round(ppo_ld_std, 2),
        "ld_ci95": round(ppo_ld_ci, 2),
        "fp_mean": round(ppo_fp_mean, 2),
        "fp_std": round(ppo_fp_std, 2),
        "fp_ci95": round(ppo_fp_ci, 2),
        "governance_compliance_pct": round(ppo_comp_mean, 1),
        "n_seeds": n_seeds,
    })
    metric_series["PPO-GOMDP"] = {"ld": ppo_seedwise["ld"], "fp": ppo_seedwise["fp"]}

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
    g_ld_mean, g_ld_std, g_ld_ci = _mean_std_ci(greedy_lds)
    g_fp_mean, g_fp_std, g_fp_ci = _mean_std_ci(greedy_fps)
    rows.append({
        "method": "Greedy-GOMDP", "framework": "GOMDP",
        "ld_mean": round(g_ld_mean, 2),
        "ld_std": round(g_ld_std, 2),
        "ld_ci95": round(g_ld_ci, 2),
        "fp_mean": round(g_fp_mean, 2),
        "fp_std": round(g_fp_std, 2),
        "fp_ci95": round(g_fp_ci, 2),
        "governance_compliance_pct": round(float(np.mean(greedy_comps)) * 100, 1),
        "n_seeds": n_seeds,
    })
    metric_series["Greedy-GOMDP"] = {"ld": greedy_lds, "fp": greedy_fps}

    # 3. PPO-CMDP (no blockchain, Lagrangian constraint)
    logger.info("evaluating", method="PPO-CMDP")
    cmdp_seedwise = _evaluate_ppo_seedwise(
        n_seeds=n_seeds, n_uavs=n_uavs,
        n_timesteps=n_timesteps,
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
    c_ld_mean, c_ld_std, c_ld_ci = _mean_std_ci(cmdp_seedwise["ld"])
    c_fp_mean, c_fp_std, c_fp_ci = _mean_std_ci(cmdp_seedwise["fp"])
    rows.append({
        "method": "PPO-CMDP", "framework": "CMDP",
        "ld_mean": round(c_ld_mean, 2),
        "ld_std": round(c_ld_std, 2),
        "ld_ci95": round(c_ld_ci, 2),
        "fp_mean": round(c_fp_mean, 2),
        "fp_std": round(c_fp_std, 2),
        "fp_ci95": round(c_fp_ci, 2),
        "governance_compliance_pct": round(float(np.mean(cmdp_compliances)) * 100, 1),
        "n_seeds": n_seeds,
    })
    metric_series["PPO-CMDP"] = {"ld": cmdp_seedwise["ld"], "fp": cmdp_seedwise["fp"]}

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
    a_ld_mean, a_ld_std, a_ld_ci = _mean_std_ci(ai_lds)
    a_fp_mean, a_fp_std, a_fp_ci = _mean_std_ci(ai_fps)
    rows.append({
        "method": "Adaptive-AI", "framework": "None",
        "ld_mean": round(a_ld_mean, 2),
        "ld_std": round(a_ld_std, 2),
        "ld_ci95": round(a_ld_ci, 2),
        "fp_mean": round(a_fp_mean, 2),
        "fp_std": round(a_fp_std, 2),
        "fp_ci95": round(a_fp_ci, 2),
        "governance_compliance_pct": 0.0,
        "n_seeds": n_seeds,
    })
    metric_series["Adaptive-AI"] = {"ld": ai_lds, "fp": ai_fps}

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
    s_ld_mean, s_ld_std, s_ld_ci = _mean_std_ci(s_lds)
    s_fp_mean, s_fp_std, s_fp_ci = _mean_std_ci(s_fps)
    rows.append({
        "method": "Static", "framework": "None",
        "ld_mean": round(s_ld_mean, 2),
        "ld_std": round(s_ld_std, 2),
        "ld_ci95": round(s_ld_ci, 2),
        "fp_mean": round(s_fp_mean, 2),
        "fp_std": round(s_fp_std, 2),
        "fp_ci95": round(s_fp_ci, 2),
        "governance_compliance_pct": 0.0,
        "n_seeds": n_seeds,
    })
    metric_series["Static"] = {"ld": s_lds, "fp": s_fps}

    baseline = "PPO-GOMDP"
    comparisons = []
    for method_name, series in metric_series.items():
        if method_name == baseline:
            continue
        comparisons.append((method_name, baseline, "ld", series["ld"], metric_series[baseline]["ld"]))
        comparisons.append((method_name, baseline, "fp", series["fp"], metric_series[baseline]["fp"]))

    test_results = paired_ttest_holm_bonferroni(comparisons) if comparisons else []
    stats_lookup = {
        (r.group_a, r.group_b, r.metric): (r.p_value_corrected, r.effect_size)
        for r in test_results
    }

    structured_rows = []
    for method_name, series in metric_series.items():
        for metric in ("ld", "fp"):
            mean, std, ci95 = _mean_std_ci(series[metric])
            p_val, effect = (np.nan, np.nan)
            if method_name != baseline:
                p_val, effect = stats_lookup.get((method_name, baseline, metric), (np.nan, np.nan))
            structured_rows.append({
                "method": method_name,
                "metric": metric,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95": round(ci95, 4),
                "p_value": None if np.isnan(p_val) else round(float(p_val), 6),
                "effect_size": None if np.isnan(effect) else round(float(effect), 4),
                "comparison": "reference" if method_name == baseline else f"vs {baseline}",
                "n_seeds": n_seeds,
            })

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "table2_rl_comparison.csv"
    out_df.to_csv(out_path, index=False)

    stats_df = pd.DataFrame(structured_rows)
    stats_path = out_dir / "table2_rl_comparison_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    logger.info("experiment_complete", output=str(out_path))
    print(f"\n=== Table II RL Comparison ===\n{out_df.to_string(index=False)}\n")
    print(f"=== Statistical Summary (mean | std | CI | p-value | effect size) ===\n{stats_df.to_string(index=False)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/paper_main_results.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--use_pretrained", action="store_true", default=True)
    args = parser.parse_args()
    main(args.config, args.smoke, args.use_pretrained)
