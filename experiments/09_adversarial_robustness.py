#!/usr/bin/env python3
"""Experiment 09 — Adversarial robustness evaluation (Table 3 in paper).

Tests all attack types: sensor spoofing, alert injection, strategic spoofing.
Compares GOMDP vs. Central+Sig vs. Central for each attack.

Paper reference: Table 3, Section VI-D (Adversarial Robustness).
Output: results/runs/<hash>/table3_adversarial.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys; _sys.path.insert(0, 'src'); _sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")

ATTACK_CONFIGS = [
    {"attack_type": "no_attack",  "parameter": "---", "p_spoof": 0.0, "strategic": False, "metric": "fp_pct", "display_attack": "No attack"},
    {"attack_type": "spoofing",   "parameter": "p=0.05", "p_spoof": 0.05, "strategic": False, "metric": "fp_pct", "display_attack": "Spoofing (i.i.d.)"},
    {"attack_type": "spoofing",   "parameter": "p=0.10", "p_spoof": 0.10, "strategic": False, "metric": "fp_pct", "display_attack": "Spoofing (i.i.d.)"},
    {"attack_type": "spoofing",   "parameter": "p=0.20", "p_spoof": 0.20, "strategic": False, "metric": "fp_pct", "display_attack": "Spoofing (i.i.d.)"},
    {"attack_type": "spoofing_strategic", "parameter": "p=0.10", "p_spoof": 0.10, "strategic": True, "metric": "fp_pct", "display_attack": "Spoofing (strategic)"},
    {"attack_type": "injection",  "parameter": "p_att=1", "p_spoof": 0.0, "strategic": False, "metric": "injection_ratio", "display_attack": "Alert injection (success)"},
]


def main(config_path: str, smoke: bool = False) -> None:
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

    for atk in ATTACK_CONFIGS:
        attack_type = atk["attack_type"]
        parameter = atk["parameter"]
        p_spoof = atk["p_spoof"]
        strategic = atk["strategic"]
        metric = atk["metric"]
        display_attack = atk["display_attack"]

        # Run GOMDP, Central+Sig, Central
        gomdp_vals, sig_vals, central_vals = [], [], []
        for seed in range(n_seeds):
            # 1. GOMDP
            r_gomdp = run_episode(
                seed=seed, config_name="gomdp",
                n_uavs=n_uavs, n_timesteps=n_timesteps,
                enable_governance=True, enable_hitl=True,
                enable_blockchain=True, enable_verification=True,
                enable_coordination=True,
                p_spoof=p_spoof, n_byzantine=0,
                attack_type="spoofing_strategic" if strategic else ("none" if attack_type == "no_attack" else attack_type),
            )
            # 2. Central+Sig
            r_sig = run_episode(
                seed=seed, config_name="central_sig",
                n_uavs=n_uavs, n_timesteps=n_timesteps,
                enable_governance=False, enable_hitl=False,
                enable_blockchain=False, enable_verification=True,
                enable_coordination=True,
                p_spoof=p_spoof, n_byzantine=0,
                attack_type="spoofing_strategic" if strategic else ("none" if attack_type == "no_attack" else attack_type),
            )
            # 3. Central
            r_central = run_episode(
                seed=seed, config_name="central",
                n_uavs=n_uavs, n_timesteps=n_timesteps,
                enable_governance=False, enable_hitl=False,
                enable_blockchain=False, enable_verification=False,
                enable_coordination=True,
                p_spoof=p_spoof, n_byzantine=0,
                attack_type="spoofing_strategic" if strategic else ("none" if attack_type == "no_attack" else attack_type),
            )

            if metric == "injection_ratio":
                gomdp_vals.append(int(getattr(r_gomdp, "injection_success", 0)))
                sig_vals.append(int(getattr(r_sig, "injection_success", 0)))
                central_vals.append(int(getattr(r_central, "injection_success", 0)))
            else:
                gomdp_vals.append(r_gomdp.fp_pct)
                sig_vals.append(r_sig.fp_pct)
                central_vals.append(r_central.fp_pct)

        if metric == "injection_ratio":
            # Ratio string representation e.g. "X/N"
            gomdp_str = f"{sum(gomdp_vals)}/{n_seeds}"
            sig_str = f"{sum(sig_vals)}/{n_seeds}"
            central_str = f"{sum(central_vals)}/{n_seeds}"
        else:
            gomdp_str = round(float(np.mean(gomdp_vals)), 1)
            sig_str = round(float(np.mean(sig_vals)), 1)
            central_str = round(float(np.mean(central_vals)), 1)

        rows.append({
            "attack_type": display_attack,
            "parameter": parameter,
            "gomdp": gomdp_str,
            "central_sig": sig_str,
            "central": central_str,
            "metric": metric
        })
        logger.info("attack_evaluated", attack=attack_type, param=parameter,
                    gomdp_val=gomdp_str)

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "table3_adversarial.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("experiment_complete", output=str(out_path))
    print(f"\n=== Table 3 Adversarial Robustness ===\n{out_df.to_string(index=False)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/adversarial_robustness.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
