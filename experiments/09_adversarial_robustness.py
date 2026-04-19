#!/usr/bin/env python3
"""Experiment 09 — Adversarial robustness evaluation (Table V in paper).

Tests all attack types: sensor spoofing, alert injection, Byzantine faults.
Compares GOMDP vs. centralized (no blockchain) for each attack.

Paper reference: Table V, Section VI-D (Adversarial Robustness).
Output: results/runs/<hash>/table5_adversarial.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys; _sys.path.insert(0, 'src'); _sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode
from wildfire_governance.blockchain.smart_contract import GovernanceSmartContract
from wildfire_governance.gomdp.breach_probability import (
    compute_breach_probability_gomdp,
    compute_breach_probability_centralized,
)
from wildfire_governance.utils.config import load_config
from wildfire_governance.utils.logging import get_structured_logger
from wildfire_governance.utils.reproducibility import generate_run_hash

logger = get_structured_logger(__name__)
RESULTS_BASE = Path("results/runs")

ATTACK_CONFIGS = [
    {"attack_type": "no_attack",  "p_spoof": 0.0, "n_byzantine": 0},
    {"attack_type": "spoofing",   "parameter": "p=0.05", "p_spoof": 0.05, "n_byzantine": 0},
    {"attack_type": "spoofing",   "parameter": "p=0.10", "p_spoof": 0.10, "n_byzantine": 0},
    {"attack_type": "spoofing",   "parameter": "p=0.20", "p_spoof": 0.20, "n_byzantine": 0},
    {"attack_type": "injection",  "parameter": "p_att=1.0", "p_spoof": 0.0, "n_byzantine": 0},
    {"attack_type": "byzantine",  "parameter": "f=0", "p_spoof": 0.0, "n_byzantine": 0},
    {"attack_type": "byzantine",  "parameter": "f=1", "p_spoof": 0.0, "n_byzantine": 1},
    {"attack_type": "byzantine",  "parameter": "f=2", "p_spoof": 0.0, "n_byzantine": 2},
    {"attack_type": "byzantine",  "parameter": "f=3", "p_spoof": 0.0, "n_byzantine": 3},
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
        parameter = atk.get("parameter", "")
        p_spoof = atk["p_spoof"]
        n_byzantine = atk["n_byzantine"]

        # GOMDP runs
        gomdp_fps, central_fps = [], []
        for seed in range(n_seeds):
            # GOMDP
            r_gomdp = run_episode(
                seed=seed, config_name="gomdp",
                n_uavs=n_uavs, n_timesteps=n_timesteps,
                enable_governance=True, enable_hitl=True,
                enable_blockchain=True, enable_verification=True,
                enable_coordination=True,
                p_spoof=p_spoof, n_byzantine=n_byzantine,
            )
            gomdp_fps.append(r_gomdp.fp_pct)

            # Centralized (no blockchain)
            if attack_type != "byzantine":
                r_central = run_episode(
                    seed=seed, config_name="central",
                    n_uavs=n_uavs, n_timesteps=n_timesteps,
                    enable_governance=False, enable_hitl=False,
                    enable_blockchain=False, enable_verification=False,
                    enable_coordination=True,
                    p_spoof=p_spoof,
                )
                central_fps.append(r_central.fp_pct)

        # Theoretical breach probability from Theorem 2
        if attack_type == "injection":
            breaches = 0
            total = 0
            n_trials = max(1, n_seeds * max(1, n_timesteps // 50))
            contract = GovernanceSmartContract(tau=0.80)

            for _ in range(n_trials):
                success = contract.attempt_unauthorised_injection((0, 0, 1, 1))
                if success:
                    breaches += 1
                total += 1

            p_breach_gomdp = breaches / total
            p_breach_central = compute_breach_probability_centralized(1.0)
        elif attack_type == "byzantine":
            p_c = 0.3
            f_actual = n_byzantine

            if f_actual <= (7 - 1) // 3:
                p_breach_gomdp = compute_breach_probability_gomdp(7, f_actual, p_c)
            else:
                p_breach_gomdp = 1.0
            p_breach_central = None
        else:
            breaches = 0
            total = 0
            n_trials = max(1, n_seeds * max(1, n_timesteps // 50))
            contract = GovernanceSmartContract(tau=0.80)

            for _ in range(n_trials):
                success = contract.attempt_unauthorised_injection((0, 0, 1, 1))
                if success:
                    breaches += 1
                total += 1

            p_breach_gomdp = breaches / total
            p_breach_central = None

        rows.append({
            "attack_type": attack_type,
            "parameter": parameter,
            "gomdp_fp": round(float(np.mean(gomdp_fps)), 2),
            "gomdp_fp_std": round(float(np.std(gomdp_fps)), 2),
            "central_fp": round(float(np.mean(central_fps)), 2) if central_fps else None,
            "central_fp_std": round(float(np.std(central_fps)), 2) if central_fps else None,
            "p_breach_gomdp": round(float(p_breach_gomdp), 3) if p_breach_gomdp is not None else None,
            "p_breach_central": round(float(p_breach_central), 3) if p_breach_central is not None else None,
        })
        logger.info("attack_evaluated", attack=attack_type, param=parameter,
                    gomdp_fp=round(float(np.mean(gomdp_fps)), 2))

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "table5_adversarial.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("experiment_complete", output=str(out_path))
    print(f"\n=== Table V Adversarial Robustness ===\n{out_df.to_string(index=False)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/adversarial_robustness.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
