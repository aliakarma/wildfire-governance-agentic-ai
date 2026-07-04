#!/usr/bin/env python3
import subprocess
import sys
import os

smoke = ["--smoke"] if "--smoke" in sys.argv else []
skip_training = "--use_pretrained"

# Set PYTHONPATH and GOMDP_RUN_HASH
os.environ["PYTHONPATH"] = f"src{os.pathsep}."
os.environ["GOMDP_RUN_HASH"] = "reproduced"

commands = [
    ["experiments/01_main_comparison.py", "--config", "configs/experiments/paper_main_results.yaml"],
    ["experiments/02_ablation_study.py", "--config", "configs/experiments/paper_main_results.yaml"],
    ["experiments/03_scalability.py", "--config", "configs/experiments/scalability_uav_fleet.yaml"],
    ["experiments/04_false_alert_rate.py", "--config", "configs/experiments/paper_main_results.yaml"],
    ["experiments/05_tradeoff_frontier.py", "--config", "configs/experiments/paper_main_results.yaml"],
    ["experiments/06_threshold_sensitivity.py", "--config", "configs/experiments/sensitivity_thresholds.yaml"],
    ["experiments/11_ppo_training.py", "--config", "configs/experiments/ppo_training.yaml", skip_training],
    ["experiments/11b_rl_comparison.py", "--config", "configs/experiments/paper_main_results.yaml"],
    ["experiments/09_adversarial_robustness.py", "--config", "configs/experiments/adversarial_robustness.yaml"],
    ["experiments/10_stress_testing.py", "--config", "configs/experiments/stress_testing.yaml"],
    ["experiments/08_viirs_california.py", "--config", "configs/experiments/realworld_viirs.yaml"]
]

python_bin = sys.executable

for cmd in commands:
    full_cmd = [python_bin] + cmd + smoke
    print(f"\n========================================\nRunning: {' '.join(full_cmd)}\n========================================")
    subprocess.run(full_cmd, check=False)
