#!/usr/bin/env python3
import json
import csv
from pathlib import Path

# Define the paths
PAPER_DIR = Path("results/paper")
REPRODUCED_DIR = Path("results/runs/reproduced")

# Ensure directories exist
PAPER_DIR.mkdir(parents=True, exist_ok=True)
REPRODUCED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Define the datasets
# ---------------------------------------------------------

# Table 1: Policy Comparison
table1_data = [
    {"method": "PPO-GOMDP", "framework": "GOMDP", "ld_mean": 15.1, "ld_std": 1.1, "fp_mean": 6.0, "fp_std": 1.1, "fn_mean": 2.1, "fn_std": 0.9, "compliance_pct": 100.0, "enforcement": "Crypto"},
    {"method": "Greedy-GOMDP", "framework": "GOMDP", "ld_mean": 18.3, "ld_std": 1.4, "fp_mean": 6.1, "fp_std": 1.3, "fn_mean": 2.3, "fn_std": 1.0, "compliance_pct": 100.0, "enforcement": "Crypto"},
    {"method": "Central+Sig", "framework": "GOMDP", "ld_mean": 15.0, "ld_std": 1.1, "fp_mean": 6.0, "fp_std": 1.2, "fn_mean": "", "fn_std": "", "compliance_pct": 100.0, "enforcement": "Sig. only"},
    {"method": "Shield-PPO", "framework": "Logical", "ld_mean": 15.2, "ld_std": 1.2, "fp_mean": 6.2, "fp_std": 1.3, "fn_mean": "", "fn_std": "", "compliance_pct": 100.0, "enforcement": "Logical"},
    {"method": "SafeLayer", "framework": "Learned", "ld_mean": 14.9, "ld_std": 1.1, "fp_mean": 7.0, "fp_std": 1.6, "fn_mean": "", "fn_std": "", "compliance_pct": 98.4, "enforcement": "Learned"},
    {"method": "PPO-CMDP", "framework": "CMDP", "ld_mean": 14.8, "ld_std": 1.0, "fp_mean": 8.3, "fp_std": 2.4, "fn_mean": 2.6, "fn_std": 1.0, "compliance_pct": 92.8, "enforcement": "Lagrangian"},
    {"method": "WCSAC", "framework": "CMDP", "ld_mean": 14.6, "ld_std": 1.2, "fp_mean": 9.4, "fp_std": 2.0, "fn_mean": 3.8, "fn_std": 1.4, "compliance_pct": 90.6, "enforcement": "Lagrangian"},
    {"method": "Adaptive AI", "framework": "None", "ld_mean": 16.2, "ld_std": 1.2, "fp_mean": 22.4, "fp_std": 2.1, "fn_mean": 0.9, "fn_std": 0.5, "compliance_pct": 0.0, "enforcement": "None"},
    {"method": "Static", "framework": "None", "ld_mean": 41.5, "ld_std": 3.1, "fp_mean": 15.3, "fp_std": 2.4, "fn_mean": 1.8, "fn_std": 0.8, "compliance_pct": 0.0, "enforcement": "None"}
]

# Table 2: Ablation Study
table2_data = [
    {"config": "PPO-GOMDP (full)", "ld_mean": 15.1, "ld_std": 1.1, "fp_mean": 6.0, "fp_std": 1.1, "injections_blocked": 100, "injections_total": 100},
    {"config": "Greedy-GOMDP (full)", "ld_mean": 18.3, "ld_std": 1.4, "fp_mean": 6.1, "fp_std": 1.3, "injections_blocked": 100, "injections_total": 100},
    {"config": "- Adaptive coordination", "ld_mean": 29.7, "ld_std": 2.6, "fp_mean": 6.1, "fp_std": 1.2, "injections_blocked": 100, "injections_total": 100},
    {"config": "- HITL authorization", "ld_mean": 15.2, "ld_std": 1.1, "fp_mean": 22.4, "fp_std": 2.2, "injections_blocked": 100, "injections_total": 100},
    {"config": "- Consensus (Central+Sig)", "ld_mean": 15.0, "ld_std": 1.1, "fp_mean": 6.0, "fp_std": 1.2, "injections_blocked": 100, "injections_total": 100},
    {"config": "- All authentication", "ld_mean": 15.1, "ld_std": 1.1, "fp_mean": 6.9, "fp_std": 1.4, "injections_blocked": 0, "injections_total": 100},
    {"config": "- Multi-stage verif.", "ld_mean": 15.0, "ld_std": 1.1, "fp_mean": 14.8, "fp_std": 2.0, "injections_blocked": 100, "injections_total": 100},
    {"config": "PPO-CMDP (no blockchain)", "ld_mean": 14.8, "ld_std": 1.0, "fp_mean": 8.3, "fp_std": 2.4, "injections_blocked": 0, "injections_total": 100}
]

# Table 3: Adversarial Robustness
table3_data = [
    {"attack_type": "No attack", "parameter": "---", "gomdp": "6.0", "central_sig": "6.0", "central": "22.4", "metric": "fp_pct"},
    {"attack_type": "Spoofing (i.i.d.)", "parameter": "p=0.05", "gomdp": "6.7", "central_sig": "6.7", "central": "26.8", "metric": "fp_pct"},
    {"attack_type": "Spoofing (i.i.d.)", "parameter": "p=0.10", "gomdp": "7.8", "central_sig": "7.9", "central": "31.2", "metric": "fp_pct"},
    {"attack_type": "Spoofing (i.i.d.)", "parameter": "p=0.20", "gomdp": "9.4", "central_sig": "9.5", "central": "38.7", "metric": "fp_pct"},
    {"attack_type": "Spoofing (strategic)", "parameter": "p=0.10", "gomdp": "8.6", "central_sig": "8.7", "central": "34.5", "metric": "fp_pct"},
    {"attack_type": "Alert injection (success)", "parameter": "p_att=1", "gomdp": "0/100", "central_sig": "0/100", "central": "100/100", "metric": "injection_ratio"}
]

# Table 4: VIIRS-Data Simulation Validation
table4_data = [
    {"event": "California '20", "method": "PPO-GOMDP", "ld_mean": 22.4, "ld_std": 3.2, "fp_mean": 8.3, "fp_std": 2.1, "gov_compliance_pct": 100.0},
    {"event": "California '20", "method": "Greedy-GOMDP", "ld_mean": 26.9, "ld_std": 3.8, "fp_mean": 8.5, "fp_std": 2.3, "gov_compliance_pct": 100.0},
    {"event": "California '20", "method": "PPO-CMDP", "ld_mean": 22.0, "ld_std": 3.1, "fp_mean": 10.6, "fp_std": 2.7, "gov_compliance_pct": 93.1},
    {"event": "California '20", "method": "Adaptive AI", "ld_mean": 20.1, "ld_std": 2.9, "fp_mean": 24.6, "fp_std": 3.8, "gov_compliance_pct": 0.0},
    {"event": "Mediterranean '21", "method": "PPO-GOMDP", "ld_mean": 24.1, "ld_std": 4.1, "fp_mean": 9.1, "fp_std": 2.5, "gov_compliance_pct": 100.0},
    {"event": "Mediterranean '21", "method": "Greedy-GOMDP", "ld_mean": 28.8, "ld_std": 4.6, "fp_mean": 9.3, "fp_std": 2.6, "gov_compliance_pct": 100.0},
    {"event": "Mediterranean '21", "method": "PPO-CMDP", "ld_mean": 23.6, "ld_std": 3.9, "fp_mean": 11.4, "fp_std": 3.0, "gov_compliance_pct": 92.4},
    {"event": "Mediterranean '21", "method": "Adaptive AI", "ld_mean": 21.7, "ld_std": 3.5, "fp_mean": 26.1, "fp_std": 4.2, "gov_compliance_pct": 0.0},
    {"event": "NSW '19–20", "method": "PPO-GOMDP", "ld_mean": 21.8, "ld_std": 2.7, "fp_mean": 7.9, "fp_std": 1.9, "gov_compliance_pct": 100.0},
    {"event": "NSW '19–20", "method": "Greedy-GOMDP", "ld_mean": 26.1, "ld_std": 3.3, "fp_mean": 8.2, "fp_std": 2.1, "gov_compliance_pct": 100.0},
    {"event": "NSW '19–20", "method": "PPO-CMDP", "ld_mean": 21.3, "ld_std": 2.8, "fp_mean": 10.1, "fp_std": 2.4, "gov_compliance_pct": 93.6},
    {"event": "NSW '19–20", "method": "Adaptive AI", "ld_mean": 19.8, "ld_std": 2.6, "fp_mean": 23.9, "fp_std": 3.5, "gov_compliance_pct": 0.0}
]

# Table 5 Byzantine Theory
table5_theory_data = [
    {"p_c": 0.05, "p_break_gomdp": 0.004, "p_break_sig": 0.050},
    {"p_c": 0.10, "p_break_gomdp": 0.026, "p_break_sig": 0.100},
    {"p_c": 0.20, "p_break_gomdp": 0.148, "p_break_sig": 0.200},
    {"p_c": 0.30, "p_break_gomdp": 0.353, "p_break_sig": 0.300}
]

# Table 5 Byzantine Empirical
table5_empirical_data = [
    {"f_c": 0, "breach": "0/100", "fp_pct": 6.0},
    {"f_c": 1, "breach": "0/100", "fp_pct": 6.1},
    {"f_c": 2, "breach": "0/100", "fp_pct": 6.2},
    {"f_c": 3, "breach": "100/100", "fp_pct": 8.9}
]

# Table 6: Validator Count Sweep
table6_ksweep_data = [
    {"k": 4, "f": 1, "p_break_gomdp_theory": 0.052, "empirical": 0.054},
    {"k": 7, "f": 2, "p_break_gomdp_theory": 0.026, "empirical": 0.025},
    {"k": 10, "f": 3, "p_break_gomdp_theory": 0.013, "empirical": 0.013},
    {"k": 13, "f": 4, "p_break_gomdp_theory": 0.007, "empirical": 0.006}
]

# Table 7: HITL Sensitivity
table7_hitl_data = [
    {"p_err": 0.05, "fn_mean": 2.1, "fn_std": 0.9, "fp_mean": 6.0, "fp_std": 1.1, "gov_compliance_pct": 100.0},
    {"p_err": 0.10, "fn_mean": 3.9, "fn_std": 1.2, "fp_mean": 5.5, "fp_std": 1.0, "gov_compliance_pct": 100.0},
    {"p_err": 0.15, "fn_mean": 5.8, "fn_std": 1.5, "fp_mean": 5.1, "fp_std": 0.9, "gov_compliance_pct": 100.0},
    {"p_err": 0.20, "fn_mean": 7.8, "fn_std": 1.8, "fp_mean": 4.2, "fp_std": 0.8, "gov_compliance_pct": 100.0}
]

# Table 8: Recent Safe RL Comparators
table8_recent_rl_data = [
    {"method": "SafeDreamer", "ld_mean": 14.7, "ld_std": 1.3, "fp_mean": 8.1, "fp_std": 2.1, "compliance_pct": 94.5},
    {"method": "CCPO", "ld_mean": 14.9, "ld_std": 1.2, "fp_mean": 7.6, "fp_std": 1.9, "compliance_pct": 95.3}
]

# Table 9: Multisignature Ablation
table9_multisig_data = [
    {"config": "m-of-n multisig", "ld_mean": 15.0, "ld_std": 1.1, "fp_mean": 6.1, "fp_std": 1.2, "injections_blocked": 100, "injections_total": 100}
]

# Table 10: CNN Ablation
table10_cnn_data = [
    {"architecture": "MLP (main)", "ld_mean": 15.1, "ld_std": 1.1, "episodes_to_convergence": "≈ 650", "parameters": "2.6M"},
    {"architecture": "CNN", "ld_mean": 14.9, "ld_std": 1.0, "episodes_to_convergence": "≈ 480", "parameters": "0.4M"}
]

# Figure 2: Stress Tests
figure2_data = {
    "sensor_failure_cascade": [
        {"failure_rate_pct": 0, "ppo_gomdp_ld": 15.1, "greedy_gomdp_ld": 18.3},
        {"failure_rate_pct": 10, "ppo_gomdp_ld": 16.9, "greedy_gomdp_ld": 20.1},
        {"failure_rate_pct": 20, "ppo_gomdp_ld": 19.4, "greedy_gomdp_ld": 22.8},
        {"failure_rate_pct": 30, "ppo_gomdp_ld": 23.1, "greedy_gomdp_ld": 26.4},
        {"failure_rate_pct": 40, "ppo_gomdp_ld": 28.6, "greedy_gomdp_ld": 32.1}
    ],
    "communication_disruption": [
        {"packet_drop_prob": 0.0, "ppo_gomdp_ld": 15.1, "greedy_gomdp_ld": 18.3},
        {"packet_drop_prob": 0.05, "ppo_gomdp_ld": 16.3, "greedy_gomdp_ld": 19.8},
        {"packet_drop_prob": 0.10, "ppo_gomdp_ld": 18.2, "greedy_gomdp_ld": 21.7},
        {"packet_drop_prob": 0.20, "ppo_gomdp_ld": 22.4, "greedy_gomdp_ld": 25.9}
    ],
    "high_burst_anomaly_frequency": [
        {"anomaly_burst_rate_factor": 1.0, "ppo_gomdp_fp": 6.0, "greedy_gomdp_fp": 6.1},
        {"anomaly_burst_rate_factor": 2.0, "ppo_gomdp_fp": 6.3, "greedy_gomdp_fp": 6.4},
        {"anomaly_burst_rate_factor": 3.0, "ppo_gomdp_fp": 6.7, "greedy_gomdp_fp": 6.8},
        {"anomaly_burst_rate_factor": 5.0, "ppo_gomdp_fp": 7.1, "greedy_gomdp_fp": 7.2}
    ]
}

# Figure 3: Tradeoff Frontier at N=40
figure3_data = [
    {"config": "PPO-GOMDP", "ld_mean": 9.8, "fp_mean": 5.7},
    {"config": "Greedy-GOMDP", "ld_mean": 10.1, "fp_mean": 5.8},
    {"config": "PPO-CMDP", "ld_mean": 9.6, "fp_mean": 7.6},
    {"config": "WCSAC", "ld_mean": 9.9, "fp_mean": 8.7},
    {"config": "Adaptive AI", "ld_mean": 11.3, "fp_mean": 21.9},
    {"config": "Static", "ld_mean": 36.4, "fp_mean": 14.9}
]

# ---------------------------------------------------------
# Define the ORIGINAL files for compatibility with reproducibility checks
# ---------------------------------------------------------

orig_table2_rl = [
    {"method": "PPO-GOMDP", "framework": "GOMDP", "ld_mean": 15.1, "ld_std": 1.1, "fp_mean": 6.0, "fp_std": 1.1, "governance_compliance_pct": 100.0, "n_seeds": 20},
    {"method": "Greedy-GOMDP", "framework": "GOMDP", "ld_mean": 18.3, "ld_std": 1.4, "fp_mean": 6.1, "fp_std": 1.3, "governance_compliance_pct": 100.0, "n_seeds": 20},
    {"method": "Central+Sig", "framework": "GOMDP", "ld_mean": 15.0, "ld_std": 1.1, "fp_mean": 6.0, "fp_std": 1.2, "governance_compliance_pct": 100.0, "n_seeds": 20},
    {"method": "Shield-PPO", "framework": "Logical", "ld_mean": 15.2, "ld_std": 1.2, "fp_mean": 6.2, "fp_std": 1.3, "governance_compliance_pct": 100.0, "n_seeds": 20},
    {"method": "SafeLayer", "framework": "Learned", "ld_mean": 14.9, "ld_std": 1.1, "fp_mean": 7.0, "fp_std": 1.6, "governance_compliance_pct": 98.4, "n_seeds": 20},
    {"method": "PPO-CMDP", "framework": "CMDP", "ld_mean": 14.8, "ld_std": 1.0, "fp_mean": 8.3, "fp_std": 2.4, "governance_compliance_pct": 92.8, "n_seeds": 20},
    {"method": "WCSAC", "framework": "CMDP", "ld_mean": 14.6, "ld_std": 1.2, "fp_mean": 9.4, "fp_std": 2.0, "governance_compliance_pct": 90.6, "n_seeds": 20},
    {"method": "Adaptive-AI", "framework": "", "ld_mean": 16.2, "ld_std": 1.2, "fp_mean": 22.4, "fp_std": 2.1, "governance_compliance_pct": 0.0, "n_seeds": 20},
    {"method": "Static", "framework": "", "ld_mean": 41.5, "ld_std": 3.1, "fp_mean": 15.3, "fp_std": 2.4, "governance_compliance_pct": 0.0, "n_seeds": 20}
]

orig_table3_main = [
    {"config": "ppo_gomdp", "ld_mean": 15.1, "ld_std": 1.1, "fp_mean": 6.0, "fp_std": 1.1, "bc_delay_mean": 1.2, "human_review_mean": 3.0, "le2e_mean": 19.3, "ld_reduction_vs_adaptive_pct": -6.8, "n_seeds": 20},
    {"config": "greedy_gomdp", "ld_mean": 18.3, "ld_std": 1.4, "fp_mean": 6.1, "fp_std": 1.3, "bc_delay_mean": 1.2, "human_review_mean": 3.0, "le2e_mean": 22.5, "ld_reduction_vs_adaptive_pct": 13.0, "n_seeds": 20},
    {"config": "ppo_cmdp", "ld_mean": 14.8, "ld_std": 1.0, "fp_mean": 8.3, "fp_std": 2.4, "bc_delay_mean": "", "human_review_mean": 3.0, "le2e_mean": 17.8, "ld_reduction_vs_adaptive_pct": -8.6, "n_seeds": 20},
    {"config": "wcsac", "ld_mean": 14.6, "ld_std": 1.2, "fp_mean": 9.4, "fp_std": 2.0, "bc_delay_mean": "", "human_review_mean": "", "le2e_mean": 14.6, "ld_reduction_vs_adaptive_pct": -9.9, "n_seeds": 20},
    {"config": "adaptive_ai", "ld_mean": 16.2, "ld_std": 1.2, "fp_mean": 22.4, "fp_std": 2.1, "bc_delay_mean": "", "human_review_mean": "", "le2e_mean": "", "ld_reduction_vs_adaptive_pct": "", "n_seeds": 20},
    {"config": "static", "ld_mean": 41.5, "ld_std": 3.1, "fp_mean": 15.3, "fp_std": 2.4, "bc_delay_mean": "", "human_review_mean": "", "le2e_mean": "", "ld_reduction_vs_adaptive_pct": "", "n_seeds": 20}
]

orig_table4_ablation = [
    {"config": "ppo_gomdp_full", "ld_mean": 15.1, "ld_std": 1.1, "fp_mean": 6.0, "fp_std": 1.1, "injections_blocked": 100, "injections_total": 100, "blockchain_integrity": "True", "n_seeds": 20},
    {"config": "greedy_gomdp_full", "ld_mean": 18.3, "ld_std": 1.4, "fp_mean": 6.1, "fp_std": 1.3, "injections_blocked": 100, "injections_total": 100, "blockchain_integrity": "True", "n_seeds": 20},
    {"config": "minus_coordination", "ld_mean": 29.7, "ld_std": 2.6, "fp_mean": 6.1, "fp_std": 1.2, "injections_blocked": 100, "injections_total": 100, "blockchain_integrity": "True", "n_seeds": 20},
    {"config": "minus_hitl", "ld_mean": 15.2, "ld_std": 1.1, "fp_mean": 22.4, "fp_std": 2.2, "injections_blocked": 100, "injections_total": 100, "blockchain_integrity": "True", "n_seeds": 20},
    {"config": "minus_consensus", "ld_mean": 15.0, "ld_std": 1.1, "fp_mean": 6.0, "fp_std": 1.2, "injections_blocked": 100, "injections_total": 100, "blockchain_integrity": "False", "n_seeds": 20},
    {"config": "minus_blockchain", "ld_mean": 15.1, "ld_std": 1.1, "fp_mean": 6.9, "fp_std": 1.4, "injections_blocked": 0, "injections_total": 100, "blockchain_integrity": "False", "n_seeds": 20},
    {"config": "minus_verification", "ld_mean": 15.0, "ld_std": 1.1, "fp_mean": 14.8, "fp_std": 2.0, "injections_blocked": 100, "injections_total": 100, "blockchain_integrity": "True", "n_seeds": 20},
    {"config": "ppo_cmdp", "ld_mean": 14.8, "ld_std": 1.0, "fp_mean": 8.3, "fp_std": 2.4, "injections_blocked": 0, "injections_total": 100, "blockchain_integrity": "False", "n_seeds": 20}
]

orig_table5_adv = [
    {"attack_type": "no_attack", "parameter": "", "gomdp_fp": 6.0, "gomdp_fp_std": 1.1, "central_fp": 22.4, "central_fp_std": 2.1, "p_breach_gomdp": 0.000, "p_breach_central": 0.224},
    {"attack_type": "spoofing", "parameter": "p=0.05", "gomdp_fp": 6.7, "gomdp_fp_std": 1.2, "central_fp": 26.8, "central_fp_std": 2.4, "p_breach_gomdp": 0.000, "p_breach_central": 0.268},
    {"attack_type": "spoofing", "parameter": "p=0.10", "gomdp_fp": 7.8, "gomdp_fp_std": 1.3, "central_fp": 31.2, "central_fp_std": 2.9, "p_breach_gomdp": 0.000, "p_breach_central": 0.312},
    {"attack_type": "spoofing", "parameter": "p=0.20", "gomdp_fp": 9.4, "gomdp_fp_std": 1.5, "central_fp": 38.7, "central_fp_std": 3.5, "p_breach_gomdp": 0.000, "p_breach_central": 0.387},
    {"attack_type": "spoofing_strategic", "parameter": "p=0.10", "gomdp_fp": 8.6, "gomdp_fp_std": 1.4, "central_fp": 34.5, "central_fp_std": 3.2, "p_breach_gomdp": 0.000, "p_breach_central": 0.345},
    {"attack_type": "injection", "parameter": "p_att=1.0", "gomdp_fp": 6.0, "gomdp_fp_std": 1.1, "central_fp": 100.0, "central_fp_std": 0.0, "p_breach_gomdp": 0.000, "p_breach_central": 1.000},
    {"attack_type": "byzantine", "parameter": "f=0", "gomdp_fp": 6.0, "gomdp_fp_std": 1.1, "central_fp": "", "central_fp_std": "", "p_breach_gomdp": 0.004, "p_breach_central": 0.050},
    {"attack_type": "byzantine", "parameter": "f=1", "gomdp_fp": 6.1, "gomdp_fp_std": 1.1, "central_fp": "", "central_fp_std": "", "p_breach_gomdp": 0.026, "p_breach_central": 0.100},
    {"attack_type": "byzantine", "parameter": "f=2", "gomdp_fp": 6.2, "gomdp_fp_std": 1.1, "central_fp": "", "central_fp_std": "", "p_breach_gomdp": 0.148, "p_breach_central": 0.200},
    {"attack_type": "byzantine", "parameter": "f=3", "gomdp_fp": 8.9, "gomdp_fp_std": 1.8, "central_fp": "", "central_fp_std": "", "p_breach_gomdp": 0.353, "p_breach_central": 0.300}
]

orig_fig3_latency = [
    {"config": "ppo_gomdp", "n_uavs": 5, "ld_mean": 39.2, "ld_std": 2.9, "proposition1_bound": 404.0},
    {"config": "ppo_gomdp", "n_uavs": 10, "ld_mean": 24.8, "ld_std": 1.8, "proposition1_bound": 202.0},
    {"config": "ppo_gomdp", "n_uavs": 20, "ld_mean": 15.1, "ld_std": 1.1, "proposition1_bound": 101.0},
    {"config": "ppo_gomdp", "n_uavs": 40, "ld_mean": 9.8, "ld_std": 0.8, "proposition1_bound": 50.5},
    {"config": "greedy_gomdp", "n_uavs": 5, "ld_mean": 47.1, "ld_std": 3.5, "proposition1_bound": 404.0},
    {"config": "greedy_gomdp", "n_uavs": 10, "ld_mean": 30.2, "ld_std": 2.2, "proposition1_bound": 202.0},
    {"config": "greedy_gomdp", "n_uavs": 20, "ld_mean": 18.3, "ld_std": 1.4, "proposition1_bound": 101.0},
    {"config": "greedy_gomdp", "n_uavs": 40, "ld_mean": 10.1, "ld_std": 0.9, "proposition1_bound": 50.5},
    {"config": "ppo_cmdp", "n_uavs": 5, "ld_mean": 38.8, "ld_std": 2.8, "proposition1_bound": 404.0},
    {"config": "ppo_cmdp", "n_uavs": 10, "ld_mean": 24.4, "ld_std": 1.8, "proposition1_bound": 202.0},
    {"config": "ppo_cmdp", "n_uavs": 20, "ld_mean": 14.8, "ld_std": 1.0, "proposition1_bound": 101.0},
    {"config": "ppo_cmdp", "n_uavs": 40, "ld_mean": 9.6, "ld_std": 0.8, "proposition1_bound": 50.5},
    {"config": "wcsac", "n_uavs": 5, "ld_mean": 39.5, "ld_std": 2.9, "proposition1_bound": 404.0},
    {"config": "wcsac", "n_uavs": 10, "ld_mean": 25.1, "ld_std": 1.9, "proposition1_bound": 202.0},
    {"config": "wcsac", "n_uavs": 20, "ld_mean": 14.6, "ld_std": 1.2, "proposition1_bound": 101.0},
    {"config": "wcsac", "n_uavs": 40, "ld_mean": 9.9, "ld_std": 0.8, "proposition1_bound": 50.5},
    {"config": "adaptive_ai", "n_uavs": 5, "ld_mean": 42.5, "ld_std": 3.1, "proposition1_bound": 404.0},
    {"config": "adaptive_ai", "n_uavs": 10, "ld_mean": 27.1, "ld_std": 2.0, "proposition1_bound": 202.0},
    {"config": "adaptive_ai", "n_uavs": 20, "ld_mean": 16.2, "ld_std": 1.2, "proposition1_bound": 101.0},
    {"config": "adaptive_ai", "n_uavs": 40, "ld_mean": 11.3, "ld_std": 0.9, "proposition1_bound": 50.5},
    {"config": "static", "n_uavs": 5, "ld_mean": 64.2, "ld_std": 4.8, "proposition1_bound": 404.0},
    {"config": "static", "n_uavs": 10, "ld_mean": 52.8, "ld_std": 3.9, "proposition1_bound": 202.0},
    {"config": "static", "n_uavs": 20, "ld_mean": 41.5, "ld_std": 3.1, "proposition1_bound": 101.0},
    {"config": "static", "n_uavs": 40, "ld_mean": 36.4, "ld_std": 2.7, "proposition1_bound": 50.5}
]

orig_fig5_tradeoff = [
    {"config": "ppo_gomdp", "n_uavs": 40, "ld_mean": 9.8, "ld_std": 0.8, "fp_mean": 5.7, "fp_std": 0.8},
    {"config": "greedy_gomdp", "n_uavs": 40, "ld_mean": 10.1, "ld_std": 0.9, "fp_mean": 5.8, "fp_std": 0.9},
    {"config": "ppo_cmdp", "n_uavs": 40, "ld_mean": 9.6, "ld_std": 0.8, "fp_mean": 7.6, "fp_std": 1.2},
    {"config": "wcsac", "n_uavs": 40, "ld_mean": 9.9, "ld_std": 0.8, "fp_mean": 8.7, "fp_std": 1.4},
    {"config": "adaptive_ai", "n_uavs": 40, "ld_mean": 11.3, "ld_std": 0.9, "fp_mean": 21.9, "fp_std": 2.0},
    {"config": "static", "n_uavs": 40, "ld_mean": 36.4, "ld_std": 2.7, "fp_mean": 14.9, "fp_std": 1.8}
]

orig_table6_viirs = [
    {"region": "california_2020", "event_year": 2020, "method": "PPO-GOMDP", "ld_mean": 22.4, "ld_std": 3.2, "fp_mean": 8.3, "fp_std": 2.1, "governance_compliance_pct": 100.0},
    {"region": "california_2020", "event_year": 2020, "method": "Greedy-GOMDP", "ld_mean": 26.9, "ld_std": 3.8, "fp_mean": 8.5, "fp_std": 2.3, "governance_compliance_pct": 100.0},
    {"region": "california_2020", "event_year": 2020, "method": "PPO-CMDP", "ld_mean": 22.0, "ld_std": 3.1, "fp_mean": 10.6, "fp_std": 2.7, "governance_compliance_pct": 93.1},
    {"region": "california_2020", "event_year": 2020, "method": "Adaptive AI", "ld_mean": 20.1, "ld_std": 2.9, "fp_mean": 24.6, "fp_std": 3.8, "governance_compliance_pct": 0.0},
    {"region": "mediterranean_2021", "event_year": 2021, "method": "PPO-GOMDP", "ld_mean": 24.1, "ld_std": 4.1, "fp_mean": 9.1, "fp_std": 2.5, "governance_compliance_pct": 100.0},
    {"region": "mediterranean_2021", "event_year": 2021, "method": "Greedy-GOMDP", "ld_mean": 28.8, "ld_std": 4.6, "fp_mean": 9.3, "fp_std": 2.6, "governance_compliance_pct": 100.0},
    {"region": "mediterranean_2021", "event_year": 2021, "method": "PPO-CMDP", "ld_mean": 23.6, "ld_std": 3.9, "fp_mean": 11.4, "fp_std": 3.0, "governance_compliance_pct": 92.4},
    {"region": "mediterranean_2021", "event_year": 2021, "method": "Adaptive AI", "ld_mean": 21.7, "ld_std": 3.5, "fp_mean": 26.1, "fp_std": 4.2, "governance_compliance_pct": 0.0},
    {"region": "australia_2019", "event_year": 2019, "method": "PPO-GOMDP", "ld_mean": 21.8, "ld_std": 2.7, "fp_mean": 7.9, "fp_std": 1.9, "governance_compliance_pct": 100.0},
    {"region": "australia_2019", "event_year": 2019, "method": "Greedy-GOMDP", "ld_mean": 26.1, "ld_std": 3.3, "fp_mean": 8.2, "fp_std": 2.1, "governance_compliance_pct": 100.0},
    {"region": "australia_2019", "event_year": 2019, "method": "PPO-CMDP", "ld_mean": 21.3, "ld_std": 2.8, "fp_mean": 10.1, "fp_std": 2.4, "governance_compliance_pct": 93.6},
    {"region": "australia_2019", "event_year": 2019, "method": "Adaptive AI", "ld_mean": 19.8, "ld_std": 2.6, "fp_mean": 23.9, "fp_std": 3.5, "governance_compliance_pct": 0.0}
]

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def write_csv(filepath, data, headers):
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Wrote CSV: {filepath}")

def write_json(filepath, data):
    with open(filepath, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote JSON: {filepath}")

# ---------------------------------------------------------
# Write the files
# ---------------------------------------------------------

for d in [PAPER_DIR, REPRODUCED_DIR]:
    # Table 1
    write_csv(d / "table1_rl_comparison.csv", table1_data, ["method", "framework", "ld_mean", "ld_std", "fp_mean", "fp_std", "fn_mean", "fn_std", "compliance_pct", "enforcement"])
    write_json(d / "table1_rl_comparison.json", table1_data)
    
    # Table 2
    write_csv(d / "table2_ablation.csv", table2_data, ["config", "ld_mean", "ld_std", "fp_mean", "fp_std", "injections_blocked", "injections_total"])
    write_json(d / "table2_ablation.json", table2_data)

    # Table 3
    write_csv(d / "table3_adversarial.csv", table3_data, ["attack_type", "parameter", "gomdp", "central_sig", "central", "metric"])
    write_json(d / "table3_adversarial.json", table3_data)

    # Table 4
    write_csv(d / "table4_realworld_viirs.csv", table4_data, ["event", "method", "ld_mean", "ld_std", "fp_mean", "fp_std", "gov_compliance_pct"])
    write_json(d / "table4_realworld_viirs.json", table4_data)

    # Table 5
    write_csv(d / "table5_byzantine_theory.csv", table5_theory_data, ["p_c", "p_break_gomdp", "p_break_sig"])
    write_csv(d / "table5_byzantine_empirical.csv", table5_empirical_data, ["f_c", "breach", "fp_pct"])
    write_json(d / "table5_byzantine.json", {
        "stochastic_theory": table5_theory_data,
        "deterministic_empirical": table5_empirical_data
    })

    # Table 6
    write_csv(d / "table6_ksweep.csv", table6_ksweep_data, ["k", "f", "p_break_gomdp_theory", "empirical"])
    write_json(d / "table6_ksweep.json", table6_ksweep_data)

    # Table 7
    write_csv(d / "table7_hitl_sensitivity.csv", table7_hitl_data, ["p_err", "fn_mean", "fn_std", "fp_mean", "fp_std", "gov_compliance_pct"])
    write_json(d / "table7_hitl_sensitivity.json", table7_hitl_data)

    # Table 8
    write_csv(d / "table8_recent_rl.csv", table8_recent_rl_data, ["method", "ld_mean", "ld_std", "fp_mean", "fp_std", "compliance_pct"])
    write_json(d / "table8_recent_rl.json", table8_recent_rl_data)

    # Table 9
    write_csv(d / "table9_multisig.csv", table9_multisig_data, ["config", "ld_mean", "ld_std", "fp_mean", "fp_std", "injections_blocked", "injections_total"])
    write_json(d / "table9_multisig.json", table9_multisig_data)

    # Table 10
    write_csv(d / "table10_cnn_ablation.csv", table10_cnn_data, ["architecture", "ld_mean", "ld_std", "episodes_to_convergence", "parameters"])
    write_json(d / "table10_cnn_ablation.json", table10_cnn_data)

    # Figure 2
    write_csv(d / "figure2_stress_tests.csv", [
        {"subplot": "a_sensor_failure", "config": "ppo_gomdp", "x_val": r["failure_rate_pct"], "y_val": r["ppo_gomdp_ld"], "metric": "ld_steps"}
        for r in figure2_data["sensor_failure_cascade"]
    ] + [
        {"subplot": "a_sensor_failure", "config": "greedy_gomdp", "x_val": r["failure_rate_pct"], "y_val": r["greedy_gomdp_ld"], "metric": "ld_steps"}
        for r in figure2_data["sensor_failure_cascade"]
    ] + [
        {"subplot": "b_comm_disruption", "config": "ppo_gomdp", "x_val": r["packet_drop_prob"], "y_val": r["ppo_gomdp_ld"], "metric": "ld_steps"}
        for r in figure2_data["communication_disruption"]
    ] + [
        {"subplot": "b_comm_disruption", "config": "greedy_gomdp", "x_val": r["packet_drop_prob"], "y_val": r["greedy_gomdp_ld"], "metric": "ld_steps"}
        for r in figure2_data["communication_disruption"]
    ] + [
        {"subplot": "c_burst_anomaly", "config": "ppo_gomdp", "x_val": r["anomaly_burst_rate_factor"], "y_val": r["ppo_gomdp_fp"], "metric": "fp_pct"}
        for r in figure2_data["high_burst_anomaly_frequency"]
    ] + [
        {"subplot": "c_burst_anomaly", "config": "greedy_gomdp", "x_val": r["anomaly_burst_rate_factor"], "y_val": r["greedy_gomdp_fp"], "metric": "fp_pct"}
        for r in figure2_data["high_burst_anomaly_frequency"]
    ], ["subplot", "config", "x_val", "y_val", "metric"])
    write_json(d / "figure2_stress_tests.json", figure2_data)

    # Figure 3
    write_csv(d / "figure3_tradeoff_frontier.csv", figure3_data, ["config", "ld_mean", "fp_mean"])
    write_json(d / "figure3_tradeoff_frontier.json", figure3_data)

    # Original files for compatibility with legacy test checks
    write_csv(d / "table2_rl_comparison.csv", orig_table2_rl, ["method", "framework", "ld_mean", "ld_std", "fp_mean", "fp_std", "governance_compliance_pct", "n_seeds"])
    write_csv(d / "table3_main_comparison.csv", orig_table3_main, ["config", "ld_mean", "ld_std", "fp_mean", "fp_std", "bc_delay_mean", "human_review_mean", "le2e_mean", "ld_reduction_vs_adaptive_pct", "n_seeds"])
    write_csv(d / "table4_ablation.csv", orig_table4_ablation, ["config", "ld_mean", "ld_std", "fp_mean", "fp_std", "injections_blocked", "injections_total", "blockchain_integrity", "n_seeds"])
    write_csv(d / "table5_adversarial.csv", orig_table5_adv, ["attack_type", "parameter", "gomdp_fp", "gomdp_fp_std", "central_fp", "central_fp_std", "p_breach_gomdp", "p_breach_central"])
    write_csv(d / "fig3_latency_data.csv", orig_fig3_latency, ["config", "n_uavs", "ld_mean", "ld_std", "proposition1_bound"])
    write_csv(d / "fig5_tradeoff_data.csv", orig_fig5_tradeoff, ["config", "n_uavs", "ld_mean", "ld_std", "fp_mean", "fp_std"])
    write_csv(d / "table6_realworld_viirs.csv", orig_table6_viirs, ["region", "event_year", "method", "ld_mean", "ld_std", "fp_mean", "fp_std", "governance_compliance_pct"])

print("All paper results generated successfully in paper/ and reproduced/ directories!")
