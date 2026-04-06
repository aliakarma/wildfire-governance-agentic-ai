#!/usr/bin/env bash
# =============================================================================
# Master reproduction script — wildfire-governance-agentic-ai
# Reproduces all paper tables and figures.
#
# Usage:
#   bash experiments/run_all.sh                  # Full run (~2–4 hours, uses pretrained PPO)
#   bash experiments/run_all.sh --smoke          # Quick sanity check (< 5 minutes)
#   bash experiments/run_all.sh --skip_training  # Skip PPO training, use checkpoint
#
# Cross-platform note:
#   Linux/macOS: bash experiments/run_all.sh
#   Windows (Git Bash or WSL): bash experiments/run_all.sh
#   Windows (PowerShell): Use WSL or Git Bash for this script.
#                          Individual experiments can be run directly with Python.
# =============================================================================

set -euo pipefail

# Ensure both src/ and repo root are on PYTHONPATH
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src:."

SMOKE=""
SKIP_TRAINING="--use_pretrained"

for arg in "$@"; do
    case $arg in
        --smoke)          SMOKE="--smoke" ;;
        --skip_training)  SKIP_TRAINING="--use_pretrained" ;;
        --full_training)  SKIP_TRAINING="" ;;
        *)                echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "=============================================="
echo " Wildfire Governance — Paper Reproduction"
echo " Mode: ${SMOKE:-full} ${SKIP_TRAINING:+with pretrained PPO}"
echo "=============================================="
echo ""

# Ensure results directory exists
mkdir -p results/runs

# ── Core simulation experiments ──────────────────────────────────────────
echo "[1/8] Main comparison (Table III)..."
python experiments/01_main_comparison.py \
    --config configs/experiments/paper_main_results.yaml $SMOKE

echo "[2/8] Ablation study (Table IV)..."
python experiments/02_ablation_study.py \
    --config configs/experiments/paper_main_results.yaml $SMOKE

echo "[3/8] Scalability — latency vs. fleet size (Figure 3)..."
python experiments/03_scalability.py \
    --config configs/experiments/scalability_uav_fleet.yaml $SMOKE

echo "[4/8] False alert rate (Figure 4)..."
python experiments/04_false_alert_rate.py \
    --config configs/experiments/paper_main_results.yaml $SMOKE

echo "[5/8] Latency-false alert tradeoff (Figure 5, N=40)..."
python experiments/05_tradeoff_frontier.py \
    --config configs/experiments/paper_main_results.yaml $SMOKE

echo "[6/8] Threshold sensitivity (Section VI-C7)..."
python experiments/06_threshold_sensitivity.py \
    --config configs/experiments/sensitivity_thresholds.yaml $SMOKE

# ── PPO-GOMDP training and RL comparison ────────────────────────────────
echo "[7a/8] PPO-GOMDP training (Table II)..."
python experiments/11_ppo_training.py \
    --config configs/experiments/ppo_training.yaml $SMOKE $SKIP_TRAINING

echo "[7b/8] RL policy comparison (Table II)..."
python experiments/11b_rl_comparison.py \
    --config configs/experiments/paper_main_results.yaml $SMOKE

# ── Adversarial and stress tests ─────────────────────────────────────────
echo "[8a/8] Adversarial robustness (Table V)..."
python experiments/09_adversarial_robustness.py \
    --config configs/experiments/adversarial_robustness.yaml $SMOKE

echo "[8b/8] Stress testing (Figure 6)..."
python experiments/10_stress_testing.py \
    --config configs/experiments/stress_testing.yaml $SMOKE

# ── Real-world VIIRS (gracefully skipped if data not present) ───────────
echo ""
echo "[Optional] Real-world VIIRS experiments (Table VI)..."
python experiments/08_viirs_california.py \
    --config configs/experiments/realworld_viirs.yaml $SMOKE \
    || echo "   -> Skipped (run 'make download-viirs' to enable)"

echo ""
echo "=============================================="
echo " All experiments complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  Verify results:  bash scripts/check_reproducibility.sh"
echo "  Generate figures: make figures"
echo "  Results saved to: results/runs/"
echo ""
