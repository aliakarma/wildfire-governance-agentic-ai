# Reproducibility Guide

## Hardware Requirements

| | Minimum | Recommended |
|---|---------|-------------|
| CPU cores | 4 | 16+ |
| RAM | 8 GB | 32 GB |
| Storage | 5 GB | 20 GB |
| GPU | Not required | Optional (PPO training only) |

## Estimated Runtimes

| Experiment | Script | Time (8 cores) |
|-----------|--------|----------------|
| Main comparison (Table III) | `01_main_comparison.py` | ~25 min |
| Ablation (Table IV) | `02_ablation_study.py` | ~40 min |
| Scalability (Fig. 3) | `03_scalability.py` | ~50 min |
| False alert rate (Fig. 4) | `04_false_alert_rate.py` | ~25 min |
| Tradeoff frontier (Fig. 5) | `05_tradeoff_frontier.py` | ~15 min |
| Adversarial (Table V) | `09_adversarial_robustness.py` | ~30 min |
| Stress testing (Fig. 6) | `10_stress_testing.py` | ~40 min |
| RL comparison (Table II) | `11b_rl_comparison.py` | ~20 min |
| **Full reproduction** | `run_all.sh --skip_training` | **~2–4 hours** |

## Step-by-Step Reproduction

### Step 1: Install

```bash
# Bash
pip install -e .
export PYTHONPATH=src

# PowerShell
pip install -e .
$env:PYTHONPATH = "src"
```

### Step 2: Smoke Test (30 seconds)

```bash
python -m pytest tests/smoke/ -v --timeout=60
```

### Step 3: Reproduce All Results (~2–4 hours)

```bash
# Uses pre-trained PPO checkpoint — no training needed
bash experiments/run_all.sh --skip_training

# OR: Full run including PPO training (~6–10 hours)
bash experiments/run_all.sh --full_training
```

### Step 4: Verify Against Paper Values

```bash
bash scripts/check_reproducibility.sh
```

Expected output:
```
[PASS] greedy_gomdp/ld_mean: paper=18.30, run=18.XX, err=X.X%
[PASS] greedy_gomdp/fp_mean: paper=6.10, run=6.XX, err=X.X%
All checks PASSED — results reproduce within 5% tolerance.
```

### Step 5: Regenerate Figures

```bash
bash scripts/generate_paper_figures.sh
# Figures saved to results/figures/
```

## Numerical Variation

Results are stochastic — 20 random seeds reduce variance but do not eliminate it.
Expected variation across platforms (NumPy MKL vs OpenBLAS): ±5% relative.

The GOMDP governance invariant (Theorem 1) is deterministic: **100% compliance
is not a statistical claim** — it is enforced by the cryptographic smart contract
and will reproduce exactly regardless of hardware or seed.

## Additional Missing Experiments (Multi-Seed + Stepwise Invariants)

### Experiment 1: Multi-seed PPO training reproducibility (seeds 0–4)

Trains PPO-GOMDP from scratch across independent seeds and aggregates learning curve
variance plus final-episode reward distributions.

```bash
python experiments/exp1_multiseed_ppo_training.py --config configs/experiments/ppo_training.yaml --seed_start 0 --seed_end 4
```

Outputs:
- `results/experiments/<hash>/ppo_multiseed_learning_curves.csv`
- `results/experiments/<hash>/ppo_multiseed_reward_mean_std_by_episode.csv`
- `results/experiments/<hash>/ppo_multiseed_final_rewards.csv`

If episode 500 exists, a quick convergence check is written to:
- `results/experiments/<hash>/ppo_multiseed_episode500_check.txt`

### Experiment 2: Stepwise GovernanceInvariantChecker behavior without blockchain (CMDP)

Logs `theorem1_step_satisfied` per-step (and violation reasons) for CMDP/no-blockchain runs.

```bash
python experiments/exp2_cmdp_invariant_stepwise.py --config configs/experiments/paper_main_results.yaml --n_seeds 20
```

Outputs:
- `results/runs/<hash>/cmdp_stepwise_invariant.csv`
- `results/runs/<hash>/cmdp_episode_invariant_summary.csv`

### Experiment 3: PPO training-seed ablation vs Greedy baseline

Evaluate multiple PPO checkpoints (trained with different seeds) against Greedy-GOMDP and report
mean ± std of \(L_d\) improvement (%). This tests whether the 17.5% claim is robust across training seeds.

```bash
# After running Experiment 1, point this at its checkpoints directory:
python experiments/exp3_multiseed_rl_ablation.py --checkpoints_dir results/experiments/<hash>/checkpoints --n_eval_seeds 20
```

## Pre-Committed Paper Results

All paper result CSVs are committed to `results/paper/`. To regenerate figures
without running experiments:

```bash
# Bash / PowerShell
bash scripts/generate_paper_figures.sh
```
