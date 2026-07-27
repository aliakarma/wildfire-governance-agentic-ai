# Governance-Invariant MDPs: A Framework and Formal Safety Case for Agentic Wildfire Monitoring

**Anonymous Submission — AAAI 2027 Reviewer & Auditor Artifact Package**  
*Anonymous Affiliation*

[![Paper](https://img.shields.io/badge/Paper-AAAI--2027%20submission-blue)](Paper/AAAI/Wildfire.pdf)
[![Results](https://img.shields.io/badge/results-354%20cells%20verified-brightgreen)](scripts/verify_paper_alignment.py)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

> Wildfire early warning must be fast enough to save lives yet accountable enough to trust: a false public alert can trigger a needless mass evacuation, while a missed detection can be fatal, so the authority to issue an alert cannot be ceded to an unaccountable autonomous agent. Constrained reinforcement learning (RL), the standard tool for safe autonomy, enforces safety only in expectation and thus permits violations. We introduce the **Governance-Invariant MDP (GOMDP)**, which makes mandatory human authorization a hard, auditable, non-repudiable constraint enforced at the environment level as a cryptographic state-transition invariant rather than a policy penalty. Instantiated for wildfire monitoring with a Hyperledger Fabric smart contract and evaluated on three real fire events derived from NASA VIIRS/FIRMS data, GOMDP lets no public alert issue without a valid human authorization: under stated assumptions, any policy, however suboptimal or adversarial, satisfies the governance predicate with probability negligibly close to one (**Theorem 1**), and a closed-form breach bound (**Theorem 2**) delimits when Byzantine consensus outperforms a single verifier. A layered decomposition isolates what each enforcement layer buys. Across 20 seeds, PPO-GOMDP cuts detection latency by **17.5%** versus a training-free baseline (18.3 → 15.1 steps) at **100% governance compliance** and cuts false public alerts from **22.4% to 6.0%**, with statistically equivalent latency to constrained RL (TOST, *p* = 0.004) and robustness under adversarial stress. GOMDP is a deployable, auditable accountability layer for safety-critical civic AI.

---

## Guide for AAAI Reviewers & Reproducibility Auditors

This repository contains the complete implementation, experiment suite, interactive web dashboard, and verification tools accompanying the manuscript.

### Key Verification Highlights
1. **Cell-by-Cell Manuscript Verification:** Every number in the paper's tables and figures (354 cells total) is transcribed and verified against live computation via `python scripts/verify_paper_alignment.py`.
2. **Deterministic & Runnable Code:** All metrics are evaluated over 20 random seeds with exact standard deviations, paired $t$-tests, and TOST equivalence tests.
3. **Interactive Simulation Dashboard:** A full-stack web application allows reviewers to run live simulation episodes step-by-step, mount adversarial attacks, inspect smart contract transactions, and perform side-by-side A/B comparisons.

---

## Quick Start (Environment Setup)

### System Requirements
- **Python:** 3.10 or higher
- **Node.js:** 18+ (only required for running the interactive web dashboard)
- **OS:** Linux, macOS, or Windows

---

### Option A — Conda (Recommended)

```bash
# Bash (Linux/macOS)
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py
make test-smoke
```

```powershell
# PowerShell (Windows)
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py
python -m pytest tests/smoke/ -v --no-cov --timeout=60
```

---

### Option B — pip + venv

```bash
# Bash (Linux/macOS)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py
make test-smoke
```

```powershell
# PowerShell (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e ".[dev]"
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py
python -m pytest tests/smoke/ -v --no-cov --timeout=60
```

---

### Option C — Docker (Zero Setup)

```bash
# Bash
docker-compose up wildfire-gov
```

---

## Step-by-Step Reviewer Reproduction Guide

### Step 1: Verify Paper Alignment Gate (< 30 Seconds)
Assert that every committed CSV file matches the manuscript cell-by-cell:

```bash
python scripts/verify_paper_alignment.py
```
*Expected Output:* `354 cells checked | RESULT: PASS — every committed value matches the manuscript`

---

### Step 2: Run Smoke & Integration Tests (< 2 Minutes)
Verify core GOMDP environment invariants, Ed25519 signing, and fire propagation mechanics:

```bash
python -m pytest tests/smoke/ -v --no-cov
```
*Expected Output:* `7 passed in ~6 seconds`

---

### Step 3: Reproduce Paper Results & Tables

#### A. Fast Sanity Reproduction (2 seeds x 100 timesteps, ~3 minutes)
```bash
# Linux/macOS
make reproduce-smoke

# PowerShell / Cross-Platform
bash experiments/run_all.sh --smoke
```

#### B. Full Multi-Seed Paper Reproduction (20 seeds, ~2–4 hours on CPU)
```bash
# Linux/macOS
make reproduce

# PowerShell / Cross-Platform
bash experiments/run_all.sh --skip_training
```

---

### Step 4: Check Reproducibility Diff Against Manuscript
Verify that fresh runs match the committed manuscript values within a 5% statistical tolerance:

```bash
bash scripts/check_reproducibility.sh
```

---

### Step 5: Regenerate All Manuscript Figures
Regenerate all figures from `results/paper/` CSV data:

```bash
make figures
# Output figures saved in results/figures/
```

---

## Step-by-Step Interactive Web Dashboard Guide

The repository includes a web dashboard that streams live simulation steps over WebSocket, rendering the UAV swarm search, Bayesian verification, and smart contract ledger in real time.

### Launching the Dashboard (One Command)

```bash
# 1. Install dashboard dependencies
pip install -r dashboard/backend/requirements.txt

# 2. Launch backend API + pre-built frontend server
python dashboard/run_dashboard.py --port 8123
```

Open your browser at: **`http://127.0.0.1:8123/`**

---

### Key Dashboard Screens for Reviewers

1. **Live Simulation (`Live` Tab):**
   - Interactive grid canvas with UAV flight paths, battery status, heat intensity, and live alert streams.
   - Adjust active fleet size $N$, sector partition $Z$, confidence threshold $\tau$, and coordination policy in real time.
2. **Governance Explorer (`Governance` Tab):**
   - Real-time predicate inspector evaluating $\mathcal{G}(s_t, a_t) = [\Conf_t^{(2)} > \tau \land \HA_t = 1]$.
   - PBFT validator ring visualizer showing BFT safety thresholds ($k=7, f=2$) and on-chain Ed25519 transaction hashes.
3. **Adversarial Lab (`Adversarial` Tab):**
   - Mount live attacks: sensor spoofing ($p_{\text{spoof}}$), alert injection ($p_{\text{att}}$), and Byzantine validator compromise ($f_c$).
   - Real-time verification of Theorem 2's breach bounds.
4. **Side-by-Side Comparison (`A/B Compare` Tab):**
   - Run two policies on the **exact same random seed** side-by-side (e.g., Governed PPO-GOMDP vs. Ungoverned Adaptive AI) to observe 100% compliance enforcement vs. unconstrained false alerts.
5. **Paper Experiments Dropdown (`Paper Experiments` Menu):**
   - **Ablation (Table 2):** Component knockout metrics.
   - **Scalability (Figs 2 & 3):** False-alert and latency scaling vs. fleet size $N \in \{5, 10, 20, 40\}$.
   - **Learning (Fig 3):** PPO-GOMDP validation learning curve across training episodes.
   - **HITL (Table 7):** Sensitivity to operator error rate $p_{\text{err}} \in \{0.05, 0.10, 0.15, 0.20\}$.
   - **Statistics (Sec. 6.2):** Statistical significance tests ($t$-test, Holm–Bonferroni, and TOST equivalence $p=0.004$).
   - **All Experiments:** Interactive registry of all manuscript artifacts and downloadable CSVs.

---

## Manuscript-to-Code Mapping & Provenance

The table below maps each table and figure in the manuscript to its committed data, script, and code implementation:

| Manuscript Item | Canonical ID | CSV Path | Experiment Script | Key Implementation File |
| :--- | :--- | :--- | :--- | :--- |
| **Table 1** (Policy Comparison) | `table1_rl_comparison` | `results/paper/table1_rl_comparison.csv` | `experiments/11b_rl_comparison.py` | `src/wildfire_governance/rl/gomdp_env.py` |
| **Table 2** (Ablation Study) | `table2_ablation` | `results/paper/table2_ablation.csv` | `experiments/02_ablation_study.py` | `src/wildfire_governance/gomdp/invariant_checker.py` |
| **Table 4** (Config Parameters) | `table_config_parameters` | `results/paper/table_config_parameters.csv` | Specification | `src/wildfire_governance/simulation/grid_environment.py` |
| **Table 5** (Full Metric Summary) | `table1_rl_comparison_main` | `results/paper/table1_rl_comparison_main.csv` | `experiments/01_main_comparison.py` | `src/wildfire_governance/metrics/evaluator.py` |
| **Table 6** (Adversarial Robustness) | `table3_adversarial` | `results/paper/table3_adversarial.csv` | `experiments/09_adversarial_robustness.py` | `src/wildfire_governance/adversarial/spoofer.py` |
| **Table 7** (VIIRS Real-World Data) | `table4_realworld_viirs` | `results/paper/table4_realworld_viirs.csv` | `experiments/08_viirs_california.py` | `experiments/_viirs_runner.py` |
| **Table 8** (Validator Compromise) | `table5_byzantine` | `results/paper/table5_byzantine_empirical.csv` | `experiments/13_byzantine_compromise.py` | `src/wildfire_governance/blockchain/consensus.py` |
| **Table 9** (Validator Sweep) | `table6_ksweep` | `results/paper/table6_ksweep.csv` | `experiments/14_ksweep.py` | `src/wildfire_governance/blockchain/consensus.py` |
| **Table 10** (HITL Sensitivity) | `table7_hitl_sensitivity` | `results/paper/table7_hitl_sensitivity.csv` | `experiments/06b_hitl_sensitivity.py` | `src/wildfire_governance/governance/hitl_interface.py` |
| **Table 11** (Multisig Ablation) | `table9_multisig` | `results/paper/table9_multisig.csv` | `experiments/16_multisig.py` | `src/wildfire_governance/blockchain/crypto_utils.py` |
| **Sec. 6.2** (Statistical Tests) | `statistical_tests` | `results/paper/statistical_tests.csv` | `scripts/generate_all_paper_results.py` | `src/wildfire_governance/metrics/statistical_tests.py` |
| **Figure 2** ($F_p$ vs $N$) | `fig2_false_alerts` | `results/paper/fig2_false_alerts.csv` | `experiments/04b_false_alert_scaling.py` | `src/wildfire_governance/decision/greedy_policy.py` |
| **Figure 3** ($L_d$ vs $N$) | `fig3_latency` | `results/paper/fig3_latency_data.csv` | `experiments/03_scalability.py` | `src/wildfire_governance/agents/coordination_engine.py` |

---

## Repository Code Architecture

```
wildfire-governance-agentic-ai/
├── configs/                     YAML experiment configurations
├── data/                        Dataset scripts & synthetic fallbacks
├── experiments/                 Reproducible experiment scripts (01–16)
├── Paper/                       AAAI manuscript source & styles
│   └── AAAI/                    Wildfire.tex, aaai2027.sty, references.bib
├── results/                     Committed CSVs, JSONs, and seed bundles
│   └── paper/                   354 verified cells matching manuscript
├── scripts/                     Verification and build automation scripts
├── src/wildfire_governance/
│   ├── gomdp/                   GOMDP framework (Definition 1, Theorems 1–2)
│   ├── simulation/              Wildfire grid environment & fire propagation
│   ├── agents/                  UAV agents & coordination engine
│   ├── decision/                Belief state & greedy policy
│   ├── verification/            Two-stage Bayesian fusion pipeline
│   ├── blockchain/              Smart contract & Ed25519 cryptographic gate
│   ├── governance/              HITL authorization interface & oracle model
│   ├── rl/                      PPO-GOMDP environment & training logic
│   ├── adversarial/             Sensor spoofer, alert injector, Byzantine sim
│   └── metrics/                 Statistical tests & evaluation metrics
└── tests/                       Smoke, unit, and integration tests
```

---

## Dataset Setup

| Dataset | Source Provider | Application | Download Command |
| :--- | :--- | :--- | :--- |
| **VIIRS 375m Active Fire** | NASA FIRMS | Real-world active fire events | `make download-viirs` |
| **NIFC Fire Perimeters** | NIFC | True alarm validation labels | `python data/scripts/download_nifc.py` |
| **GOES-16 Fire Detection** | NOAA (AWS S3) | Satellite feed simulation | `python data/scripts/download_goes16.py` |

*Note:* All code automatically falls back to deterministic synthetic data if optional real-world datasets are not downloaded. Smoke tests and unit tests run entirely on synthetic data without requiring API keys.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
