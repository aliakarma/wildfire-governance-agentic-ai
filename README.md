# Governance-Invariant MDPs (GOMDP)

**A Formal Safety Framework and Reference Implementation for Agentic Wildfire Monitoring**

[![Tests](https://img.shields.io/badge/tests-87%20passed-brightgreen)](tests/)
[![Paper Alignment](https://img.shields.io/badge/manuscript%20alignment-354%2F354%20cells-blue)](scripts/verify_paper_alignment.py)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-MkDocs-orange)](docs/)

Implementation, experiment suite, verified empirical results, interactive web dashboard, and formal proofs for Governance-Invariant Markov Decision Processes (GOMDP).

---

## Table of Contents

- [Project Overview and Core Results](#project-overview-and-core-results)
- [System Architecture and Methodology](#system-architecture-and-methodology)
- [System Requirements](#system-requirements)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Quick Start in Three Commands](#quick-start-in-three-commands)
- [Minimal Programmatic Python Example](#minimal-programmatic-python-example)
- [Interactive Web Dashboard](#interactive-web-dashboard)
- [Repository Structure](#repository-structure)
- [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
- [Manuscript-to-Code Traceability](#manuscript-to-code-traceability)
- [Model Weights and Policy Training](#model-weights-and-policy-training)
- [Datasets and Synthetic Fallbacks](#datasets-and-synthetic-fallbacks)
- [Troubleshooting and FAQ](#troubleshooting-and-faq)
- [Contributing and Community](#contributing-and-community)
- [Citation](#citation)
- [License](#license)

---

## Project Overview and Core Results

Wildfire early warning demands rapid detection to protect human lives and forestry assets. Unverified public alerts can trigger panic, unnecessary mass evacuations, and economic disruption. Autonomous agents operating under standard constrained reinforcement learning (CMDP) enforce safety constraints only in expectation. This leaves individual decision steps open to dangerous violations during deployment.

GOMDP enforces human oversight as an invariant state-transition constraint at the environment boundary. In GOMDP, public alert actions require a cryptographically signed human authorization token. If an agent attempts to emit an alert without meeting the confidence threshold $\tau$ and obtaining valid operator authorization, the environment transitions the state to an alert-free sink state without propagating the alert.

```
Predicate: G(s_t, a_t) = [ Conf_t^(2) > tau  AND  HA_t = 1 ]
```

### Key Performance Summary

All metrics reflect evaluations across 20 random seeds (seeds 0 through 19) with exact standard deviations:

| Policy | Latency $L_d$ (steps) | False Public Alerts $F_p$ (%) | Missed Detections $F_n$ (%) | Governance Compliance (%) |
| :--- | :---: | :---: | :---: | :---: |
| **PPO-GOMDP (Ours)** | **15.1 +/- 1.1** | **6.0 +/- 1.1%** | **4.2 +/- 0.8%** | **100.0%** |
| Greedy-GOMDP | 18.3 +/- 1.4 | 6.1 +/- 1.2% | 4.4 +/- 0.9% | 100.0% |
| PPO-CMDP | 14.8 +/- 1.0 | 8.3 +/- 1.3% | 4.1 +/- 0.8% | 92.8% (violates safety) |
| Ungoverned Adaptive AI | 16.2 +/- 1.2 | 22.4 +/- 2.5% | 4.0 +/- 0.8% | 0.0% |
| Safe Shield PPO | 15.0 +/- 1.0 | 6.1 +/- 1.1% | 4.3 +/- 0.9% | 100.0% (defenseless to injection) |

### Quantitative Findings

1. **17.5% Latency Reduction**: Learned patrol coordination under PPO-GOMDP reduces detection latency from 18.3 to 15.1 steps compared to unlearned coordination baselines.
2. **100% Invariant Compliance**: By construction, any policy operating in GOMDP satisfies the governance predicate with probability negligibly close to 1 (**Theorem 1**).
3. **False Alert Suppression**: Two-stage Bayesian verification combined with human validation suppresses false public alerts from 22.4% down to 6.0%.
4. **Statistical Equivalence**: Latency matches unconstrained policies within statistical tolerance (two one-sided tests, TOST $p = 0.004$).

---

## System Architecture and Methodology

GOMDP structures autonomous wildfire monitoring as a governed multi-stage decision process.

```
+------------------------+      +-------------------------+      +---------------------------+
|  Heterogeneous Sensors | ---> |   Wildfire Digital Twin | ---> | Two-Stage Bayesian Fusion |
| (UAVs, IoT, Satellite) |      | (Grid Propagation Dynamics)|   |  (tau_1 = 0.60, tau_2 = 0.80)
+------------------------+      +-------------------------+      +---------------------------+
                                                                               |
                                                                               v
+------------------------+      +-------------------------+      +---------------------------+
|   Public Alert Issued  | <--- | PBFT Consensus (k=7,f=2)| <--- | Human Authorization Gate  |
|  (Dissemination Layer) |      | (Smart Contract Gate)   |      |        (HA_t = 1)         |
+------------------------+      +-------------------------+      +---------------------------+
```

### Formal Framework

A Governance-Invariant MDP is defined by the tuple:

$$\mathcal{M}_G = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}_G, \mathcal{R}_G, \Omega, \mathcal{O}, \gamma \rangle$$

- $\mathcal{S}$: Environment state encompassing UAV coordinates, battery reserves, ground truth fire grids, and active alerts.
- $\mathcal{A}$: Hybrid action space combining UAV movement vectors and alert submission intents.
- $\mathcal{T}_G$: Governed transition operator that evaluates the governance predicate $\mathcal{G}(s, a)$ before permitting public alerts.
- $\mathcal{R}_G$: Risk-adjusted reward balancing search efficiency, battery usage, and early detection rewards.

### Theoretical Guarantees

- **Theorem 1 (Policy-Agnostic Safety)**: For any policy $\pi$, valid smart contract execution and signature verification ensure that the probability of an unauthorized public alert is upper-bounded by $T \cdot \epsilon_{\text{sig}}$, where $\epsilon_{\text{sig}} < 2^{-128}$ under Ed25519 signatures.
- **Theorem 2 (Breach Probability Bound)**: Under Byzantine validator compromise with adversary budget $f_c$, the probability of a governance breach follows the hypergeometric cumulative distribution over quorum size $k=7$ and fault tolerance $f=2$.

---

## System Requirements

### Hardware
- **Processor**: x86_64 or ARM64 multi-core processor (8 cores recommended for 20-seed multi-processing).
- **Memory**: 4 GB RAM minimum (8 GB recommended for full episode sweeps).
- **Disk Space**: 1 GB for repository source, dependencies, and committed results.

### Software
- **Python**: 3.10, 3.11, 3.12, or 3.13.
- **Node.js**: 18+ (optional, only required if rebuilding the dashboard frontend from source).
- **PyTorch**: 2.2.1+ (optional, only required if retraining the PPO policy from scratch).

---

## Installation and Environment Setup

Choose one of three setup methods depending on your environment.

### Option A: Python venv (Standard)

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
```

### Option C: Docker (Zero Setup)

```bash
docker-compose up wildfire-gov
```

---

## Quick Start in Three Commands

You can verify the entire manuscript and launch the interactive dashboard in under four minutes:

```bash
# 1. Install package in editable mode
pip install -e ".[dev]"

# 2. Check all 354 manuscript cells against committed outputs (~30 seconds)
python scripts/verify_paper_alignment.py

# 3. Launch live web dashboard
python dashboard/run_dashboard.py --port 8123
```

Open `http://127.0.0.1:8123/` in your web browser.

---

## Minimal Programmatic Python Example

You can run a simulation episode directly in Python:

```python
from experiments.utils.runner import run_episode

# Execute a single governed episode on random seed 42
result = run_episode(
    seed=42,
    config_name="ppo_gomdp",
    grid_size=100,
    n_timesteps=1000,
    n_uavs=20,
    enable_governance=True,
    enable_hitl=True,
    enable_blockchain=True,
)

print(f"Detection latency: {result.ld:.1f} steps")
print(f"False public alert rate: {result.fp_pct:.2f}%")
print(f"Governance compliance: {result.governance_compliant}")
print(f"Adversarial injections blocked: {result.n_injections_blocked}")
```

---

## Interactive Web Dashboard

The web dashboard provides live simulation rendering, interactive parameter tuning, and cryptographic audit exploration.

```bash
python dashboard/run_dashboard.py --port 8123
```

> [!NOTE]
> Node.js is not required to run the dashboard. The application serves pre-compiled production assets directly from `dashboard/frontend/out/`.

### Operational Screens

1. **Live Simulation (`Live` Tab)**: Real-time canvas rendering active fire propagation, UAV patrol paths, battery states, and detection markers.
2. **Governance Explorer (`Governance` Tab)**: Step-by-step predicate evaluator tracking $\mathcal{G}(s_t, a_t)$, validator quorum status ($k=7, f=2$), and on-chain transaction logs.
3. **Adversarial Lab (`Adversarial` Tab)**: Real-time attack simulation for sensor spoofing ($p_{\text{spoof}}$), alert injection ($p_{\text{att}}$), and Byzantine validator faults ($f_c$).
4. **Side-by-Side Comparison (`A/B Compare` Tab)**: Synchronous evaluation comparing governed and ungoverned agents on identical random seeds.
5. **Manuscript Experiments (`Paper Experiments` Menu)**: Interactive exploration of ablation studies, scalability curves, and statistical significance tests.

Refer to [docs/dashboard_guide.md](docs/dashboard_guide.md) for full dashboard architecture specifications.

---

## Repository Structure

```
wildfire-governance-agentic-ai/
├── src/wildfire_governance/    Core Python package
│   ├── gomdp/                  GOMDP definitions and invariant checkers
│   ├── simulation/             Digital twin, fire propagation, sensor models
│   ├── agents/                 UAV agents and coordination engine
│   ├── decision/               Belief states and risk-weighted greedy policy
│   ├── verification/           Two-stage Bayesian fusion and threshold adapters
│   ├── blockchain/             Smart contracts, PBFT consensus, Ed25519 signing
│   ├── governance/             HITL authorization gate and operator models
│   ├── rl/                     PPO agent, actor-critic network, environment
│   ├── adversarial/            Sensor spoofer, alert injector, Byzantine nodes
│   └── metrics/                Statistical testing and governance metrics
├── configs/                    YAML configuration files for experiments and models
├── experiments/                Reproduction scripts for manuscript tables and figures
├── results/paper/              Committed empirical result CSVs and per-seed bundles
├── dashboard/                  Full-stack interactive web application
│   ├── backend/                FastAPI simulation streamer and WebSocket endpoints
│   └── frontend/               Next.js dashboard interface
├── docs/                       MkDocs documentation source and guides
├── notebooks/                  Jupyter walkthroughs and Google Colab notebook
├── tests/                      87 unit, integration, and smoke tests
├── scripts/                    Automation tools for verification and benchmarks
├── .github/                    Issue templates, PR template, and contributing guide
├── environment.yml             Conda environment specification
├── pyproject.toml              Python packaging configuration
├── TRAINING.md                 PPO training procedures and convergence tolerances
├── PROVENANCE.md               Manuscript to code artifact traceability map
└── README.md                   Primary documentation
```

---

## Step-by-Step Reproduction Guide

### Step 1: Run Core Tests (< 15 seconds)

Run the fast smoke suite:

```bash
pytest tests/smoke/ -v
```

Expected output: `7 passed in ~2.7s`. To run the full test suite:

```bash
pytest -q
```

Expected output: `87 passed in ~14s`.

### Step 2: Verify Manuscript Alignment Gate (< 30 seconds)

Assert that every number in every table and figure matches the committed results cell by cell:

```bash
python scripts/verify_paper_alignment.py
```

Expected output: `354 cells checked | RESULT: PASS`.

### Step 3: Fast Sanity Reproduction (~3 minutes)

Execute a fast two-seed sanity sweep over the simulation core:

```bash
# Linux / macOS
make reproduce-smoke

# Windows / Cross-platform
bash experiments/run_all.sh --smoke
```

### Step 4: Full Multi-Seed Reproduction (20 Seeds)

Run the complete multi-seed reproduction suite:

```bash
# Linux / macOS
make reproduce

# Windows / Cross-platform
bash experiments/run_all.sh --skip_training
```

### Step 5: Check Statistical Reproducibility Diffs

Verify that newly computed numbers match the frozen paper values within a 5% statistical tolerance:

```bash
bash scripts/check_reproducibility.sh
```

### Step 6: Regenerate Manuscript Figures

Generate all publication vector figures from the CSV datasets:

```bash
make figures
```

Figures are saved to `results/figures/`.

---

## Manuscript-to-Code Traceability

Every table and figure in the manuscript maps to an explicit script and result CSV:

| Manuscript Ref | Canonical ID | Result CSV Path | Generation Script | Implementation Module | Provenance Class |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 1** (Policy Comparison) | `table1_rl_comparison` | `results/paper/table1_rl_comparison.csv` | `experiments/11b_rl_comparison.py` | `src/wildfire_governance/rl/gomdp_env.py` | measured |
| **Table 2** (Ablation Study) | `table2_ablation` | `results/paper/table2_ablation.csv` | `experiments/02_ablation_study.py` | `src/wildfire_governance/gomdp/invariant_checker.py` | measured |
| **Table 4** (Config Parameters) | `table_config_parameters` | `results/paper/table_config_parameters.csv` | Specification Table | `configs/experiments/paper_main_results.yaml` | specification |
| **Table 5** (Full Metric Summary) | `table1_rl_comparison_main` | `results/paper/table1_rl_comparison_main.csv` | `experiments/01_main_comparison.py` | `src/wildfire_governance/metrics/governance_metrics.py` | measured |
| **Table 6** (Adversarial Robustness) | `table3_adversarial` | `results/paper/table3_adversarial.csv` | `experiments/09_adversarial_robustness.py` | `src/wildfire_governance/adversarial/` | measured |
| **Table 7** (VIIRS Real-World Fire) | `table4_realworld_viirs` | `results/paper/table4_realworld_viirs.csv` | `experiments/08_viirs_california.py` | `experiments/_viirs_runner.py` | measured |
| **Table 8** (Validator Compromise) | `table5_byzantine` | `results/paper/table5_byzantine_empirical.csv` | `experiments/13_byzantine_compromise.py` | `src/wildfire_governance/blockchain/consensus.py` | exact |
| **Table 9** (Validator Sweep) | `table6_ksweep` | `results/paper/table6_ksweep.csv` | `experiments/14_ksweep.py` | `src/wildfire_governance/blockchain/consensus.py` | exact |
| **Table 10** (HITL Sensitivity) | `table7_hitl_sensitivity` | `results/paper/table7_hitl_sensitivity.csv` | `experiments/06b_hitl_sensitivity.py` | `src/wildfire_governance/governance/hitl_interface.py` | measured |
| **Table 11** (Multisig Ablation) | `table9_multisig` | `results/paper/table9_multisig.csv` | `experiments/16_multisig.py` | `src/wildfire_governance/blockchain/crypto_utils.py` | exact |
| **Figure 2** ($F_p$ Scaling vs $N$) | `fig2_false_alerts` | `results/paper/fig2_false_alerts.csv` | `experiments/04b_false_alert_scaling.py` | `src/wildfire_governance/decision/greedy_policy.py` | measured |
| **Figure 3** ($L_d$ Scaling vs $N$) | `fig3_latency` | `results/paper/fig3_latency_data.csv` | `experiments/03_scalability.py` | `src/wildfire_governance/agents/coordination_engine.py` | measured |

Refer to [PROVENANCE.md](PROVENANCE.md) and [results/paper/MANIFEST.yaml](results/paper/MANIFEST.yaml) for individual seed tolerances and audit criteria.

---

## Model Weights and Policy Training

The trained model weights are included in the repository at:

```
src/wildfire_governance/rl/checkpoints/ppo_gomdp_best.pt
```

This checkpoint enables immediate evaluation of Table 1, Table 5, and Table 7 without training.

To retrain the PPO-GOMDP policy from scratch:

```bash
# Retrain PPO policy (~4 hours on 8 CPU cores)
python experiments/11_ppo_training.py --config configs/experiments/ppo_training.yaml

# Evaluate trained weights over 5 seeds
python -m wildfire_governance.rl.evaluator --use_pretrained --n_seeds 5
```

You can also train the policy in the cloud with the Google Colab notebook at [notebooks/colab_train_ppo_gomdp.ipynb](notebooks/colab_train_ppo_gomdp.ipynb). Refer to [TRAINING.md](TRAINING.md) for hyperparameter specifications.

---

## Datasets and Synthetic Fallbacks

| Dataset | Source Provider | Application | Acquisition Command |
| :--- | :--- | :--- | :--- |
| **VIIRS 375m Active Fire** | NASA FIRMS | Real fire ignition patterns | `make download-viirs` |
| **NIFC Fire Perimeters** | National Interagency Fire Center | Burn perimeter validation | `python data/scripts/download_nifc.py` |
| **GOES-16 Fire Detection** | NOAA (AWS Open Data) | Satellite stream simulation | `python data/scripts/download_goes16.py` |

The repository contains deterministic synthetic generators that activate automatically if external datasets are not downloaded. All unit tests, smoke tests, and paper alignment checks execute entirely on synthetic data without requiring network access.

---

## Troubleshooting and FAQ

### Port 8123 is already in use
Pass an alternate port with the `--port` flag:
```bash
python dashboard/run_dashboard.py --port 8124
```

### PyTorch is not installed
PyTorch is only required for training or evaluating neural network weights. All closed-form proofs, the full dashboard, smoke tests, and 80+ unit tests run on CPU with NumPy and SciPy without PyTorch installed.

### PowerShell script execution policy on Windows
If you encounter script execution restrictions when activating the virtual environment, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

---

## Contributing and Community

- Guidelines for submitting code or reporting issues: [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)
- Community standards and expectations: [.github/CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)
- Project version history: [CHANGELOG.md](CHANGELOG.md)

---

## Citation

If you use this codebase or the GOMDP framework in your research, please cite:

```bibtex
@inproceedings{gomdp2027wildfire,
  title     = {Governance-Invariant MDPs: A Framework and Formal Safety Case for Agentic Wildfire Monitoring},
  year      = {2027},
  note      = {Code and verification artifacts available at https://github.com/aliakarma/wildfire-governance-agentic-ai}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
