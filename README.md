# Governance-Constrained Agentic AI: A Governance-Invariant MDP Framework with Blockchain-Enforced Human Oversight for Safety-Critical Wildfire Monitoring

**Ali Akarma · Toqeer Ali Syed · Salman Jan · Hammad Muneer · Abdul Khadar Jilani**
*Islamic University of Madinah · Arab Open University–Bahrain · Islamia University of Bahawalpur · University of Technology Bahrain*

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TII-blue)](https://doi.org/10.1109/TII.2025.XXXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-2512.XXXXX-red)](https://arxiv.org/abs/2512.XXXXX)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/aliakarma/wildfire-governance-agentic-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/aliakarma/wildfire-governance-agentic-ai/actions)
[![codecov](https://codecov.io/gh/aliakarma/wildfire-governance-agentic-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/aliakarma/wildfire-governance-agentic-ai)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Abstract

> Safety-critical agentic AI systems require a qualitatively stronger form of constraint satisfaction than existing constrained Markov decision process (CMDP) approaches, which enforce safety constraints via Lagrangian relaxation and therefore permit violations in expectation. We introduce the **Governance-Invariant MDP (GOMDP)**—a formal framework in which safety constraints are enforced at the environment level via cryptographic state-transition invariants rather than as soft policy penalties. We prove that any policy, including arbitrarily suboptimal ones, operating within a GOMDP satisfies the governance predicate with probability one (**Theorem 1: Policy-Agnostic Safety**). PPO-GOMDP reduces detection latency by 17.5% relative to the greedy baseline while maintaining **100% governance compliance**, versus 92.8% for standard constrained RL. False public alert rates are reduced from 22.4% to 6.1% (*p* < 0.01). Adversarial stress tests confirm the GOMDP invariant holds under sensor spoofing, Byzantine faults, and up to 20% packet loss.

---

## What Is Novel?

| Prior Work | Safety Guarantee | Violation Rate | Non-Repudiation | Adversarial Tolerance |
|------------|-----------------|----------------|-----------------|----------------------|
| CMDP / CPO (Altman 1999; Achiam 2017) | In-expectation | 5–15% | None | None |
| Safe Shielding (Alshiekh 2018) | Per-trajectory | ~0% | None | Centralised only |
| **GOMDP (Ours)** | **Per-trajectory, prob. 1** | **0%** | **Cryptographic** | **Byzantine-fault-tolerant** |

The GOMDP enforces safety at the environment boundary via a cryptographic invariant; any policy—random, greedy, or trained—satisfies the governance predicate by construction (Theorem 1). Safety is decoupled from optimality (Corollary 1).

Note: In the experiments and codebase the smart-contract and blockchain enforcement are simulated in software (a Python model of a smart contract and consensus). Theorem 1 is a formal property of the GOMDP model; our implementation exercises a simulated cryptographic enforcement mechanism for reproducible experiments and stress tests. Interpret claims about "environment-level cryptographic enforcement" as properties of the GOMDP model together with the simulated smart-contract implementation used for evaluation.

---

## Quick Start

### Option A — Conda (Recommended)

```bash
# Bash (Linux/macOS)
git clone https://github.com/aliakarma/wildfire-governance-agentic-ai.git
cd wildfire-governance-agentic-ai
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py
make test-smoke
```

```powershell
# PowerShell (Windows)
git clone https://github.com/aliakarma/wildfire-governance-agentic-ai.git
Set-Location wildfire-governance-agentic-ai
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py
python -m pytest tests/smoke/ -v --no-cov --timeout=60
```

### Option B — pip + venv

```bash
# Bash
git clone https://github.com/aliakarma/wildfire-governance-agentic-ai.git
cd wildfire-governance-agentic-ai
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
pip install -r requirements-dev.txt
pip install -e ".[dev]"
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py
make test-smoke
```

```powershell
# PowerShell (Windows)
git clone https://github.com/aliakarma/wildfire-governance-agentic-ai.git
Set-Location wildfire-governance-agentic-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e ".[dev]"
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py
python -m pytest tests/smoke/ -v --no-cov --timeout=60
```

### Option C — Docker (Zero Setup)

```bash
# Bash
git clone https://github.com/aliakarma/wildfire-governance-agentic-ai.git
cd wildfire-governance-agentic-ai
docker-compose up wildfire-gov
```

```powershell
# PowerShell (Windows)
git clone https://github.com/aliakarma/wildfire-governance-agentic-ai.git
Set-Location wildfire-governance-agentic-ai
docker-compose up wildfire-gov
```

### Option D — Interactive Dashboard

A live web dashboard that runs the **real** simulation one timestep at a time and
streams it to an animated UI — watch the UAV swarm search, communicate, verify a
detection, and encircle a slowly spreading wildfire, alongside the governance
ledger, adversarial lab, and live benchmarks. Requires the repo's Python
environment (Option A or B) plus **Node.js 18+**.

```bash
# Bash (Linux/macOS)
pip install -r dashboard/backend/requirements.txt   # backend extras (FastAPI, uvicorn, …)

cd dashboard/frontend
npm install
npm run build                                        # builds dashboard/frontend/out/
cd ../..

python -m uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8123

# → open http://127.0.0.1:8123/
```

```powershell
# PowerShell (Windows)
pip install -r dashboard/backend/requirements.txt

Set-Location dashboard/frontend
npm install
npm run build
Set-Location ../..

python -m uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8123
# → open http://127.0.0.1:8123/
```

> On Windows, some reserved port ranges (e.g. 8000) may be blocked; pick another
> port such as 8123. For hot-reload dev mode (API + Next.js dev server in two
> terminals) or a one-command Docker run, see [`dashboard/README.md`](dashboard/README.md).

---

## Reproduce All Paper Results

```bash
# Ensure checkpoint + synthetic fallback data are present
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py

# Bash — full reproduction (~2–4 hours on 8 CPU cores, uses pre-trained PPO)
make reproduce

# Bash — smoke test version (< 5 minutes, 2 seeds × 100 steps)
make reproduce-smoke

# Verify results match paper within 5% tolerance
bash scripts/check_reproducibility.sh

# Regenerate all paper figures from results/paper/ CSVs
make figures
```

```powershell
# Ensure checkpoint + synthetic fallback data are present
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py

# PowerShell — full reproduction
bash experiments/run_all.sh --skip_training

# PowerShell — smoke
bash experiments/run_all.sh --smoke
```

---

## Dataset Setup

| Dataset | Provider | Used For | Download |
|---------|----------|----------|----------|
| VIIRS 375m Active Fire | NASA FIRMS | Ground-truth fire detection | `make download-viirs` |
| NIFC Fire Perimeters | NIFC | True alarm labels for Fp | `python data/scripts/download_nifc.py` |
| GOES-16 Fire Detection | NOAA (free S3) | Satellite feed simulation | `python data/scripts/download_goes16.py` |

See [`data/README.md`](data/README.md) for full instructions, API key setup, and checksums.

> **Note:** All experiments fall back to synthetic data automatically if real datasets are not downloaded. The smoke test and unit tests use only synthetic data and require no API keys.

---

## Results Summary

All quantitative results and data plots presented in the paper are fully detailed in the [results/paper](results/paper) directory. For details on the exact scripts, configurations, and seeds that generated each result, see [PROVENANCE.md](PROVENANCE.md). Below is a summary of the final paper results.

### Table 1 — Policy Comparison ($N=20$ UAVs, 20 seeds)
*File paths*: [results/paper/table1_rl_comparison.csv](results/paper/table1_rl_comparison.csv) / [results/paper/table1_rl_comparison.json](results/paper/table1_rl_comparison.json)

| Method | $L_d$ (steps) | $F_p$ (%) | $FN_r$ (%) | Compliance | Enforcement |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **PPO-GOMDP** | **15.1 ± 1.1** | **6.0 ± 1.1** | **2.1 ± 0.9** | **100.0%** | Cryptographic |
| Greedy-GOMDP | 18.3 ± 1.4 | 6.1 ± 1.3 | 2.3 ± 1.0 | **100.0%** | Cryptographic |
| Central+Sig | 15.0 ± 1.1 | 6.0 ± 1.2 | — | **100.0%** | Signature only |
| Shield-PPO | 15.2 ± 1.2 | 6.2 ± 1.3 | — | 100.0% | Logical |
| SafeLayer | 14.9 ± 1.1 | 7.0 ± 1.6 | — | 98.4% | Learned |
| PPO-CMDP | 14.8 ± 1.0 | 8.3 ± 2.4 | 2.6 ± 1.0 | 92.8% | Lagrangian |
| WCSAC | 14.6 ± 1.2 | 9.4 ± 2.0 | 3.8 ± 1.4 | 90.6% | Lagrangian |
| Adaptive AI | 16.2 ± 1.2 | 22.4 ± 2.1 | 0.9 ± 0.5 | 0.0% | None |
| Static | 41.5 ± 3.1 | 15.3 ± 2.4 | 1.8 ± 0.8 | 0.0% | None |

### Table 2 — Ablation Study ($N=20$, 20 seeds)
*File paths*: [results/paper/table2_ablation.csv](results/paper/table2_ablation.csv) / [results/paper/table2_ablation.json](results/paper/table2_ablation.json)

| Configuration | $L_d$ (steps) | $F_p$ (%) | Injections Blocked |
| :--- | :---: | :---: | :---: |
| **PPO-GOMDP (full)** | **15.1 ± 1.1** | **6.0 ± 1.1** | **100/100** |
| Greedy-GOMDP (full) | 18.3 ± 1.4 | 6.1 ± 1.3 | **100/100** |
| *− Adaptive coordination* | 29.7 ± 2.6 | 6.1 ± 1.2 | **100/100** |
| *− HITL authorization* | 15.2 ± 1.1 | 22.4 ± 2.2 | **100/100** |
| *− Consensus (Central+Sig)* | 15.0 ± 1.1 | 6.0 ± 1.2 | **100/100** |
| *− All authentication* | 15.1 ± 1.1 | 6.9 ± 1.4 | 0/100 |
| *− Multi-stage verification* | 15.0 ± 1.1 | 14.8 ± 2.0 | **100/100** |
| **PPO-CMDP (no blockchain)** | 14.8 ± 1.0 | 8.3 ± 2.4 | 0/100 |

### Table 3 — Adversarial Robustness ($N=20$, 20 seeds)
*File paths*: [results/paper/table3_adversarial.csv](results/paper/table3_adversarial.csv) / [results/paper/table3_adversarial.json](results/paper/table3_adversarial.json)

| Attack / Condition | Param. | GOMDP $F_p$ (%) | Cent.+Sig $F_p$ (%) | Central $F_p$ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **No attack** | — | 6.0 | 6.0 | 22.4 |
| **Spoofing (i.i.d.)** | $p=0.05$ | 6.7 | 6.7 | 26.8 |
| **Spoofing (i.i.d.)** | $p=0.10$ | 7.8 | 7.9 | 31.2 |
| **Spoofing (i.i.d.)** | $p=0.20$ | 9.4 | 9.5 | 38.7 |
| **Spoofing (strategic)** | $p=0.10$ | 8.6 | 8.7 | 34.5 |
| **Alert injection (success)** | $p_{att}=1$ | 0/100 | 0/100 | 100/100 |

### Table 4 — VIIRS-Data Simulation Validation ($N=20$, 20 seeds)
*File paths*: [results/paper/table4_realworld_viirs.csv](results/paper/table4_realworld_viirs.csv) / [results/paper/table4_realworld_viirs.json](results/paper/table4_realworld_viirs.json)

| Event | Method | $L_d$ (steps) | $F_p$ (%) | Gov. Compliance |
| :--- | :--- | :---: | :---: | :---: |
| **California '20** | PPO-GOMDP | 22.4 ± 3.2 | 8.3 ± 2.1 | 100% |
| | Greedy-GOMDP | 26.9 ± 3.8 | 8.5 ± 2.3 | 100% |
| | PPO-CMDP | 22.0 ± 3.1 | 10.6 ± 2.7 | 93.1% |
| | Adaptive AI | 20.1 ± 2.9 | 24.6 ± 3.8 | 0% |
| **Mediterranean '21** | PPO-GOMDP | 24.1 ± 4.1 | 9.1 ± 2.5 | 100% |
| | Greedy-GOMDP | 28.8 ± 4.6 | 9.3 ± 2.6 | 100% |
| | PPO-CMDP | 23.6 ± 3.9 | 11.4 ± 3.0 | 92.4% |
| | Adaptive AI | 21.7 ± 3.5 | 26.1 ± 4.2 | 0% |
| **NSW '19–20** | PPO-GOMDP | 21.8 ± 2.7 | 7.9 ± 1.9 | 100% |
| | Greedy-GOMDP | 26.1 ± 3.3 | 8.2 ± 2.1 | 100% |
| | PPO-CMDP | 21.3 ± 2.8 | 10.1 ± 2.4 | 93.6% |
| | Adaptive AI | 19.8 ± 2.6 | 23.9 ± 3.5 | 0% |

### Supplementary Results and Sensitivity Analysis

Additional experimental data points and detailed configurations can be found directly in [results/paper/README.md](results/paper/README.md), including:
* **Table 5 — Validator/Verifier Compromise** (Resilience bounds under validator corruption)
* **Table 6 — Validator-Count Sweep** (BFT safety thresholds)
* **Table 7 — HITL Error Rate Sensitivity** (Operator workload limits)
* **Table 8 — Recent Safe RL Comparators** (Comparisons with SafeDreamer and CCPO)
* **Table 9 — multisig Ablation** (Threshold signatures vs Byzantine consensus)
* **Table 10 — CNN-Architecture Ablation** (Performance/efficiency tradeoffs)
* **Figure 2 & Figure 3 Quantitative Data** (Stress test and tradeoff frontier plots)

---

## Repository Structure

```
wildfire-governance-agentic-ai/
├── configs/               YAML experiment configurations
├── data/                  Data download scripts + synthetic data
├── experiments/           Reproducible experiment scripts (01–12)
├── notebooks/             Interactive demos and analysis
├── results/paper/         Pre-committed paper result CSVs
├── scripts/               Utility shell scripts
├── src/wildfire_governance/
│   ├── gomdp/             GOMDP framework (Definition 1, Theorems 1–2)
│   ├── simulation/        Wildfire grid environment + fire propagation
│   ├── agents/            UAV agents + coordination engine
│   ├── decision/          Belief state, greedy policy
│   ├── verification/      Two-stage Bayesian fusion pipeline
│   ├── blockchain/        Hyperledger Fabric simulation + smart contract
│   ├── governance/        HITL interface + alert dissemination
│   ├── rl/                PPO-GOMDP agent + training + checkpoints
│   ├── adversarial/       Sensor spoofer, alert injector, Byzantine sim
│   ├── metrics/           Ld, Fp, Le2e, Holm-Bonferroni tests
│   └── utils/             Config, logging, reproducibility
└── tests/                 Unit, integration, smoke tests
```

---


## Acknowledgements

We thank NASA FIRMS for VIIRS data access, NIFC for historical fire perimeter data, NOAA for GOES-16 open data via AWS, and ECMWF/Copernicus for ERA5 reanalysis. Compute resources provided by the AI Center, Islamic University of Madinah.

## License

MIT — see [LICENSE](LICENSE).
