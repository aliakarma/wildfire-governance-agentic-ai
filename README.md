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
make test-smoke
```

```powershell
# PowerShell (Windows)
git clone https://github.com/aliakarma/wildfire-governance-agentic-ai.git
Set-Location wildfire-governance-agentic-ai
conda env create -f environment.yml
conda activate wildfire-gov
pip install -e ".[dev]"
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

---

## Reproduce All Paper Results

```bash
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
| ERA5 Reanalysis | ECMWF/Copernicus | Meteorological forcing | `python data/scripts/download_era5.py` |
| MTBS Burn Severity | USGS/USFS | Fuel load maps | `python data/scripts/download_mtbs.py` |

See [`data/README.md`](data/README.md) for full instructions, API key setup, and checksums.

> **Note:** All experiments fall back to synthetic data automatically if real datasets are not downloaded. The smoke test and unit tests use only synthetic data and require no API keys.

---

## Results Summary

### Table II — RL Policy Comparison (N=20 UAVs, 20 seeds)

| Method | Ld (steps) | Fp (%) | Gov. Compliance | Framework |
|--------|-----------|--------|-----------------|-----------|
| **PPO-GOMDP** | **15.1 ± 1.1** | **6.0 ± 1.1** | **100.0%** | GOMDP |
| Greedy-GOMDP | 18.3 ± 1.4 | 6.1 ± 1.3 | **100.0%** | GOMDP |
| PPO-CMDP | 14.8 ± 1.0 | 8.3 ± 2.4 | 92.8% | CMDP |
| Adaptive AI | 16.2 ± 1.2 | 22.4 ± 2.1 | 0.0% | None |
| Static | 41.5 ± 3.1 | 15.3 ± 2.4 | 0.0% | None |

*Generated by: `python experiments/11b_rl_comparison.py --config configs/experiments/paper_main_results.yaml`*

### Table V — Adversarial Robustness

| Attack | GOMDP Fp | Central Fp | P_breach (GOMDP) |
|--------|---------|------------|-----------------|
| No attack | 6.0% | 22.4% | 0.000 |
| Spoofing p=0.10 | 7.8% | 31.2% | 0.000 |
| Alert injection | 6.0% | **100%** | **0.000** |
| Byzantine f=2 | 6.2% | — | 0.097 |
| Byzantine f=3 | 8.9% | — | 0.581 |

*Generated by: `python experiments/09_adversarial_robustness.py --config configs/experiments/adversarial_robustness.yaml`*

### Table VI — Real-World VIIRS Validation

| Region | Ld (steps) | Fp (%) | Gov. Compliance | vs. IoT Baseline |
|--------|-----------|--------|-----------------|-----------------|
| California 2020 | 22.4 ± 3.2 | 8.3 ± 2.1 | 100% | −50% |
| Mediterranean 2021 | 24.1 ± 4.1 | 9.1 ± 2.5 | 100% | −47% |
| Australia 2019–20 | 21.8 ± 2.7 | 7.9 ± 1.9 | 100% | −52% |

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

## Citation

```bibtex
@article{akarma2025gomdp,
  title     = {Governance-Constrained Agentic {AI}: A Governance-Invariant {MDP} Framework
               with Blockchain-Enforced Human Oversight for Safety-Critical Wildfire Monitoring},
  author    = {Akarma, Ali and Syed, Toqeer Ali and Jan, Salman and
               Muneer, Hammad and Jilani, Abdul Khadar},
  journal   = {},
  year      = {2026},
  doi       = {}
}
```

---

## Acknowledgements

We thank NASA FIRMS for VIIRS data access, NIFC for historical fire perimeter data, NOAA for GOES-16 open data via AWS, and ECMWF/Copernicus for ERA5 reanalysis. Compute resources provided by the AI Center, Islamic University of Madinah.

## License

MIT — see [LICENSE](LICENSE).
