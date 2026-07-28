# Wildfire Governance Agentic AI

**Governance-Invariant MDPs: A Framework and Formal Safety Case for Agentic Wildfire Monitoring**

*Anonymous Submission — AAAI 2027 Reviewer Package*

---

## Core Contribution

This repository implements the **Governance-Invariant MDP (GOMDP)** framework, which provides a qualitatively stronger form of constraint satisfaction than existing CMDP approaches:

| Property | CMDP / CPO | **GOMDP (Ours)** |
|----------|-----------|-----------------|
| Safety guarantee | In-expectation | **Per-trajectory, prob. >= 1 - T*eps_sig** |
| Violation rate | 5–15% | **0%** |
| Non-repudiation | None | **Cryptographic** |
| Adversarial tolerance | None | **Byzantine-fault-tolerant** |

**Theorem 1 (Policy-Agnostic Safety):** Any policy operating in a GOMDP satisfies the governance predicate with probability negligibly close to one, regardless of optimality gap - conditional on Ed25519 signature security, a Byzantine validator threshold k >= 3f+1, and faithful chaincode implementation. The guarantee secures *authorization integrity*, not the correctness of the human decision.

---

## Quick Navigation

- [Installation](installation.md) — Set up the environment
- [Quick Start](quickstart.md) — Run in 5 minutes
- [Architecture](architecture.md) — System design
- [Datasets](datasets.md) — Real-world VIIRS data setup
- [Reproducibility](reproducibility.md) — Reproduce all paper results
- [Blockchain Setup](blockchain_setup.md) — Hyperledger Fabric details

---

## Key Results

| Method | Ld (steps) | Fp (%) | Governance |
|--------|-----------|--------|-----------|
| **PPO-GOMDP** | **15.1 ± 1.1** | **6.0%** | **100%** |
| Greedy-GOMDP | 18.3 ± 1.4 | 6.1% | 100% |
| PPO-CMDP | 14.8 ± 1.0 | 8.3% | 92.8% ← violates |
| Adaptive AI | 16.2 ± 1.2 | 22.4% | 0% |
| IoT threshold pipeline ([6]) | ~45 | — | — |

The IoT threshold pipeline is qualitative context only: it differs in hardware, scenarios, modality, and latency definition, so no improvement over it is claimed. Every value above is verified against the manuscript by `scripts/verify_paper_alignment.py`.
