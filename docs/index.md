# Wildfire Governance Agentic AI

**Governance-Constrained Agentic AI: A Governance-Invariant MDP Framework with Blockchain-Enforced Human Oversight for Safety-Critical Wildfire Monitoring**

*Ali Akarma · Toqeer Ali Syed · Salman Jan · Hammad Muneer · Abdul Khadar Jilani*

---

## Core Contribution

This repository implements the **Governance-Invariant MDP (GOMDP)** framework, which provides a qualitatively stronger form of constraint satisfaction than existing CMDP approaches:

| Property | CMDP / CPO | **GOMDP (Ours)** |
|----------|-----------|-----------------|
| Safety guarantee | In-expectation | **Per-trajectory, prob. 1** |
| Violation rate | 5–15% | **0%** |
| Non-repudiation | None | **Cryptographic** |
| Adversarial tolerance | None | **Byzantine-fault-tolerant** |

**Theorem 1 (Policy-Agnostic Safety):** Any policy operating in a GOMDP satisfies the governance predicate with probability one, regardless of optimality gap.

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
| PPO-CMDP | 14.8 ± 1.0 | 8.3% | 0.0% ← violates |
| Adaptive AI | 16.2 ± 1.2 | 22.4% | 0% |
| IoT baseline ([6]) | ~45 | — | 0% |
