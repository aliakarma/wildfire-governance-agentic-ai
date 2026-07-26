# Paper result CSVs — canonical set, metric definitions, caveats

This directory holds the **frozen** result CSVs behind every paper table and
figure. "Frozen" means the paper numbers are fixed; the simulation is calibrated
to reproduce them (never the reverse). The authoritative map from each file to its
paper reference, script, dashboard view, and provenance class is
[MANIFEST.yaml](MANIFEST.yaml) (machine-readable) and
[../../PROVENANCE.md](../../PROVENANCE.md) (human-readable). The calibration
methodology and its documented deviations are in [CALIBRATION.md](CALIBRATION.md).

## Canonical files (this directory)

| File | Paper ref | Provenance |
| :--- | :--- | :--- |
| `table1_rl_comparison.csv` | Table 1 — policy comparison | calibration |
| `table1_rl_comparison_main.csv` | Table 5 — full-metric comparison | calibration |
| `table2_ablation.csv` | Table 2 — ablation | calibration (injection-blocking exact) |
| `table3_adversarial.csv` | Table 6 — adversarial robustness | calibration (injection-success exact) |
| `table4_realworld_viirs.csv` | VIIRS validation (3 events) | calibration |
| `table5_byzantine_theory.csv` · `table5_byzantine_empirical.csv` | validator/verifier compromise | **exact** (+ empirical breach exact) |
| `table6_ksweep.csv` | validator-count sweep | **exact** |
| `table7_hitl_sensitivity.csv` | HITL error-rate sensitivity | calibration (compliance exact) |
| `table8_recent_rl.csv` | recent Safe-RL comparators | calibration |
| `table9_multisig.csv` | m-of-n multisig | calibration (injection-blocking exact) |
| `table10_cnn_ablation.csv` | CNN-architecture ablation | **reference** |
| `fig2_false_alerts.csv` | Fig 2 — F_p vs N | calibration |
| `fig3_learning_curve.csv` | Fig 3 — learning curve | **reference** |
| `fig3_latency_data.csv` | Fig 4 — L_d vs N | calibration |
| `figure2_stress_tests.csv` · `figure3_tradeoff_frontier.csv` · `fig5_tradeoff_data.csv` | — | **supplementary (not in paper)** |

`.json` twins accompany the primary tables. Per-seed source data lives under
[per_seed/](per_seed/).

## Metric definitions

### False-alert rate (F_p)
- **Definition:** false-discovery rate = (# false alerts) / (# alerts broadcast) × 100%.
- **NOT** the classical signal-processing false-positive rate.
- **Computation:** [../../experiments/utils/runner.py](../../experiments/utils/runner.py).

### Detection latency (L_d)
- **Definition:** timesteps from **actual ignition** to first detection (confidence > 0.60), measured from the environment-provided ignition time (not assumed t=0).

### Governance compliance
- **Definition:** fraction of broadcast alerts carrying a valid governance certificate.
- **GOMDP configs:** 100% by Theorem 1 (environment-level enforcement) — this column is **exact**, not calibrated.
- **CMDP configs:** ~92–93% (HITL approval alone, no blockchain enforcement).

## CSV-specific notes

### Injection-blocking columns (`table2_ablation`, `table9_multisig`)
- **Deterministic per configuration**, not stochastic:
  - With blockchain / multisig enforcement: 100/100 blocked (a forged alert carries zero valid signatures).
  - Without enforcement (`− all authentication`, `PPO-CMDP no blockchain`): 0/100.
- This binary enforcement is an **exact** column and is checked at 2% tolerance.

### Byzantine / k-sweep (`table5_byzantine_*`, `table6_ksweep`)
- Closed-form Theorem-2 breach probabilities (`p_c = 0.10`, `f = ⌊(k−1)/3⌋`), reproduced **exactly** from real computation by [../../experiments/13_byzantine_compromise.py](../../experiments/13_byzantine_compromise.py) and [14_ksweep.py](../../experiments/14_ksweep.py).

### Reference tables (`table10_cnn_ablation`, `fig3_learning_curve`)
- **Training-derived**: parameter counts / convergence episodes / validation-L_d curve come from PPO training runs and are aggregated from committed per-seed data, not recomputed live. Reported informationally by the checker.

## Configuration caveats

- `governance.hitl.rejection_rate: 0.05` — the operator error rate `p_err` of Eq. (5); default for all experiments (matches `configs/base.yaml`).
- `blockchain.consensus.mean_delay_steps: 1.2` — a **simulated** consensus model (Python), not a real Hyperledger Fabric deployment. No Fabric network is deployed anywhere in this repository, and the paper makes no claim based on one. Latency bound: E[L_d] ≤ A/(v·N) + δ, δ ≈ 1.2 (BC) + 3.0 (human verify).
- Cryptography is **real**: Ed25519 signing/verification and SHA3-256 hashing via the `cryptography` library (`src/wildfire_governance/blockchain/crypto_utils.py`). Alert authorisation additionally enforces a registered-validator allowlist and a per-event nonce ledger, so injection and replay resistance are measured through the same code path a legitimate alert takes.

## See also
- [../../PROVENANCE.md](../../PROVENANCE.md) — canonical artifact map + provenance classes.
- [CALIBRATION.md](CALIBRATION.md) — calibration methodology + documented deviations.
- [../../experiments/README.md](../../experiments/README.md) — experiment script guide.
