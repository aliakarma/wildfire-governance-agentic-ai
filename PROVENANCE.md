# Provenance Record for Paper Results

This document is the **single canonical map** from every table and figure in the
manuscript to (1) the CSV committed under [results/paper/](results/paper/),
(2) the script that regenerates it from live computation, (3) the dashboard view
that shows it, and (4) its **provenance class**.

The machine-readable version is
[results/paper/MANIFEST.yaml](results/paper/MANIFEST.yaml); the checker that
diffs a fresh run against the committed CSVs is
[scripts/check_reproducibility.py](scripts/check_reproducibility.py).

## Governing principle

**Every number in the paper is produced by running the code in this repository.**
No result is hand-set, back-filled, or calibrated toward a predetermined target.
Where the simulation cannot support a claim, the claim is removed from the paper
rather than the simulation adjusted to fit it.

This inverts the methodology used in earlier revisions of this work, which
treated the manuscript's numbers as frozen targets and tuned the simulator
toward them. That approach is abandoned: targets are an output, not an input.

## Environment & run details

- **Seeds:** 0–19 (deterministic RNG), 20 UAVs default
- **One simulation core:** [experiments/utils/runner.py](experiments/utils/runner.py)::`run_episode`, driven by the method taxonomy in [src/wildfire_governance/methods/registry.py](src/wildfire_governance/methods/registry.py). The dashboard benchmark path calls the same core.
- **Aggregator:** [scripts/generate_all_paper_results.py](scripts/generate_all_paper_results.py) (aggregates live per-seed CSVs; refuses to write paper files without `--per-seed`).

---

## Provenance classes

| Class | Meaning | Tolerance |
| :--- | :--- | :--- |
| **exact** | Closed-form or deterministic: Theorem 2 breach math, validator sweep, injection/replay blocking. Reproduces bit-for-bit from real computation. | 2% |
| **measured** | Produced by the stochastic simulation core over the stated seed set. Reruns vary within seed noise. | 5% |
| **training-derived** | From PPO training runs (learning curve, convergence episode, wall-clock). Recomputed by rerunning training, not by the evaluation harness. | — |
| **supplementary** | Not in the paper. Badged *Supplementary* in the dashboard. | n/a |

---

## Canonical artifact map

| Paper ref | Canonical ID | CSV | Script | Dashboard | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 1** (policy comparison) | `table1_rl_comparison` | [table1_rl_comparison.csv](results/paper/table1_rl_comparison.csv) | [11b_rl_comparison.py](experiments/11b_rl_comparison.py) | Benchmark | measured |
| **Table 5** (full-metric comparison) | `table_main_comparison` | [table1_rl_comparison_main.csv](results/paper/table1_rl_comparison_main.csv) | [01_main_comparison.py](experiments/01_main_comparison.py) | Benchmark | measured |
| **Table 2** (ablation) | `table2_ablation` | [table2_ablation.csv](results/paper/table2_ablation.csv) | [02_ablation_study.py](experiments/02_ablation_study.py) | Ablation | measured (injection column **exact**) |
| **Table 6** (adversarial) | `table3_adversarial` | [table3_adversarial.csv](results/paper/table3_adversarial.csv) | [09_adversarial_robustness.py](experiments/09_adversarial_robustness.py) | Adversarial Lab | measured (injection **exact**) |
| **VIIRS** (3 events) | `table4_realworld_viirs` | [table4_realworld_viirs.csv](results/paper/table4_realworld_viirs.csv) | [08_viirs_california.py](experiments/08_viirs_california.py) · [08b](experiments/08b_viirs_mediterranean.py) · [08c](experiments/08c_viirs_australia.py) | VIIRS | measured |
| **Validator compromise** | `table5_byzantine` | [theory](results/paper/table5_byzantine_theory.csv) · [empirical](results/paper/table5_byzantine_empirical.csv) | [13_byzantine_compromise.py](experiments/13_byzantine_compromise.py) | Adversarial | **exact** |
| **Validator-count sweep** | `table6_ksweep` | [table6_ksweep.csv](results/paper/table6_ksweep.csv) | [14_ksweep.py](experiments/14_ksweep.py) | Adversarial | **exact** |
| **HITL sensitivity** | `table7_hitl_sensitivity` | [table7_hitl_sensitivity.csv](results/paper/table7_hitl_sensitivity.csv) | [06b_hitl_sensitivity.py](experiments/06b_hitl_sensitivity.py) | HITL | measured (compliance **exact**) |
| **m-of-n multisig** | `table9_multisig` | [table9_multisig.csv](results/paper/table9_multisig.csv) | [16_multisig.py](experiments/16_multisig.py) | Adversarial | measured (injection **exact**) |
| **Figure 2** (F_p vs N) | `fig2_false_alerts` | [fig2_false_alerts.csv](results/paper/fig2_false_alerts.csv) | [04b_false_alert_scaling.py](experiments/04b_false_alert_scaling.py) | Scalability | measured |
| **Figure 4** (L_d vs N) | `fig4_latency` | [fig3_latency_data.csv](results/paper/fig3_latency_data.csv) | [03_scalability.py](experiments/03_scalability.py) | Scalability | measured |
| §5.2 training details | `ppo_training` | `results/runs/*/ppo_learning_curve.csv` | [11_ppo_training.py](experiments/11_ppo_training.py) | Learning | training-derived |

### Removed from the paper (previously reported, now withdrawn)

| Former artifact | Why removed |
| :--- | :--- |
| Fabric consensus microbenchmark | No Hyperledger Fabric network is deployed in this repository. The consensus model is a Python simulation; no measurement of a real Fabric deployment exists, so the claim was withdrawn. |
| SafeDreamer / CCPO comparison | These were calibrated stand-ins on the shared constrained-baseline path, not implementations of the published algorithms. The comparison was withdrawn; the methods are now cited as related work only. |
| CNN-architecture ablation | No CNN encoder was trained. Withdrawn. |
| PPO-GOMDP learning curve figure | Superseded by the live training run; see `results/runs/*/ppo_learning_curve.csv`. |

### Supplementary — not in the paper

| Canonical ID | CSV | Script | Note |
| :--- | :--- | :--- | :--- |
| `tradeoff_frontier` | [figure3_tradeoff_frontier.csv](results/paper/figure3_tradeoff_frontier.csv) | [05_tradeoff_frontier.py](experiments/05_tradeoff_frontier.py) | Pareto L_d/F_p at N=40 |
| `stress_tests` | [figure2_stress_tests.csv](results/paper/figure2_stress_tests.csv) | [10_stress_testing.py](experiments/10_stress_testing.py) | sensor/comms/burst stressors |

---

## What the cryptographic layer actually does

The governance mechanism is implemented, not simulated:

- **Ed25519 signing and verification** and **SHA3-256** hashing via the
  `cryptography` library ([crypto_utils.py](src/wildfire_governance/blockchain/crypto_utils.py)).
- **Registered-validator authorisation**: `verify_and_execute` rejects any
  certificate whose presenting public key is not in the registered validator
  set. Without this an adversary could generate its own keypair, validly sign
  its own forged transaction, and present its own key — this is Theorem 1's
  Case 1, and it is enforced in code.
- **Per-event nonce ledger**: a nonce may be committed once, so a previously
  approved certificate cannot be replayed.
- **Injection and replay resistance are measured**, not asserted:
  `probe_injection` mounts real attacks (`unsigned`, `wrong_key`, `replay`)
  through the same `verify_and_execute` path a legitimate alert takes. The
  regression tests in
  [tests/integration/test_gomdp_policy_agnostic.py](tests/integration/test_gomdp_policy_agnostic.py)
  assert that all three are blocked and that the legitimate path still succeeds.

**Consensus is simulated.** `PBFTConsensus` models BFT ordering delay in Python.
No Fabric network, no chaincode deployment, no physical UAVs. The paper states
this.

---

## How to reproduce & verify

```bash
# 0. Data + checkpoint (synthetic fallback needs no API keys)
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py

# 1. Regenerate every canonical CSV from live computation
bash experiments/run_all.sh --smoke            # fast sanity (2 seeds)
bash experiments/run_all.sh --skip_training    # full multi-seed (~2–4 h CPU)

# 2. Diff the fresh run against the committed CSVs
bash scripts/check_reproducibility.sh

# 3. Explore any artifact live
python dashboard/run_dashboard.py --port 8123

# 4. Build the anonymised submission archive
python scripts/build_anonymous_archive.py
```
