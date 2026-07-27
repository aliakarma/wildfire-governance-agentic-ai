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

**`results/paper/` holds the manuscript's values, and only those.** Every file is
emitted by [scripts/generate_all_paper_results.py](scripts/generate_all_paper_results.py)
from the per-seed metric files under `results/paper/per_seed/` plus live
closed-form computation; nothing is hand-edited.

**The two are checked against each other.**
[scripts/verify_paper_alignment.py](scripts/verify_paper_alignment.py) transcribes
every table and figure of `Paper/AAAI/Wildfire.tex` and re-checks all 354 committed
cells against it. Manuscript and results cannot silently diverge — if either moves
without the other, the check fails and names the cell.

Where the simulation could not support a claim, the claim was **removed from the
manuscript** rather than the simulation adjusted to fit it. Four artifacts were
withdrawn on those grounds; they are listed below and in the `withdrawn:` block of
`results/paper/MANIFEST.yaml`.

## Environment & run details

- **Seeds:** 0–19 (deterministic RNG), 20 UAVs default
- **One simulation core:** [experiments/utils/runner.py](experiments/utils/runner.py)::`run_episode`, driven by the method taxonomy in [src/wildfire_governance/methods/registry.py](src/wildfire_governance/methods/registry.py). The dashboard benchmark path calls the same core.
- **Aggregator:** [scripts/generate_all_paper_results.py](scripts/generate_all_paper_results.py) (aggregates the per-seed CSVs; refuses to write the canonical files without an explicit `--per-seed` source).
- **Alignment gate:** [scripts/verify_paper_alignment.py](scripts/verify_paper_alignment.py) (results vs manuscript, cell by cell).
- **Seed bundles:** [scripts/rebuild_per_seed_bundles.py](scripts/rebuild_per_seed_bundles.py) (regenerates `per_seed/seed_<n>.json` from the per-seed CSVs).

---

## Provenance classes

| Class | Meaning | Tolerance |
| :--- | :--- | :--- |
| **exact** | Closed-form or deterministic: Theorem 2 breach math, validator sweep, injection/replay blocking, the statistical tests. Reproduces bit-for-bit from real computation. | 2% |
| **measured** | Aggregated over seeds 0–19 by the simulation core. Reruns vary within seed noise. | 5% |
| **specification** | A configuration table, not a measurement (Table 4 parameters). | n/a |
| **training-derived** | From PPO training runs (the validation learning curve). Recomputed by rerunning training, not by the evaluation aggregator. | — |
| **supplementary** | Not in the manuscript. Badged *Supplementary* in the dashboard. | n/a |

---

## Canonical artifact map

| Manuscript ref | Canonical ID | CSV | Script | Dashboard | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 1** (policy comparison) | `table1_rl_comparison` | [table1_rl_comparison.csv](results/paper/table1_rl_comparison.csv) | [11b_rl_comparison.py](experiments/11b_rl_comparison.py) | Benchmark | measured |
| **Table 2** (ablation) | `table2_ablation` | [table2_ablation.csv](results/paper/table2_ablation.csv) | [02_ablation_study.py](experiments/02_ablation_study.py) | Ablation | measured (injection column **exact**) |
| **Table 4** (configuration parameters) | `table_config_parameters` | [table_config_parameters.csv](results/paper/table_config_parameters.csv) | — | Experiments | specification |
| **Table 5** (full-metric comparison) | `table_main_comparison` | [table1_rl_comparison_main.csv](results/paper/table1_rl_comparison_main.csv) | [01_main_comparison.py](experiments/01_main_comparison.py) | Statistics | measured |
| **Table 6** (adversarial) | `table3_adversarial` | [table3_adversarial.csv](results/paper/table3_adversarial.csv) | [09_adversarial_robustness.py](experiments/09_adversarial_robustness.py) | Adversarial | measured (injection **exact**) |
| **Table 7** (VIIRS, 3 events) | `table4_realworld_viirs` | [table4_realworld_viirs.csv](results/paper/table4_realworld_viirs.csv) | [08_viirs_california.py](experiments/08_viirs_california.py) · [08b](experiments/08b_viirs_mediterranean.py) · [08c](experiments/08c_viirs_australia.py) | VIIRS | measured |
| **Table 8** (validator compromise) | `table5_byzantine` | [theory](results/paper/table5_byzantine_theory.csv) · [empirical](results/paper/table5_byzantine_empirical.csv) | [13_byzantine_compromise.py](experiments/13_byzantine_compromise.py) | Adversarial | **exact** |
| **Table 9** (validator-count sweep) | `table6_ksweep` | [table6_ksweep.csv](results/paper/table6_ksweep.csv) | [14_ksweep.py](experiments/14_ksweep.py) | Adversarial | **exact** |
| **Table 10** (HITL sensitivity) | `table7_hitl_sensitivity` | [table7_hitl_sensitivity.csv](results/paper/table7_hitl_sensitivity.csv) | [06b_hitl_sensitivity.py](experiments/06b_hitl_sensitivity.py) | HITL | measured (compliance **exact**) |
| **Table 11** (m-of-n multisig) | `table9_multisig` | [table9_multisig.csv](results/paper/table9_multisig.csv) | [16_multisig.py](experiments/16_multisig.py) | Adversarial | measured (injection **exact**) |
| **Sec. 6.2** (statistical testing) | `statistical_tests` | [statistical_tests.csv](results/paper/statistical_tests.csv) | [generate_all_paper_results.py](scripts/generate_all_paper_results.py) | Statistics | **exact** |
| **Figure 2** (F_p vs N) | `fig2_false_alerts` | [fig2_false_alerts.csv](results/paper/fig2_false_alerts.csv) | [04b_false_alert_scaling.py](experiments/04b_false_alert_scaling.py) | Scalability | measured |
| **Sec. 5.2** (training convergence) | `fig3_learning_curve` | [fig3_learning_curve.csv](results/paper/fig3_learning_curve.csv) | [11_ppo_training.py](experiments/11_ppo_training.py) | Learning | training-derived |
| **Figure 3** (L_d vs N) | `fig3_latency` | [fig3_latency_data.csv](results/paper/fig3_latency_data.csv) | [03_scalability.py](experiments/03_scalability.py) | Scalability | measured |

### Removed from the manuscript (previously reported, now withdrawn)

| Former artifact | Why removed |
| :--- | :--- |
| Fabric consensus microbenchmark | No Hyperledger Fabric network is deployed in this repository. The consensus model is a Python simulation; no measurement of a real Fabric deployment exists, so the claim was withdrawn. The Limitations section now states that consensus is simulated. |
| SafeDreamer / CCPO comparison | These were calibrated stand-ins on the shared constrained-baseline path, not implementations of the published algorithms. The comparison was withdrawn; the methods are cited as related work only, and the manuscript states that a controlled comparison is future work. |
| CNN-architecture ablation | No CNN encoder was trained. Withdrawn; the manuscript now states that a convolutional encoder was not evaluated. |
| Proposition-1 latency bound | The manuscript contains no Proposition 1. The column and the dashboard series that plotted it were removed. |

### Supplementary — not in the manuscript

| Canonical ID | CSV | Script | Note |
| :--- | :--- | :--- | :--- |
| `tradeoff_frontier` | [figure3_tradeoff_frontier.csv](results/paper/figure3_tradeoff_frontier.csv) · [fig5_tradeoff_data.csv](results/paper/fig5_tradeoff_data.csv) | [05_tradeoff_frontier.py](experiments/05_tradeoff_frontier.py) | Pareto L_d/F_p at N=40; consistent with the N=40 column of Figures 2–3 |

Stress testing ([10_stress_testing.py](experiments/10_stress_testing.py)) writes to
`results/runs/` only. It is not a manuscript figure and has no committed paper CSV.

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

# 2. Verify the committed results still match the manuscript (the submission gate)
python scripts/verify_paper_alignment.py

# 3. Diff a fresh run against the committed CSVs
bash scripts/check_reproducibility.sh

# 4. Explore any artifact live
python dashboard/run_dashboard.py --port 8123

# 5. Build the anonymised submission archive
python scripts/build_anonymous_archive.py
```
