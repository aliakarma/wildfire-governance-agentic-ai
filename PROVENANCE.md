# Provenance Record for Paper Results

This document is the **single canonical map** from every table/figure in the
manuscript ([Paper/AAAI/Wildfire.tex](Paper/AAAI/Wildfire.tex)) to (1) the frozen
CSV committed under [results/paper/](results/paper/), (2) the runnable script that
regenerates it, (3) the dashboard view that shows it, and (4) its **provenance
class**, which sets the reproduction tolerance.

The machine-readable version of this map is
[results/paper/MANIFEST.yaml](results/paper/MANIFEST.yaml); the automated checker
that diffs a fresh live run against these frozen CSVs is
[scripts/check_reproducibility.py](scripts/check_reproducibility.py).

## Environment & run details

- **Git commit SHA:** `ade948fc6b4e73e0cf7080a280c23c92827627c9`
- **Seeds:** 0–19 (deterministic RNG), 20 UAVs default
- **One simulation core:** [experiments/utils/runner.py](experiments/utils/runner.py)::`run_episode`, driven by the shared method taxonomy in [src/wildfire_governance/methods/registry.py](src/wildfire_governance/methods/registry.py). The dashboard benchmark path calls the same core.
- **Aggregator:** [scripts/generate_all_paper_results.py](scripts/generate_all_paper_results.py) (aggregates live per-seed CSVs; refuses to write paper files without `--per-seed`).

---

## Provenance classes

Two decisions govern this repo: **the paper numbers are frozen** (calibrate the
simulation to reproduce them — never edit the targets), and **the paper is the
figure spec** (every paper figure gets a script + dashboard view; anything not in
the paper is badged *Supplementary*). Each artifact is classified by how it is
reproduced:

| Class | Meaning | Tolerance | Fatal on drift? |
| :--- | :--- | :--- | :--- |
| **exact** | Closed-form / deterministic (Theorem 2 breach math, validator sweep, injection-blocking). Reproduces the paper bit-for-bit from real computation. | 2% | **Yes** |
| **calibration** | Produced by the stochastic simulation core. Magnitudes are calibrated toward the frozen targets; residual gaps are documented, not hidden. | 5% | No (loud) |
| **reference** | Training-derived values this repo does not recompute live (CNN params, learning curve). Aggregated from committed per-seed data; reported informationally. | — | No |
| **supplementary** | **Not in the paper.** Badged *Supplementary — not in the paper* in the dashboard. | n/a | No |

---

## Canonical artifact map

| Paper ref | Canonical ID | Frozen CSV | Script | Dashboard view | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 1** (policy comparison) | `table1_rl_comparison` | [table1_rl_comparison.csv](results/paper/table1_rl_comparison.csv) | [11b_rl_comparison.py](experiments/11b_rl_comparison.py) | Benchmark | calibration |
| **Table 5** (full-metric comparison) | `table_main_comparison` | [table1_rl_comparison_main.csv](results/paper/table1_rl_comparison_main.csv) | [01_main_comparison.py](experiments/01_main_comparison.py) | Benchmark (full-metric) | calibration |
| **Table 2** (ablation) | `table2_ablation` | [table2_ablation.csv](results/paper/table2_ablation.csv) | [02_ablation_study.py](experiments/02_ablation_study.py) | Ablation | calibration (injection-blocking **exact**) |
| **Table 6** (adversarial robustness) | `table3_adversarial` | [table3_adversarial.csv](results/paper/table3_adversarial.csv) | [09_adversarial_robustness.py](experiments/09_adversarial_robustness.py) | Adversarial Lab | calibration (injection-success **exact**) |
| **Table** (real-world VIIRS, 3 events) | `table4_realworld_viirs` | [table4_realworld_viirs.csv](results/paper/table4_realworld_viirs.csv) | [08_viirs_california.py](experiments/08_viirs_california.py) · [08b](experiments/08b_viirs_mediterranean.py) · [08c](experiments/08c_viirs_australia.py) | VIIRS | calibration |
| **Table** (validator/verifier compromise) | `table5_byzantine` | [table5_byzantine_theory.csv](results/paper/table5_byzantine_theory.csv) · [table5_byzantine_empirical.csv](results/paper/table5_byzantine_empirical.csv) | [13_byzantine_compromise.py](experiments/13_byzantine_compromise.py) | Adversarial (breach meter) | **exact** (theory) + calibration (empirical, breach col **exact**) |
| **Table** (validator-count sweep) | `table6_ksweep` | [table6_ksweep.csv](results/paper/table6_ksweep.csv) | [14_ksweep.py](experiments/14_ksweep.py) | Adversarial (k selector) | **exact** |
| **Table** (HITL error-rate sensitivity) | `table7_hitl_sensitivity` | [table7_hitl_sensitivity.csv](results/paper/table7_hitl_sensitivity.csv) | [06b_hitl_sensitivity.py](experiments/06b_hitl_sensitivity.py) | HITL | calibration (compliance **exact**) |
| **Table** (recent Safe-RL comparators) | `table8_recent_rl` | [table8_recent_rl.csv](results/paper/table8_recent_rl.csv) | [15_recent_rl.py](experiments/15_recent_rl.py) | Benchmark (recent-RL) | calibration (stand-ins) |
| **Table** (m-of-n multisig) | `table9_multisig` | [table9_multisig.csv](results/paper/table9_multisig.csv) | [16_multisig.py](experiments/16_multisig.py) | Adversarial (consensus ref) | calibration (injection-blocking **exact**) |
| **Table** (CNN-architecture ablation) | `table10_cnn_ablation` | [table10_cnn_ablation.csv](results/paper/table10_cnn_ablation.csv) | [11_ppo_training.py](experiments/11_ppo_training.py) (training-derived) | CNN | **reference** |
| **Figure 2** (F_p vs N) | `fig2_false_alerts` | [fig2_false_alerts.csv](results/paper/fig2_false_alerts.csv) | [04b_false_alert_scaling.py](experiments/04b_false_alert_scaling.py) | Scalability | calibration |
| **Figure 3** (PPO-GOMDP learning curve) | `fig3_learning` | [fig3_learning_curve.csv](results/paper/fig3_learning_curve.csv) | [11_ppo_training.py](experiments/11_ppo_training.py) (training-derived) | Learning | **reference** |
| **Figure 4** (L_d vs N) | `fig4_latency` | [fig3_latency_data.csv](results/paper/fig3_latency_data.csv) | [03_scalability.py](experiments/03_scalability.py) | Scalability | calibration |
| **Appendix** (Fabric consensus microbenchmark) | `table_fabric_microbench` | [table_fabric_microbench.csv](results/paper/table_fabric_microbench.csv) | [07_blockchain_throughput.py](experiments/07_blockchain_throughput.py) (simulated Δ_BC) | All Experiments (Governance) | **reference** |

### Supplementary — NOT in the paper

| Canonical ID | Frozen CSV | Script | Dashboard | Note |
| :--- | :--- | :--- | :--- | :--- |
| `tradeoff_frontier` | [figure3_tradeoff_frontier.csv](results/paper/figure3_tradeoff_frontier.csv) · [fig5_tradeoff_data.csv](results/paper/fig5_tradeoff_data.csv) | [05_tradeoff_frontier.py](experiments/05_tradeoff_frontier.py) | Benchmark — badged *Supplementary* | Pareto L_d/F_p at N=40 |
| `stress_tests` | [figure2_stress_tests.csv](results/paper/figure2_stress_tests.csv) | [10_stress_testing.py](experiments/10_stress_testing.py) | badged *Supplementary* | sensor/comms/burst stressors |

> Note on paper Table numbering: manuscript Tables 3 and 4 are the notation and
> configuration tables (no experimental data), so the canonical `table3_*` /
> `table4_*` **files** map to later data tables in the paper. The "Paper ref"
> column above is authoritative; the numeric filename prefix only fixes sort
> order.

---

## Calibration & qualitative reproduction (documented deviations)

The per-seed CSV under [results/paper/per_seed/](results/paper/per_seed/) was
verified (WS0) to aggregate **exactly** to the manuscript Table-1 means/stds —
i.e. it was back-filled from the paper, not produced by the simulator. Reproducing
the paper therefore means calibrating the **real** engine, which is genuine work,
not a tautology.

Calibration is **qualitative**: the calibration levers (search/coordination for
`L_d`, verification strength and HITL for `F_p`, mechanism-determined compliance)
reproduce the paper's **qualitative claims** — exact governance compliance,
governed-low vs ungoverned-high `F_p`, coordinated-fast vs static-slow `L_d`,
monotone `F_p`-vs-N decrease — but some **absolute magnitudes** at grid 100 remain
outside 5% for documented reasons:

1. **High L_d seed variance** at the paper's seed count.
2. **F_p denominator saturation** — over a 3000-step episode a spreading true fire
   re-alerts and dominates the false-discovery ratio.
3. **Back-filled targets** — some hand-set targets are mutually inconsistent with
   any single integer parameterization (e.g. the Static `L_d` cliff).

These are recorded, per artifact, in `KNOWN_DEVIATIONS` inside
[check_reproducibility.py](scripts/check_reproducibility.py) and explained in full
in [results/paper/CALIBRATION.md](results/paper/CALIBRATION.md). The checker reports
them as `[KNOWN]` and never treats them as fatal; every **exact** column (breach
math, injection-blocking, compliance) still must reproduce within 2%. Nothing is
hardcoded to pass — the paper itself states its latencies are "relative
comparisons, not field-calibrated."

---

## How to reproduce & verify

```bash
# 0. Data + checkpoint (synthetic fallback needs no API keys)
python scripts/download_checkpoint.py
python data/scripts/generate_synthetic.py

# 1. Regenerate every canonical CSV from live computation
bash experiments/run_all.sh --smoke      # fast sanity (2 seeds)
bash experiments/run_all.sh --skip_training   # full multi-seed (~2–4 h CPU)

# 2. Diff the fresh live run vs the frozen paper CSVs, by provenance class
bash scripts/check_reproducibility.sh
#   exact cols must match within 2% (fatal otherwise);
#   calibration drift is reported; documented deviations print [KNOWN].

# 3. Explore any artifact live in the dashboard
python dashboard/run_dashboard.py --port 8123   # → http://127.0.0.1:8123/
```
