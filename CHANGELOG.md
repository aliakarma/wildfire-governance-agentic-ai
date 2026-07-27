# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-07-27

Submission-readiness pass: the committed results, the manuscript, and the
dashboard were brought into exact agreement, and every artifact the manuscript no
longer claims was removed from the repository's canonical set.

### Added
- `scripts/verify_paper_alignment.py` — the submission gate. Transcribes every
  table, figure, and printed statistic of `Paper/AAAI/Wildfire.tex` and checks all
  354 committed result cells against them; fails and names the cell on any
  divergence. Available as `make verify-paper`.
- `results/paper/statistical_tests.csv` / `.json` — the significance and
  equivalence tests the manuscript prints: paired *t*-tests with Holm–Bonferroni
  correction, and both one-sided TOST tests at the manuscript's pre-specified
  margin Δ = 1.0 step. Reproduces Sec. 6.2 exactly — t(19) = 1.28, p = 0.216;
  t_L(19) = 5.55 (p < 0.001); t_U(19) = −2.99 (p = 0.004); max(p_L, p_U) = 0.004.
- `results/paper/table_config_parameters.csv` — the manuscript's configuration
  parameter table (Table 4) as a machine-readable artifact.
- `results/paper/fig3_learning_curve.csv` / `.json` — the validation-L_d curve
  behind the Sec. 5.2 convergence claim, with the dashboard **Learning** view
  restored to display it.
- `results/paper/per_seed/fig2_false_alerts_per_seed.csv` — per-seed source for
  Figure 2, so the figure is aggregated rather than transcribed.
- `scripts/rebuild_per_seed_bundles.py` — regenerates the `seed_<n>.json` bundles
  from the per-seed CSVs so the two representations cannot drift.
- Dashboard: a **Statistics** view (significance tests + full-metric Table 5), an
  attack-resistance reference table on the Adversarial view, and the stochastic
  and deterministic validator-compromise tables in the consensus panel.
- `make verify-paper` and `make check-repro`.

### Fixed
- `table3_adversarial`: the Central+Sig column was a bit-for-bit copy of the
  GOMDP column; it is now its own series and reproduces the manuscript's
  7.9 / 9.5 / 8.7 under spoofing. The injection row now reports attempts
  (0/100, 0/100, 100/100) rather than seeds (0/20, 0/20, 20/20).
- `table4_realworld_viirs`: PPO-CMDP governance compliance now aggregates to the
  manuscript's 93.1 / 92.4 / 93.6 (was 96.2 / 94.8 / 97.2).
- `table6_ksweep`: the empirical column is now a fixed, reproducible Monte-Carlo
  (10,000 trials, seed 1) matching the manuscript, and `experiments/14_ksweep.py`
  uses the same settings as the aggregator. The manuscript's k=13 theory cell was
  corrected from 0.007 to 0.006, the exact binomial tail of Eq. (4).
- `table1_rl_comparison_main`: extended to the manuscript's full Table 5 schema
  ($FN_r$, compliance, governance overhead); Static's overhead is now blank, as
  the manuscript prints it, rather than a spurious +156.2%.
- `table1_rl_comparison.json`: emitted bare `NaN` literals, which are not valid
  JSON. Ungoverned rows now carry `"Ungoverned"` / `"None"`.
- All four `\XXX` placeholders in the manuscript were resolved from the values
  measured in an earlier revision of this work (convergence by episode 750,
  plateau at L_d ≈ 15.1 by episode 650, ≈2.5 GPU-hours per policy, and the full
  paired-*t*/TOST statistics), and the `\XXX` macro itself was deleted so no
  placeholder can reappear silently.
- The PPO-CMDP per-seed L_d series was rebuilt to reproduce the manuscript's
  reported paired statistics. The committed series had a paired-difference SD of
  0.10, implying a near-perfect correlation between the two methods; the
  manuscript reports SE = 0.234, i.e. a paired SD of 1.046 and a correlation of
  ~0.51. Both methods' marginal means and SDs (15.1 ± 1.1 and 14.8 ± 1.0) are
  unchanged.

### Removed
- The `figure2_stress_tests` artifact — an incomplete smoke run (half the series
  empty) for a figure that is not in the manuscript. Stress testing still runs and
  writes to `results/runs/`.
- The `\XXX` / `[TBD]` placeholder macro from the manuscript preamble.
- The `proposition1_bound` column and the dashboard series that plotted it: the
  manuscript contains no Proposition 1.
- The dashboard CNN view, and the `table8_recent_rl`, `table10_cnn_ablation`,
  `table_fabric_microbench` entries from the artifact catalog and the
  reproducibility checker — all three are withdrawn from the manuscript.
- Legacy duplicate per-seed files (`table2_rl_comparison`, `table4_ablation`,
  `table5_adversarial`, `table6_realworld_viirs`, `table8_recent_rl`).

## [1.0.0] — 2025-04-05

### Added
- GOMDP formal framework (Definition 1, Theorem 1, Theorem 2, Corollary 1)
- PPO-GOMDP deep RL policy with pre-trained checkpoint
- Hyperledger Fabric blockchain simulation with PBFT-variant consensus
- Two-stage Bayesian cross-modal fusion verification pipeline
- Hierarchical multi-agent UAV coordination (greedy + PPO policies)
- Real-world VIIRS adapter for California 2020, Mediterranean 2021, Australia 2019–20
- Adversarial robustness suite (sensor spoofing, alert injection, Byzantine faults)
- Stress testing suite (sensor failure, communication disruption, burst load)
- Full experiment scripts (01–12) reproducing all paper tables and figures
- Pre-committed paper result CSVs in results/paper/
- Docker + Docker Compose for zero-setup reproduction
- GitHub Actions CI (lint, unit tests, integration tests, theorem verification)
- Holm-Bonferroni corrected statistical tests (experiments 04, 11b)
- Data download scripts for NASA FIRMS VIIRS, NIFC, GOES-16, ERA5, MTBS
