# Environment parameterisation

How the simulation environment is configured, and why each parameter has the
value it does.

## How the committed results relate to this file

`results/paper/` holds the manuscript's values. They are emitted by
[`scripts/generate_all_paper_results.py`](../../scripts/generate_all_paper_results.py)
from the per-seed metric files in [per_seed/](per_seed/) plus live closed-form
computation, and re-checked cell by cell against `Paper/AAAI/Wildfire.tex` by
[`scripts/verify_paper_alignment.py`](../../scripts/verify_paper_alignment.py).

Where the simulation could not support a claim, the claim was **removed from the
manuscript** rather than the simulation adjusted to fit it; the four artifacts
withdrawn on those grounds are listed in the `withdrawn:` block of
[MANIFEST.yaml](MANIFEST.yaml) and in [PROVENANCE.md](../../PROVENANCE.md).

A fresh live run of the simulation core reproduces the manuscript's *qualitative*
claims — exact governance compliance, governed-low vs ungoverned-high $F_p$,
coordinated-fast vs static-slow $L_d$, and deterministic injection blocking — but
its absolute $L_d$ / $F_p$ magnitudes differ from the reported ones for the
modelling reasons documented below.
[`scripts/check_reproducibility.py`](../../scripts/check_reproducibility.py)
reports each such gap as `[KNOWN]` with its reason. The manuscript states that its
latencies are within-model comparisons, not field-calibrated values.

This file documents the environment parameters themselves, as modelling choices.

## One registry, one core

* **Registry** — `src/wildfire_governance/methods/registry.py` defines each
  method once: structural flags, the authorisation taxonomy, and per-method
  knobs.
* **Core** — `experiments/utils/runner.py::run_episode` is the single episode
  engine. The dashboard's method presets (`dashboard/backend/schema.py`) project
  from the same registry, so both sides share one taxonomy.

## Metrics and what determines them

| Metric | Determined by |
|---|---|
| **Compliance** | The enforcement mechanism, not a parameter. Crypto / signature / shield → 100% (Theorem 1). None → 0%. Soft/projection paths admit a residual violation rate. |
| **F_p** (false-alert rate) | Fraction of anomalies surviving to broadcast: `anomaly_rate` / `anomaly_intensity` (global) and `verify_strength` (per method). |
| **L_d** (detection latency) | Time from ignition to first true detection: `footprint_radius` (global) and coordination quality (sector tiling plus within-sector sweep). |

## Fidelity dependence

L_d and F_p do not transfer across grid size and must be evaluated at the
paper's fidelity (grid 100 × 3000 steps):

* L_d depends on coverage fraction (fleet footprint area / grid area). The same
  footprint gives very different latency at grid 60 versus grid 100, because
  coverage saturates.
* F_p is a ratio coupled to both fidelity and footprint: sparser coverage
  observes fewer anomalies, lowering F_p.

Grid 60 is used only for cheap directional checks. Every value reported in the
paper is produced at grid 100.

## Environment parameters

Locked in `registry.py` (`CALIBRATION_ENV`): `footprint_radius` 5,
`anomaly_rate` 2.0, `anomaly_intensity` (0.55, 0.99), `uav_speed` 1, plus
per-method `verify_strength` / `soft_leak` / `alert_threshold` /
`detection_probability`.

## Known modelling limitation

Over a 3000-step episode a spreading true fire re-alerts, and those repeat
alerts dominate the F_p denominator — so injected anomalies move F_p less than
the event rate would suggest. Addressing this requires structural event
de-duplication in the runner. It is a property of the model, and F_p should be
read as a within-model comparison across methods rather than an absolute
operational false-alarm rate. The paper states this.

## Check

```
python experiments/calibrate.py --fidelity full
```
