# Environment parameterisation

How the simulation environment is configured, and why each parameter has the
value it does.

## Methodology change (important)

Earlier revisions of this work treated the manuscript's numbers as **frozen
targets** and tuned the simulator until it reproduced them, recording any
residual gap as a "documented deviation". Per-seed data for the main comparison
table was additionally **back-filled** — sampled from the manuscript's stated
mean ± std rather than produced by the simulator.

**That methodology is abandoned.** Targets are an output of the code, not an
input to it. Every number in the current manuscript comes from a live run; where
the simulation could not support a claim, the claim was removed from the paper.
See [PROVENANCE.md](../../PROVENANCE.md) for the artifact map and the list of
claims withdrawn on those grounds.

This file now documents only the environment parameters themselves, as modelling
choices.

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
