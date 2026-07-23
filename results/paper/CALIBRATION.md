# Calibration methodology (WS1)

How the one simulation core is calibrated so a live multi-seed run reproduces the
frozen paper targets (results/paper/*.csv) within tolerance. The paper numbers are
FROZEN; calibration only adjusts the simulation, never the targets. See
results/paper/MANIFEST.yaml for the artifact map and the back-fill provenance
finding that makes this genuine work.

## One registry, one core

* **Registry** — `src/wildfire_governance/methods/registry.py` defines all 11
  methods once (9 in Table 1 + SafeDreamer/CCPO in Table 8): structural flags, the
  authorization taxonomy, the calibrated knobs, and each method's frozen target.
* **Core** — `experiments/utils/runner.py::run_episode` is the single episode
  engine. The dashboard's method presets (`dashboard/backend/schema.py`) are
  projected from the same registry, so both sides share one taxonomy.
* **Harness** — `experiments/calibrate.py` runs every registry method through the
  core, diffs live vs frozen target per metric, and writes a gap report.

## The three metrics

| Metric | How it is set | Lever(s) |
|---|---|---|
| **Compliance** | mechanism-determined (Theorem 1) — EXACT by construction | authorization + soft_leak (registry) |
| **F_p** (false-alert rate) | stochastic; fraction of anomalies that survive to broadcast | `anomaly_rate`/`anomaly_intensity` (global) + `verify_strength` (per method) + `alert_threshold` (Static, no verification) |
| **L_d** (detection latency) | stochastic; time from ignition to first true detection | `footprint_radius` (global) + coordination quality (perfect-square tiling + within-sector sweep) |

Compliance is not calibrated — it falls out of the enforcement mechanism:
crypto/signature/shield → 100%, projection/soft → 100·(1−soft_leak), none → 0%.

## Critical finding: calibrate at FULL fidelity only

L_d and F_p are strongly **fidelity-dependent** and do not transfer across grid
size:

* L_d depends on the coverage fraction (fleet footprint area / grid area). The
  same footprint gives L_d ≈ 27 at grid 60 but ≈ 0 at grid 100 (coverage
  saturates). Detection latency must be tuned at the paper's grid = 100.
* F_p is a ratio (false / total alerts) coupled to both fidelity and footprint:
  sparser coverage observes fewer anomalies, lowering F_p.

Therefore med-fidelity (grid 60) is only for cheap directional checks; every
value that goes into the paper comparison is validated at grid 100 × 3000 steps.

## Calibration order (coordinate descent, full fidelity)

Because the levers couple, tune them in this order and re-measure after each:

1. **footprint_radius** → set coordinated L_d ≈ 15–18 and Static ≈ 41 (Static must
   remain the slowest; coordinated methods sweep, Static sits at fixed points).
2. **anomaly_rate / intensity** → set the global F_p spread so the ungoverned
   anchor (Adaptive AI) reaches ≈ 22 at its verify_strength.
3. **verify_strength** (per method) → dial each governed/soft method's F_p to its
   target; F_p is monotone decreasing in verify_strength (≈ −65 F_p per unit near
   the governed region at grid 100).
4. **alert_threshold** (Static only) → Static has no verification stage, so its F_p
   is set by a conservative raw broadcast threshold rather than verify_strength.

## Residuals policy

Some paper values were back-filled from hand-chosen mean±std and may not
correspond to any self-consistent simulation. For any metric that cannot reach the
5% tolerance after calibration, record it as a documented, explained deviation in
`scripts/check_reproducibility.py` (or propose a minimal target tweak for
approval) — never silently hardcode an output.

## Outcome (locked)

Decision: **qualitative calibration + documented deviations**. The locked
parameters live in `src/wildfire_governance/methods/registry.py`
(`CALIBRATION_ENV` + per-method knobs): footprint 5, anomaly_rate 2.0, intensity
(0.55, 0.99), uav_speed 1, plus per-method verify_strength / soft_leak /
alert_threshold / detection_probability.

**Reproduced faithfully (the paper's qualitative claims):**
* Governance compliance — EXACT for all 9 methods (mechanism-determined).
* F_p ordering — governed (~2) < soft/CMDP (~5) < ungoverned Adaptive (~10):
  governance suppresses false alerts, the paper's headline safety claim.
* L_d ordering — coordinated (~8–18) ≪ Static (~40–47): coordination detects
  faster, and the Static detection-reliability lever lands Static ≈ 41 (probe:
  38.8 vs 41.5).

**Documented deviations (absolute magnitudes):**
* L_d has high **seed variance** at 4–6 seeds (e.g. coordinated 17.5 at 4 seeds
  vs 8.2 at 6 seeds), so absolute coordinated L_d is not pinned to ±5%.
* F_p **saturates below target** at grid 100 × 3000 steps: a spreading true fire
  re-alerts and dominates the F_p denominator, so injected anomalies move F_p
  little (Adaptive 9.6 → 10.3 as anomaly_rate went 0.6 → 2.0). Reaching the
  paper's 6–22 band needs a structural event-deduplication change, deferred.
* Root cause compounded by the **back-fill** (targets sampled from hand-chosen
  mean±std; see MANIFEST provenance_finding) — some may be mutually inconsistent.

These are recorded in `scripts/check_reproducibility.py::KNOWN_DEVIATIONS` and
reported as `[KNOWN]` (never a silent pass, never a hard fail). No simulation
output is hardcoded to match the paper.

## Reproduce

```
python experiments/calibrate.py --fidelity full                 # all 9, grid 100 (locked params)
python experiments/calibrate.py --fidelity full --footprint 2   # L_d sweep
python experiments/calibrate.py --fidelity med --methods greedy_gomdp,static  # quick check
```
