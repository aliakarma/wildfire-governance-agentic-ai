# Changelog

## [1.1.0] — 2026-04-06  (Reviewer Repair Release)

### Critical Fixes (7 rejection-level issues resolved)

**Issue 1 — PPO policy gradient** (`src/wildfire_governance/rl/ppo_agent.py`)
- Added `RolloutBuffer` dataclass storing `(obs, actions_per_uav, log_probs_old, values, rewards, dones)` per step
- `select_actions()` now returns `(allocation, log_probs, value)` — actual log-probs of taken actions
- `update_from_buffer()` implements correct clipped surrogate: `L = min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)`
- No more `torch.randint` in the policy loss path

**Issue 2 — GOMDP bypass path** (`src/wildfire_governance/rl/gomdp_env.py`)
- `self._gomdp.step_alert_action()` now called for EVERY alert attempt
- Removed the `else` branch that broadcast alerts without a certificate on HITL rejection
- Compliance rate derived from `self._gomdp.get_compliance_rate()` (actually invoked counter)

**Issue 3 — Hardcoded PPO-CMDP compliance** (`experiments/11b_rl_comparison.py`)
- Deleted `ppo_cmdp["governance_compliance_pct"] = max(0.0, 92.8 if not smoke else ...)` 
- PPO-CMDP compliance is now measured empirically via `GovernanceInvariantChecker`

**Issue 4 — Breach probability formula** (`src/wildfire_governance/gomdp/breach_probability.py`, `experiments/09_adversarial_robustness.py`)
- Documented canonical values: k=7, f=2, p_c=0.3 → P_breach = 0.353 (not 0.097)
- Corrected `09_adversarial_robustness.py` to use `cfg.blockchain.max_byzantine` (not `max(0, 7//3-1)=1`)
- Updated `results/paper/table5_adversarial.csv` with correct breach probabilities
- `paper_numerical_check()` now asserts against computed value, not a different hardcoded expected

**Issue 5 — Silent checkpoint fallback** (`src/wildfire_governance/rl/evaluator.py`)
- `FileNotFoundError` is now raised (not silently swallowed) when checkpoint is missing
- Any reproduction run without `ppo_gomdp_best.pt` aborts loudly

**Issue 6 — Circular invariant checker** (`src/wildfire_governance/gomdp/invariant_checker.py`)
- Independent re-evaluation: requires `cert is not None AND len(cert)==64 AND confidence>τ AND human_approval`
- Added poisoned-trajectory test: `alert=True, cert=None` is correctly flagged as a violation
- Added non-vacuous test: confirms governance pipeline was actually invoked

**Issue 7 — VIIRS hardcoded scalar** (`experiments/_viirs_runner.py`)
- `_run_iot_threshold_baseline()` computes Ld from the same VIIRS grids and same ground truth
- Removed `iot_baseline_ld = 45.0` from all code paths
- Speedup metric is now a fair, co-evaluated comparison

### Moderate Fixes

- **M5** — Fixed `from _viirs_runner import` → `from experiments._viirs_runner import` in `08b/c`
- **M6** — Checkpoint saved only when `compliance == 1.0 AND ep_ld < best_compliant_ld`
- **Mi4** — Removed unused `torchvision` from `environment.yml`

## [1.0.0] — 2025-04-05

### Added
- GOMDP formal framework (Definition 1, Theorem 1, Theorem 2)
- PPO-GOMDP deep RL policy
- Hyperledger Fabric blockchain simulation
- Two-stage Bayesian cross-modal fusion verification pipeline
- Hierarchical multi-agent UAV coordination
- Real-world VIIRS adapter
- Adversarial robustness suite
- Full experiment scripts (01–12)
- Docker + GitHub Actions CI
