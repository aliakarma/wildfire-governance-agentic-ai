"""Canonical method registry (WS1) — ONE taxonomy for all 11 methods.

Every method that appears in the manuscript (Table 1's nine, plus SafeDreamer and
CCPO from Table 8) is defined here exactly once, mapping a method id to:

  * the structural flags the shared simulation core (experiments/utils/runner.py
    :: run_episode) consumes — enable_governance / hitl / blockchain /
    verification / coordination, and policy;
  * the fine-grained authorization taxonomy the dashboard already uses to make
    methods behave distinctly — authorization / verify_strength / soft_leak /
    search (documented in dashboard/backend/schema.py); and
  * the frozen paper target (L_d, F_p, compliance) used as the calibration oracle
    by experiments/calibrate.py.

This module is the single source of truth: both the experiments side and the
dashboard schema resolve method definitions from here, so there is one taxonomy,
not two. Calibrating a method means adjusting the *core* parameters and these
per-method knobs until a live multi-seed run reproduces the paper target within
tolerance — never editing the target.

authorization semantics (decides governance compliance):
  crypto     on-chain cert (HITL + signature + PBFT consensus)   -> 100%
  signature  HITL + signature, no consensus                      -> 100%
  shield     logical shield enforcing G(s,a) locally             -> 100%
  projection learned safety layer (SafeLayer): near-total        -> 100*(1-soft_leak)
  soft       Lagrangian / worst-case penalty (in-expectation)    -> 100*(1-soft_leak)
  none       broadcast with no authorisation                     -> 0%
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MethodSpec:
    """One method's canonical definition."""

    method_id: str
    label: str
    framework: str
    enforcement: str
    # Structural flags consumed by run_episode today.
    governance: bool
    hitl: bool
    blockchain: bool
    verification: bool
    coordination: bool
    policy: str                 # "greedy" | "ppo" | "static"
    # Fine-grained taxonomy (dashboard + calibrated core).
    authorization: str          # crypto|signature|shield|projection|soft|none
    verify_strength: float      # discriminative power of the verification pipeline
    soft_leak: float            # fraction of soft-path broadcasts left unauthorised
    search: str                 # "ppo" | "greedy" | "static"
    # Frozen paper target (calibration oracle); None where the paper omits it.
    target_ld: Optional[float] = None
    target_fp: Optional[float] = None
    target_compliance: Optional[float] = None
    in_table1: bool = True      # False for Table-8 recent-RL baselines
    alert_threshold: Optional[float] = None  # raise the raw broadcast bar (Static)
    detection_probability: Optional[float] = None  # per-method sensor reliability (Static)


# ── The 11 methods ───────────────────────────────────────────────────────────
# verify_strength / soft_leak are calibration starting points (tuned in WS1);
# targets are frozen from results/paper/table1_rl_comparison.csv and table8.
_SPECS: List[MethodSpec] = [
    MethodSpec("ppo_gomdp", "PPO-GOMDP", "GOMDP", "Crypto",
               True, True, True, True, True, "ppo",
               "crypto", 0.89, 0.0, "ppo",
               target_ld=15.1, target_fp=6.0, target_compliance=100.0),
    MethodSpec("greedy_gomdp", "Greedy-GOMDP", "GOMDP", "Crypto",
               True, True, True, True, True, "greedy",
               "crypto", 0.89, 0.0, "greedy",
               target_ld=18.3, target_fp=6.1, target_compliance=100.0),
    MethodSpec("central_sig", "Central+Sig", "GOMDP", "Sig. only",
               True, True, False, True, True, "ppo",
               "signature", 0.90, 0.0, "ppo",
               target_ld=15.0, target_fp=6.0, target_compliance=100.0),
    MethodSpec("shield_ppo", "Shield-PPO", "Logical", "Logical",
               True, False, False, True, True, "ppo",
               "shield", 0.85, 0.0, "ppo",
               target_ld=15.2, target_fp=6.2, target_compliance=100.0),
    MethodSpec("safelayer", "SafeLayer", "Learned", "Learned",
               True, False, False, True, True, "ppo",
               "projection", 0.82, 0.016, "ppo",
               target_ld=14.9, target_fp=7.0, target_compliance=98.4),
    MethodSpec("ppo_cmdp", "PPO-CMDP", "CMDP", "Lagrangian",
               False, True, False, True, True, "ppo",
               "soft", 0.77, 0.072, "ppo",
               target_ld=14.8, target_fp=8.3, target_compliance=92.8),
    MethodSpec("wcsac", "WCSAC", "CMDP", "Lagrangian",
               False, True, False, True, True, "ppo",
               "soft", 0.74, 0.094, "ppo",
               target_ld=14.6, target_fp=9.4, target_compliance=90.6),
    MethodSpec("adaptive_ai", "Adaptive AI", "None", "None",
               False, False, False, True, True, "greedy",
               "none", 0.30, 1.0, "greedy",
               target_ld=16.2, target_fp=22.4, target_compliance=0.0),
    MethodSpec("static", "Static", "None", "None",
               False, False, False, False, False, "static",
               "none", 0.80, 1.0, "static",
               target_ld=41.5, target_fp=15.3, target_compliance=0.0,
               alert_threshold=0.87, detection_probability=0.38),
    # Table 8 — recent constrained-RL baselines (not in Table 1).
    MethodSpec("safedreamer", "SafeDreamer", "CMDP", "Lagrangian",
               False, True, False, True, True, "ppo",
               "soft", 0.78, 0.055, "ppo",
               target_ld=14.7, target_fp=8.1, target_compliance=94.5, in_table1=False),
    MethodSpec("ccpo", "CCPO", "CMDP", "Projection",
               False, True, False, True, True, "ppo",
               "soft", 0.80, 0.047, "ppo",
               target_ld=14.9, target_fp=7.6, target_compliance=95.3, in_table1=False),
]

METHODS: Dict[str, MethodSpec] = {m.method_id: m for m in _SPECS}

# ── Locked global calibration (WS1) ──────────────────────────────────────────
# The calibrated environment parameters that, together with the per-method knobs
# above, reproduce the paper's QUALITATIVE claims at grid 100: exact governance
# compliance, governed-low vs ungoverned-high F_p, and coordinated-fast vs
# static-slow L_d ordering. Absolute L_d/F_p magnitudes are DOCUMENTED DEVIATIONS
# (seed variance + F_p denominator saturation at 3000 steps + back-filled targets;
# the paper itself states latencies are "relative comparisons, not field
# calibrated"). See results/paper/CALIBRATION.md and the KNOWN_DEVIATIONS table in
# scripts/check_reproducibility.py.
CALIBRATION_ENV: Dict[str, object] = {
    "footprint_radius": 5,          # coordinated L_d ~ paper range at grid 100
    "anomaly_rate": 2.0,            # F_p ordering; magnitude saturates (documented)
    "anomaly_intensity": (0.55, 0.99),
    "uav_speed": 1,                 # coordinated-L_d backup lever (unused at rest)
}


def get_method(method_id: str) -> MethodSpec:
    if method_id not in METHODS:
        raise KeyError(f"unknown method '{method_id}'; known: {list(METHODS)}")
    return METHODS[method_id]


def all_method_ids() -> List[str]:
    return list(METHODS)


def table1_method_ids() -> List[str]:
    return [m.method_id for m in _SPECS if m.in_table1]


def run_episode_kwargs(method_id: str) -> Dict[str, Any]:
    """Kwargs for experiments/utils/runner.py::run_episode.

    Returns the structural flags plus the calibrated authorization knobs. The
    core honours the knobs it understands and ignores the rest, so this stays
    forward-compatible as run_episode grows to consume verify_strength /
    authorization / soft_leak during calibration.
    """
    m = get_method(method_id)
    return {
        "config_name": m.method_id,
        "enable_governance": m.governance,
        "enable_hitl": m.hitl,
        "enable_blockchain": m.blockchain,
        "enable_verification": m.verification,
        "enable_coordination": m.coordination,
        "policy": m.policy,
        "authorization": m.authorization,
        "verify_strength": m.verify_strength,
        "soft_leak": m.soft_leak,
        "search": m.search,
        "alert_threshold": m.alert_threshold,
        "detection_probability": m.detection_probability,
    }


def target(method_id: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(L_d, F_p, compliance) frozen paper target for this method."""
    m = get_method(method_id)
    return m.target_ld, m.target_fp, m.target_compliance
