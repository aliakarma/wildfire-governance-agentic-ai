"""Parameter schema + validation for the dashboard.

A single source of truth that the frontend control panel reads (via
``GET /api/config/schema``) and the backend uses to clamp/validate any
incoming run request. Keeping ranges here prevents a user from launching an
unbounded (DoS-shaped) live run.
"""
from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Parameter schema — drives the UI control panel and server-side validation.
# ---------------------------------------------------------------------------
PARAM_SCHEMA: Dict[str, Dict[str, Any]] = {
    "grid_size":           {"type": "int",   "min": 20,  "max": 200,  "step": 10,  "default": 100,  "unit": "cells", "i18n": "param.grid_size"},
    "n_uavs":              {"type": "int",   "min": 1,   "max": 60,   "step": 1,   "default": 20,   "unit": "UAVs",  "i18n": "param.n_uavs"},
    "n_sectors":           {"type": "int",   "min": 4,   "max": 100,  "step": 1,   "default": 25,   "unit": "Z",     "i18n": "param.n_sectors"},
    "n_timesteps":         {"type": "int",   "min": 100, "max": 3000, "step": 100, "default": 600,  "unit": "steps", "i18n": "param.n_timesteps"},
    "tau":                 {"type": "float", "min": 0.50,"max": 0.99, "step": 0.01,"default": 0.80, "unit": "τ","i18n": "param.tau"},
    "seed":                {"type": "int",   "min": 0,   "max": 9999, "step": 1,   "default": 0,    "unit": "",      "i18n": "param.seed"},
    "policy":              {"type": "enum",  "options": ["greedy", "ppo"], "default": "greedy", "i18n": "param.policy"},
    "method":              {"type": "enum",  "options": ["ppo_gomdp", "greedy_gomdp", "central_sig", "ppo_cmdp", "adaptive_ai", "static"], "default": "ppo_gomdp", "i18n": "param.method"},
    # Adversarial
    "attack_type":         {"type": "enum",  "options": ["none", "spoofing", "spoofing_strategic", "injection", "byzantine"], "default": "none", "i18n": "param.attack"},
    "p_spoof":             {"type": "float", "min": 0.0, "max": 0.5,  "step": 0.01,"default": 0.0,  "unit": "p",     "i18n": "param.p_spoof"},
    "n_byzantine":         {"type": "int",   "min": 0,   "max": 3,    "step": 1,   "default": 0,    "unit": "f",     "i18n": "param.n_byzantine"},
    "p_drop":              {"type": "float", "min": 0.0, "max": 0.3,  "step": 0.01,"default": 0.0,  "unit": "p",     "i18n": "param.p_drop"},
    "sensor_failure_rate": {"type": "float", "min": 0.0, "max": 0.4,  "step": 0.05,"default": 0.0,  "unit": "%",     "i18n": "param.sensor_fail"},
}

# ---------------------------------------------------------------------------
# Method presets → governance/architecture flags used by run_episode.
# Mirrors the configurations in experiments/11b_rl_comparison.py & 02_ablation.
# ---------------------------------------------------------------------------
METHOD_PRESETS: Dict[str, Dict[str, Any]] = {
    "ppo_gomdp":    {"label": "PPO-GOMDP",    "enforcement": "crypto",     "governance": True,  "hitl": True,  "blockchain": True,  "verification": True,  "coordination": True,  "policy": "ppo"},
    "greedy_gomdp": {"label": "Greedy-GOMDP", "enforcement": "crypto",     "governance": True,  "hitl": True,  "blockchain": True,  "verification": True,  "coordination": True,  "policy": "greedy"},
    "central_sig":  {"label": "Central+Sig",  "enforcement": "signature",  "governance": True,  "hitl": True,  "blockchain": False, "verification": True,  "coordination": True,  "policy": "greedy"},
    "ppo_cmdp":     {"label": "PPO-CMDP",     "enforcement": "lagrangian", "governance": False, "hitl": True,  "blockchain": False, "verification": True,  "coordination": True,  "policy": "ppo"},
    "adaptive_ai":  {"label": "Adaptive AI",  "enforcement": "none",       "governance": False, "hitl": False, "blockchain": False, "verification": True,  "coordination": True,  "policy": "greedy"},
    "static":       {"label": "Static",       "enforcement": "none",       "governance": False, "hitl": False, "blockchain": False, "verification": False, "coordination": False, "policy": "greedy"},
}

# UI colour token per method (kept in sync with Dashboard_Guide.md §9.1).
METHOD_COLORS: Dict[str, Dict[str, str]] = {
    "ppo_gomdp":    {"light": "#E4572E", "dark": "#FF6B3D"},
    "greedy_gomdp": {"light": "#C77D0A", "dark": "#F2B455"},
    "central_sig":  {"light": "#2D6BB0", "dark": "#6FB1FF"},
    "ppo_cmdp":     {"light": "#7A5AF8", "dark": "#A78BFA"},
    "adaptive_ai":  {"light": "#8A8F98", "dark": "#9AA6B8"},
    "static":       {"light": "#5B616E", "dark": "#6B7280"},
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def validate_and_default(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a fully-populated, clamped parameter dict.

    Unknown keys are dropped; missing keys get schema defaults; numeric values
    are clamped to their [min, max] range; enums fall back to their default if
    an invalid option is supplied.
    """
    params = params or {}
    out: Dict[str, Any] = {}
    for name, spec in PARAM_SCHEMA.items():
        raw = params.get(name, spec.get("default"))
        kind = spec["type"]
        if kind == "int":
            try:
                val = int(round(float(raw)))
            except (TypeError, ValueError):
                val = int(spec["default"])
            out[name] = int(_clamp(val, spec["min"], spec["max"]))
        elif kind == "float":
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = float(spec["default"])
            out[name] = float(_clamp(val, spec["min"], spec["max"]))
        elif kind == "enum":
            out[name] = raw if raw in spec["options"] else spec["default"]
        else:
            out[name] = raw

    # Cross-field guardrail: cap total work for the live path.
    if out["grid_size"] * out["n_timesteps"] > 200 * 3000:
        out["n_timesteps"] = max(100, (200 * 3000) // out["grid_size"])
    return out


def method_flags(method: str) -> Dict[str, Any]:
    """Return governance/architecture flags for a method preset."""
    return METHOD_PRESETS.get(method, METHOD_PRESETS["ppo_gomdp"])
