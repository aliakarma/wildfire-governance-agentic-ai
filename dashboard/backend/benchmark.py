"""Live multi-seed benchmark — real numbers with confidence intervals.

Runs the requested methods across N seeds through the same streaming engine the
live viewer uses (``simulation_service.summarize_episode``), then aggregates
mean / std / 95% CI. This is what the Reproducibility panel diffs against the
paper reference — it computes; it does NOT read results/paper/*.csv as its own
output.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .schema import METHOD_PRESETS, method_flags
from .simulation_service import summarize_episode


def _mean_std_ci(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": 0.0, "ci95": 0.0, "n": 0}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "std": 0.0, "ci95": 0.0, "n": 1}
    from scipy import stats
    std = float(arr.std(ddof=1))
    t = float(stats.t.ppf(0.975, df=arr.size - 1))
    return {"mean": mean, "std": std, "ci95": float(t * std / np.sqrt(arr.size)), "n": int(arr.size)}


def run_benchmark(
    methods: List[str],
    n_seeds: int = 5,
    n_uavs: int = 16,
    grid_size: int = 60,
    n_timesteps: int = 300,
    tau: float = 0.72,
) -> Dict[str, Any]:
    """Compute a live comparison table (mean ± 95% CI) across seeds."""
    n_seeds = max(2, min(int(n_seeds), 30))
    rows: List[Dict[str, Any]] = []
    raw: List[Dict[str, Any]] = []

    for method in methods:
        if method not in METHOD_PRESETS:
            continue
        f = method_flags(method)
        lds: List[float] = []
        fps: List[float] = []
        comps: List[float] = []
        for seed in range(n_seeds):
            summary = summarize_episode({
                "method": method, "grid_size": grid_size, "n_uavs": n_uavs,
                "n_timesteps": n_timesteps, "seed": seed, "tau": tau,
            }).get("summary", {})
            ld = summary.get("ld")
            fp = summary.get("fp_pct", 0.0)
            comp = summary.get("compliance", 100.0)
            lds.append(ld)
            fps.append(fp)
            comps.append(comp)
            raw.append({"method": method, "seed": seed, "ld": ld, "fp_pct": fp, "compliance": comp})
        ld_s, fp_s = _mean_std_ci(lds), _mean_std_ci(fps)
        rows.append({
            "method": method, "label": f["label"], "enforcement": f["enforcement"],
            "ld_mean": round(ld_s["mean"], 2), "ld_std": round(ld_s["std"], 2), "ld_ci95": round(ld_s["ci95"], 2),
            "fp_mean": round(fp_s["mean"], 2), "fp_std": round(fp_s["std"], 2), "fp_ci95": round(fp_s["ci95"], 2),
            "compliance_pct": round(float(np.mean(comps)), 1), "n_seeds": n_seeds,
        })
    return {"source": "live", "rows": rows, "raw": raw,
            "config": {"n_seeds": n_seeds, "n_uavs": n_uavs, "grid_size": grid_size,
                       "n_timesteps": n_timesteps, "tau": tau}}
