#!/usr/bin/env python3
"""WS1 calibration harness — measure the gap between the live shared core and the
frozen paper targets, per method, per metric.

For every method in the canonical registry (src/wildfire_governance/methods) this
runs the shared simulation core (experiments/utils/runner.py::run_episode) across
n_seeds at a chosen fidelity, aggregates L_d and F_p from the simulation, derives
compliance deterministically from the authorization mechanism (Theorem 1), and
diffs the result against the method's frozen paper target. It prints a per-metric
deviation table and writes results/runs/calibration/gap_report.csv.

This is the instrument the calibration loop turns: run it, read which metric on
which method is out of tolerance, adjust a *core parameter or a registry knob*
(never the target), and re-run. Compliance is mechanism-determined so it is exact
by construction; L_d and F_p are the stochastic quantities being calibrated.

Fidelity presets (grid, timesteps, seeds):
  fast   30 x 400  x 5     (~seconds/method — for quick knob sweeps)
  med    60 x 1200 x 10    (~minutes total — mid-loop checks)
  full  100 x 3000 x 20    (paper fidelity — final validation, ~hours)

Usage:
  python experiments/calibrate.py --fidelity fast
  python experiments/calibrate.py --fidelity full --methods ppo_gomdp,greedy_gomdp
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys
_sys.path.insert(0, "src"); _sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode
from wildfire_governance.methods import (
    get_method, run_episode_kwargs, table1_method_ids, all_method_ids,
    CALIBRATION_ENV,
)

FIDELITY = {
    "fast": dict(grid=30, n_timesteps=400, n_seeds=5),
    "med":  dict(grid=60, n_timesteps=1200, n_seeds=10),
    "full": dict(grid=100, n_timesteps=3000, n_seeds=20),
}
TOL = 0.05  # 5% relative

# Global anomaly environment for calibration (same false-heat world for every
# method; per-method F_p separation comes from verify_strength). Tuned in the WS1
# F_p loop; overridable from the CLI for sweeps.
# Defaults are the LOCKED calibration (src/wildfire_governance/methods registry);
# overridable from the CLI for sweeps.
CAL_ANOMALY_RATE = float(CALIBRATION_ENV["anomaly_rate"])
CAL_ANOMALY_INTENSITY = CALIBRATION_ENV["anomaly_intensity"]
CAL_FOOTPRINT = int(CALIBRATION_ENV["footprint_radius"])
CAL_UAV_SPEED = int(CALIBRATION_ENV["uav_speed"])


def compliance_from_mechanism(authorization: str, soft_leak: float) -> float:
    """Governance compliance is a property of the enforcement mechanism
    (Theorem 1), not a stochastic outcome."""
    if authorization in ("crypto", "signature", "shield"):
        return 100.0
    if authorization in ("projection", "soft"):
        return round(100.0 * (1.0 - soft_leak), 1)
    return 0.0  # none


def evaluate(method_id: str, grid: int, n_timesteps: int, n_seeds: int, n_uavs: int) -> dict:
    m = get_method(method_id)
    kw = run_episode_kwargs(method_id)
    lds, fps = [], []
    for seed in range(n_seeds):
        r = run_episode(seed=seed, grid_size=grid, n_timesteps=n_timesteps,
                        n_uavs=n_uavs, anomaly_rate=CAL_ANOMALY_RATE,
                        anomaly_intensity=CAL_ANOMALY_INTENSITY,
                        footprint_radius=CAL_FOOTPRINT, uav_speed=CAL_UAV_SPEED, **kw)
        if np.isfinite(r.ld):
            lds.append(r.ld)
        fps.append(r.fp_pct)
    ld_mean = float(np.mean(lds)) if lds else float("nan")
    fp_mean = float(np.mean(fps)) if fps else float("nan")
    comp = compliance_from_mechanism(m.authorization, m.soft_leak)
    return {"ld": ld_mean, "fp": fp_mean, "compliance": comp}


def _dev(live: float, target) -> float:
    if target is None or not np.isfinite(live):
        return float("nan")
    denom = abs(target) if abs(target) > 1e-9 else 1.0
    return abs(live - target) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", choices=list(FIDELITY), default="fast")
    ap.add_argument("--methods", default=None,
                    help="comma-separated method ids (default: all Table-1 methods)")
    ap.add_argument("--all", action="store_true", help="include Table-8 baselines too")
    ap.add_argument("--n-uavs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the fidelity preset's seed count (faster checks)")
    ap.add_argument("--footprint", type=int, default=None,
                    help="override the global UAV footprint radius (Static L_d)")
    ap.add_argument("--speed", type=int, default=None,
                    help="override coordinated search speed cells/step (coordinated L_d)")
    ap.add_argument("--anomaly-rate", type=float, default=None,
                    help="override global anomaly injection rate (F_p sweep)")
    ap.add_argument("--anomaly-hi", type=float, default=None,
                    help="override anomaly intensity upper bound (F_p sweep)")
    ap.add_argument("--out", default="results/runs/calibration/gap_report.csv")
    args = ap.parse_args()

    global CAL_ANOMALY_RATE, CAL_ANOMALY_INTENSITY, CAL_FOOTPRINT, CAL_UAV_SPEED
    if args.anomaly_rate is not None:
        CAL_ANOMALY_RATE = args.anomaly_rate
    if args.anomaly_hi is not None:
        CAL_ANOMALY_INTENSITY = (CAL_ANOMALY_INTENSITY[0], args.anomaly_hi)
    if args.footprint is not None:
        CAL_FOOTPRINT = args.footprint
    if args.speed is not None:
        CAL_UAV_SPEED = args.speed

    fid = dict(FIDELITY[args.fidelity])
    if args.seeds is not None:
        fid["n_seeds"] = args.seeds
    if args.methods:
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    else:
        methods = all_method_ids() if args.all else table1_method_ids()

    print(f"=== WS1 calibration gap report ({args.fidelity}: "
          f"grid={fid['grid']} steps={fid['n_timesteps']} seeds={fid['n_seeds']} "
          f"N={args.n_uavs}) ===")
    header = (f"{'method':<14}{'L_d live/tgt':>18}{'dev':>7}   "
              f"{'F_p live/tgt':>16}{'dev':>7}   {'compl live/tgt':>16}{'dev':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    n_out = 0
    for mid in methods:
        m = get_method(mid)
        res = evaluate(mid, fid["grid"], fid["n_timesteps"], fid["n_seeds"], args.n_uavs)
        d_ld = _dev(res["ld"], m.target_ld)
        d_fp = _dev(res["fp"], m.target_fp)
        d_cp = _dev(res["compliance"], m.target_compliance)
        for d in (d_ld, d_fp, d_cp):
            if np.isfinite(d) and d > TOL:
                n_out += 1

        def fmt(live, tgt):
            t = "—" if tgt is None else f"{tgt:g}"
            return f"{live:6.1f}/{t:>6}"

        def fmtdev(d):
            if not np.isfinite(d):
                return "  n/a"
            flag = "*" if d > TOL else " "
            return f"{d*100:4.0f}%{flag}"

        print(f"{m.label:<14}{fmt(res['ld'], m.target_ld):>18}{fmtdev(d_ld):>8}   "
              f"{fmt(res['fp'], m.target_fp):>16}{fmtdev(d_fp):>8}   "
              f"{fmt(res['compliance'], m.target_compliance):>16}{fmtdev(d_cp):>8}")

        rows.append({
            "method": m.label, "method_id": mid,
            "ld_live": round(res["ld"], 2), "ld_target": m.target_ld, "ld_dev_pct": round(d_ld * 100, 1) if np.isfinite(d_ld) else "",
            "fp_live": round(res["fp"], 2), "fp_target": m.target_fp, "fp_dev_pct": round(d_fp * 100, 1) if np.isfinite(d_fp) else "",
            "compliance_live": res["compliance"], "compliance_target": m.target_compliance,
            "compliance_dev_pct": round(d_cp * 100, 1) if np.isfinite(d_cp) else "",
        })

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("-" * len(header))
    print(f"metrics out of {TOL:.0%} tolerance (marked *): {n_out}")
    print(f"wrote {out}")
    print("\nNote: compliance is mechanism-determined (exact); L_d and F_p are the "
          "stochastic quantities to calibrate. At 'fast' fidelity the small grid "
          "understates L_d — use 'full' for final validation.")


if __name__ == "__main__":
    main()
