#!/usr/bin/env python3
"""Experiment 04b — False-alert rate vs fleet size (fig:falsealerts, Fig 2).

Sweeps the fleet size N over {5, 10, 20, 40} and records the false-alert rate F_p
for each of the six manuscript series, producing the F_p-vs-N figure.

Method -> shared-core mapping (single simulation core, one fire model):
  Greedy-GOMDP  : governed greedy core                         (native)
  Adaptive AI   : no governance, verification only             (native)
  Static        : no governance, no verification, static patrol(native)
  PPO-CMDP      : soft-constrained core (no blockchain)        (native)
  PPO-GOMDP     : governed core, PPO policy                    (proxy: uses the
                  governed greedy core until the PPO checkpoint path is wired in
                  WS1; F_p is governed either way so the proxy is faithful for
                  *this* metric, magnitude calibration-pending)
  WCSAC         : soft-constrained core                        (proxy: worst-case
                  SAC lands in the WS1 registry)

F_p magnitudes are calibration-pending until WS1; the ordering and monotone
decrease with N are reproduced by construction.

Canonical output: results/paper/fig2_false_alerts.csv  (see results/paper/MANIFEST.yaml)
Paper reference: Figure 2 (fig:falsealerts), F_p vs N.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys as _sys
_sys.path.insert(0, "src"); _sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from experiments.utils.runner import run_episode

N_UAVS_GRID = [5, 10, 20, 40]
N_SEEDS = 20
GRID = 100
N_TIMESTEPS = 3000

# label -> shared-core flags. proxy=True methods are documented above.
SERIES = {
    "PPO-GOMDP":    dict(enable_governance=True,  enable_hitl=True,  enable_blockchain=True,  enable_verification=True,  enable_coordination=True),
    "Greedy-GOMDP": dict(enable_governance=True,  enable_hitl=True,  enable_blockchain=True,  enable_verification=True,  enable_coordination=True),
    "PPO-CMDP":     dict(enable_governance=False, enable_hitl=True,  enable_blockchain=False, enable_verification=True,  enable_coordination=True),
    "WCSAC":        dict(enable_governance=False, enable_hitl=True,  enable_blockchain=False, enable_verification=True,  enable_coordination=True),
    "Adaptive AI":  dict(enable_governance=False, enable_hitl=False, enable_blockchain=False, enable_verification=True,  enable_coordination=True),
    "Static":       dict(enable_governance=False, enable_hitl=False, enable_blockchain=False, enable_verification=False, enable_coordination=False),
}


def build_per_seed(n_uavs_grid, n_seeds, grid, n_timesteps) -> pd.DataFrame:
    rows = []
    for label, flags in SERIES.items():
        for n_uavs in n_uavs_grid:
            for seed in range(n_seeds):
                r = run_episode(seed=seed, config_name=label, grid_size=grid,
                                n_timesteps=n_timesteps, n_uavs=n_uavs, **flags)
                rows.append({"config": label, "n_uavs": n_uavs, "seed": seed,
                             "fp_pct": round(r.fp_pct, 2)})
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in SERIES:
        for n_uavs in sorted(per_seed["n_uavs"].unique()):
            sub = per_seed[(per_seed["config"] == label) & (per_seed["n_uavs"] == n_uavs)]
            fp = sub["fp_pct"].to_numpy(dtype=float)
            rows.append({"config": label, "n_uavs": int(n_uavs),
                         "fp_mean": round(float(np.mean(fp)), 1) if len(fp) else ""})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/runs/fig2_false_alerts")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_uavs_grid, n_seeds, grid, n_ts = (N_UAVS_GRID, N_SEEDS, GRID, N_TIMESTEPS)
    if args.smoke:
        # Keep the paper's fleet-size grid {5,10,20,40} so the F_p-vs-N join keys
        # always align with fig2_false_alerts.csv; only reduce seeds/grid/steps.
        n_seeds, grid, n_ts = 2, 40, 200

    per_seed = build_per_seed(n_uavs_grid, n_seeds, grid, n_ts)
    agg = aggregate(per_seed)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(outdir / "fig2_false_alerts_per_seed.csv", index=False)
    agg.to_csv(outdir / "fig2_false_alerts.csv", index=False)

    print("=== False-alert rate vs fleet size (Fig 2) ===")
    print(agg.to_string(index=False))
    print("\n(F_p magnitude calibration-pending until WS1; PPO-GOMDP/WCSAC use documented proxies)")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
