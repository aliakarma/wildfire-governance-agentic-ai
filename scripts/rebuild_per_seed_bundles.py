#!/usr/bin/env python3
"""Rebuild results/paper/per_seed/seed_<n>.json from the per-seed CSVs.

The seed bundles are a convenience view: one JSON per seed holding that seed's
row from every per-seed metric file. They are derived, never edited by hand, so
regenerating them after any change to the CSVs keeps the two representations
from drifting apart (and drops tables withdrawn from the manuscript).

    python scripts/rebuild_per_seed_bundles.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PER_SEED = Path("results/paper/per_seed")

# canonical per-seed CSV -> key in the seed bundle
SOURCES = {
    "table1_rl_comparison": "table1_rl_comparison_per_seed.csv",
    "table1_rl_comparison_main": "table3_main_comparison_per_seed.csv",
    "table2_ablation": "table2_ablation_per_seed.csv",
    "table3_adversarial": "table3_adversarial_per_seed.csv",
    "table4_realworld_viirs": "table4_realworld_viirs_per_seed.csv",
    "table7_hitl_sensitivity": "table7_hitl_sensitivity_per_seed.csv",
    "table9_multisig": "table9_multisig_per_seed.csv",
    "fig2_false_alerts": "fig2_false_alerts_per_seed.csv",
    "fig3_latency_data": "fig3_latency_data_per_seed.csv",
    "fig5_tradeoff_data": "fig5_tradeoff_data_per_seed.csv",
}


def main() -> None:
    frames = {}
    for key, fname in SOURCES.items():
        path = PER_SEED / fname
        if not path.exists():
            raise SystemExit(f"missing per-seed source: {path}")
        frames[key] = pd.read_csv(path)

    seeds = sorted(set(frames["table1_rl_comparison"]["seed"].astype(int)))
    for old in PER_SEED.glob("seed_*.json"):
        old.unlink()

    for seed in seeds:
        bundle: dict = {"seed": int(seed)}
        for key, df in frames.items():
            sub = df[df["seed"] == seed].drop(columns=["seed"])
            bundle[key] = json.loads(sub.to_json(orient="records"))
        out = PER_SEED / f"seed_{seed}.json"
        out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"rebuilt {len(seeds)} seed bundles in {PER_SEED}")


if __name__ == "__main__":
    main()
