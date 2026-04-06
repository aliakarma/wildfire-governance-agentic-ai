#!/usr/bin/env python3
"""Experiment 08c — Real-world VIIRS validation: Australia Black Summer 2019-20.

FIX M5: Replaced bare `from _viirs_runner import` with package-relative import.

Paper reference: Table VI, Section VI-C (Real-World Validation).
Output: results/runs/<hash>/table6_australia.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

VIIRS_PATH = Path("data/processed/viirs_grid_australia_2019.npz")


def main(config_path: str, smoke: bool = False) -> None:
    # FIX M5: use package-relative import
    from experiments._viirs_runner import run_viirs_region  # type: ignore[import]

    run_viirs_region(
        region="australia_2019",
        viirs_path=VIIRS_PATH,
        nifc_path=None,
        output_stem="table6_australia",
        config_path=config_path,
        smoke=smoke,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/realworld_viirs.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
