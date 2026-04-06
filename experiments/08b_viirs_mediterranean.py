#!/usr/bin/env python3
"""Experiment 08b — Real-world VIIRS validation: Mediterranean 2021 (Evia Island fires).

Paper reference: Table VI, Section VI-C (Real-World Validation).
Output: results/runs/<hash>/table6_mediterranean.csv

DATA REQUIREMENT:
    data/processed/viirs_grid_mediterranean_2021.npz
    Run: python data/scripts/download_viirs.py --region mediterranean ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

VIIRS_PATH = Path("data/processed/viirs_grid_mediterranean_2021.npz")


def main(config_path: str, smoke: bool = False) -> None:
    # Import shared VIIRS runner from the california script
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from _viirs_runner import run_viirs_region  # type: ignore[import]

    run_viirs_region(
        region="mediterranean_2021",
        viirs_path=VIIRS_PATH,
        nifc_path=None,
        output_stem="table6_mediterranean",
        config_path=config_path,
        smoke=smoke,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/realworld_viirs.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
