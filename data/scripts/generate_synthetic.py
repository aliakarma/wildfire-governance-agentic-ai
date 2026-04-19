#!/usr/bin/env python3
"""Ensure required synthetic wildfire .npz files exist for reproducible runs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = [
    (10, 42, "data/synthetic/grid_10x10_seed42.npz"),
    (100, 0, "data/synthetic/grid_100x100_seed0.npz"),
]


def main() -> None:
    generator = Path("data/synthetic/generate_synthetic.py")
    if not generator.exists():
        raise FileNotFoundError(f"Missing generator script: {generator}")

    for size, seed, output_path in REQUIRED_FILES:
        output = Path(output_path)
        if output.exists():
            print(f"Synthetic file already exists: {output}")
            continue

        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Generating synthetic file: {output}")
        subprocess.run(
            [
                sys.executable,
                str(generator),
                "--size",
                str(size),
                "--seed",
                str(seed),
                "--output",
                str(output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
