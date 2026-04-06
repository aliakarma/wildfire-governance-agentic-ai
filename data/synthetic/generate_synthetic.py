#!/usr/bin/env python3
"""Generate synthetic wildfire grid data for unit tests and smoke tests.

Produces .npz files containing: grid, fire_mask, uav_positions, wind_field,
humidity_field, fuel_map, and metadata.

Usage:
    # Bash
    python data/synthetic/generate_synthetic.py --size 10 --seed 42 \
        --output data/synthetic/grid_10x10_seed42.npz

    python data/synthetic/generate_synthetic.py --size 100 --seed 0 \
        --output data/synthetic/grid_100x100_seed0.npz

    # PowerShell (same commands)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate_synthetic_grid(
    size: int = 100,
    seed: int = 42,
    n_ignition_points: int = 3,
    n_uavs: int = 10,
    wind_speed: float = 0.5,
    humidity: float = 0.4,
    fuel_density: float = 0.7,
) -> dict:
    """Generate a synthetic wildfire grid environment.

    Args:
        size: Grid side length N (produces N×N arrays).
        seed: Random seed for reproducibility.
        n_ignition_points: Number of fire ignition cells.
        n_uavs: Number of UAV positions to place.
        wind_speed: Mean wind speed (0–1).
        humidity: Mean humidity (0–1).
        fuel_density: Mean fuel load (0–1).

    Returns:
        Dict with arrays: grid, fire_mask, uav_positions, wind_field,
        humidity_field, fuel_map, metadata (JSON string).
    """
    rng = np.random.default_rng(seed)

    fuel_map = rng.uniform(fuel_density * 0.5, min(1.0, fuel_density * 1.3), (size, size)).astype(np.float32)
    humidity_field = rng.uniform(humidity * 0.6, min(1.0, humidity * 1.4), (size, size)).astype(np.float32)
    wind_field = rng.uniform(0.0, wind_speed * 1.5, (size, size)).astype(np.float32)

    fire_mask = np.zeros((size, size), dtype=np.float32)
    rows = rng.integers(0, size, size=n_ignition_points)
    cols = rng.integers(0, size, size=n_ignition_points)
    fire_mask[rows, cols] = 1.0

    heat_map = fire_mask.copy()
    noise = rng.normal(0, 0.03, (size, size)).astype(np.float32)
    grid = np.clip(heat_map + noise, 0.0, 1.0).astype(np.float32)

    uav_rows = rng.integers(0, size, size=n_uavs)
    uav_cols = rng.integers(0, size, size=n_uavs)
    uav_positions = np.stack([uav_rows, uav_cols], axis=1).astype(np.int32)

    metadata = json.dumps({
        "size": size,
        "seed": seed,
        "n_ignition_points": n_ignition_points,
        "n_uavs": n_uavs,
        "wind_speed": wind_speed,
        "humidity": humidity,
        "fuel_density": fuel_density,
        "generator": "data/synthetic/generate_synthetic.py",
    })

    return {
        "grid": grid,
        "fire_mask": fire_mask,
        "wind_field": wind_field,
        "humidity_field": humidity_field,
        "fuel_map": fuel_map,
        "uav_positions": uav_positions,
        "metadata": np.array(metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic wildfire grid data")
    parser.add_argument("--size", type=int, default=100, help="Grid side length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_ignition_points", type=int, default=3)
    parser.add_argument("--n_uavs", type=int, default=10)
    parser.add_argument("--wind_speed", type=float, default=0.5)
    parser.add_argument("--humidity", type=float, default=0.4)
    parser.add_argument("--fuel_density", type=float, default=0.7)
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path. Default: data/synthetic/grid_<size>x<size>_seed<seed>.npz",
    )
    args = parser.parse_args()

    out = Path(args.output) if args.output else (
        Path("data/synthetic") / f"grid_{args.size}x{args.size}_seed{args.seed}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_grid(
        size=args.size,
        seed=args.seed,
        n_ignition_points=args.n_ignition_points,
        n_uavs=args.n_uavs,
        wind_speed=args.wind_speed,
        humidity=args.humidity,
        fuel_density=args.fuel_density,
    )
    np.savez_compressed(out, **data)
    print(f"Synthetic grid saved to {out}")
    print(f"  Shape: {args.size}×{args.size}")
    print(f"  Fire cells: {int(data['fire_mask'].sum())}")
    print(f"  UAV positions: {len(data['uav_positions'])}")


if __name__ == "__main__":
    main()
