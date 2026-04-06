#!/usr/bin/env python3
"""Download NASA FIRMS VIIRS 375m active fire data for a specified region and date range.

Requires a free NASA FIRMS MAP_KEY:
    Register at: https://urs.earthdata.nasa.gov/
    Request key: https://firms.modaps.eosdis.nasa.gov/usfs/api/area/
    Set env var: export NASA_FIRMS_KEY="your_key_here"   (Bash)
                 $env:NASA_FIRMS_KEY="your_key_here"     (PowerShell)

Usage:
    # Bash
    python data/scripts/download_viirs.py --region california --start_date 2020-08-01 --end_date 2020-10-01

    # PowerShell
    python data/scripts/download_viirs.py --region california --start_date 2020-08-01 --end_date 2020-10-01
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

REGION_BBOXES = {
    "california":    {"min_lat": 32.5, "min_lon": -124.4, "max_lat": 42.0, "max_lon": -114.1},
    "mediterranean": {"min_lat": 35.0, "min_lon": -9.0,   "max_lat": 47.0, "max_lon": 37.0},
    "australia":     {"min_lat": -39.0, "min_lon": 114.0,  "max_lat": -10.0, "max_lon": 154.0},
    "canada":        {"min_lat": 49.0, "min_lon": -141.0,  "max_lat": 60.0,  "max_lon": -52.0},
}

RAW_DIR = Path("data/raw/viirs")
PROCESSED_DIR = Path("data/processed")


def download_viirs(
    region: str,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> Path:
    """Download VIIRS active fire CSV for the specified region and date range.

    Args:
        region: Named region (one of: california, mediterranean, australia, canada).
                Custom bbox can be passed by setting region to "custom" and providing
                min_lat, min_lon, max_lat, max_lon env vars.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        api_key: NASA FIRMS MAP_KEY. If None, reads from NASA_FIRMS_KEY env var.

    Returns:
        Path to the downloaded CSV file.
    """
    import requests

    api_key = api_key or os.environ.get("NASA_FIRMS_KEY", "")
    if not api_key:
        print(
            "\nNASA_FIRMS_KEY not set. To download real VIIRS data:\n"
            "  1. Register at: https://urs.earthdata.nasa.gov/\n"
            "  2. Request a MAP_KEY at: https://firms.modaps.eosdis.nasa.gov/usfs/api/area/\n"
            "  3. Bash:       export NASA_FIRMS_KEY='your_key'\n"
            "     PowerShell: $env:NASA_FIRMS_KEY='your_key'\n\n"
            "Synthetic fallback data is available in data/synthetic/ for testing.\n"
        )
        return _create_synthetic_fallback(region)

    if region not in REGION_BBOXES:
        raise ValueError(f"Unknown region '{region}'. Available: {list(REGION_BBOXES)}")

    bbox = REGION_BBOXES[region]
    area_str = f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}"
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv/"
        f"{api_key}/VIIRS_SNPP_NRT/{area_str}/1/{start_date}"
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"viirs_{region}_{start_date}_{end_date}.csv"

    print(f"Downloading VIIRS data for {region} ({start_date} to {end_date})...")
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            out_path.write_bytes(response.content)
            checksum = hashlib.sha256(response.content).hexdigest()[:16]
            print(f"  Saved to {out_path} (SHA256: {checksum}...)")
            return out_path
        except Exception as exc:
            print(f"  Attempt {attempt + 1}/3 failed: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    print("Download failed. Using synthetic fallback.")
    return _create_synthetic_fallback(region)


def _create_synthetic_fallback(region: str) -> Path:
    """Create a minimal synthetic VIIRS-format CSV for testing without API key."""
    import csv

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"viirs_{region}_synthetic.csv"
    if out_path.exists():
        print(f"  Synthetic fallback already exists: {out_path}")
        return out_path

    import numpy as np
    rng = np.random.default_rng(42)
    bbox = REGION_BBOXES.get(region, REGION_BBOXES["california"])
    rows = []
    for _ in range(200):
        lat = rng.uniform(bbox["min_lat"], bbox["max_lat"])
        lon = rng.uniform(bbox["min_lon"], bbox["max_lon"])
        rows.append({
            "latitude": round(float(lat), 5),
            "longitude": round(float(lon), 5),
            "bright_ti4": round(float(rng.uniform(300, 500)), 1),
            "bright_ti5": round(float(rng.uniform(290, 490)), 1),
            "frp": round(float(rng.uniform(1, 100)), 1),
            "confidence": "nominal",
            "acq_date": "2020-08-15",
            "acq_time": "0130",
            "satellite": "N",
            "instrument": "VIIRS",
        })

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Synthetic VIIRS fallback written to {out_path}")
    return out_path


def preprocess_viirs(csv_path: Path, grid_size: int = 100) -> Path:
    """Convert VIIRS CSV to a numpy .npz simulation grid.

    Args:
        csv_path: Path to the downloaded or synthetic VIIRS CSV.
        grid_size: Target grid side length.

    Returns:
        Path to the output .npz file in data/processed/.
    """
    import numpy as np
    import pandas as pd

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    out_path = PROCESSED_DIR / f"viirs_grid_{stem}.npz"

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  CSV is empty, creating zero grid: {out_path}")
        heat = np.zeros((1, grid_size, grid_size), dtype=np.float32)
        np.savez_compressed(out_path, heat_map=heat)
        return out_path

    lat = df["latitude"].values
    lon = df["longitude"].values
    frp = df["frp"].values if "frp" in df.columns else np.ones(len(df))

    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    lat_range = max(lat_max - lat_min, 1e-6)
    lon_range = max(lon_max - lon_min, 1e-6)

    heat = np.zeros((grid_size, grid_size), dtype=np.float32)
    for i in range(len(lat)):
        row = int((lat[i] - lat_min) / lat_range * (grid_size - 1))
        col = int((lon[i] - lon_min) / lon_range * (grid_size - 1))
        row = max(0, min(row, grid_size - 1))
        col = max(0, min(col, grid_size - 1))
        heat[row, col] = min(1.0, heat[row, col] + float(frp[i]) / 100.0)

    fire_mask = (heat > 0.1).astype(np.float32)
    np.savez_compressed(out_path, heat_map=heat[np.newaxis], fire_mask=fire_mask[np.newaxis])
    print(f"  Preprocessed grid saved to {out_path} (shape: {heat.shape})")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NASA FIRMS VIIRS data")
    parser.add_argument("--region", default="california",
                        choices=list(REGION_BBOXES), help="Named region")
    parser.add_argument("--start_date", default="2020-08-01", help="YYYY-MM-DD")
    parser.add_argument("--end_date", default="2020-10-01", help="YYYY-MM-DD")
    parser.add_argument("--api_key", default=None, help="NASA FIRMS MAP_KEY")
    parser.add_argument("--preprocess", action="store_true",
                        help="Also preprocess to simulation grid")
    args = parser.parse_args()

    csv_path = download_viirs(args.region, args.start_date, args.end_date, args.api_key)
    if args.preprocess:
        preprocess_viirs(csv_path)
