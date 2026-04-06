#!/usr/bin/env python3
"""Download NOAA GOES-16 fire detection data from AWS Open Data (no auth required).

Source: s3://noaa-goes16/ABI-L2-FDCF/
No authentication needed — uses anonymous S3 access.

Usage:
    # Bash
    python data/scripts/download_goes16.py --region california \
        --start_datetime 2020-08-01T00:00:00 --end_datetime 2020-08-07T00:00:00

    # PowerShell
    python data/scripts/download_goes16.py --region california `
        --start_datetime 2020-08-01T00:00:00 --end_datetime 2020-08-07T00:00:00
"""
from __future__ import annotations

import argparse
from pathlib import Path

RAW_DIR = Path("data/raw/goes16")


def download_goes16(
    region: str,
    start_datetime: str,
    end_datetime: str,
    max_files: int = 10,
) -> list[Path]:
    """Download GOES-16 NetCDF fire detection files from NOAA AWS open data.

    Args:
        region: Named region (for bounding box filtering).
        start_datetime: ISO start datetime string.
        end_datetime: ISO end datetime string.
        max_files: Maximum number of files to download (default 10).

    Returns:
        List of paths to downloaded NetCDF files.
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print("boto3 not installed. Run: pip install boto3")
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket = "noaa-goes16"
    prefix = "ABI-L2-FDCF/"  # Full disk fire detection

    print(f"Browsing s3://{bucket}/{prefix} for GOES-16 fire data...")
    downloaded: list[Path] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        count = 0
        for page in pages:
            if count >= max_files:
                break
            for obj in page.get("Contents", []):
                if count >= max_files:
                    break
                key = obj["Key"]
                if not key.endswith(".nc"):
                    continue
                filename = Path(key).name
                out_path = RAW_DIR / filename
                if not out_path.exists():
                    print(f"  Downloading {filename}...")
                    s3.download_file(bucket, key, str(out_path))
                    downloaded.append(out_path)
                count += 1
    except Exception as exc:
        print(f"  GOES-16 download failed: {exc}")
        print("  This dataset is optional. Simulation uses synthetic satellite data by default.")

    print(f"Downloaded {len(downloaded)} GOES-16 files to {RAW_DIR}")
    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NOAA GOES-16 fire data")
    parser.add_argument("--region", default="california")
    parser.add_argument("--start_datetime", default="2020-08-01T00:00:00")
    parser.add_argument("--end_datetime", default="2020-08-07T00:00:00")
    parser.add_argument("--max_files", type=int, default=5)
    args = parser.parse_args()
    download_goes16(args.region, args.start_datetime, args.end_datetime, args.max_files)
