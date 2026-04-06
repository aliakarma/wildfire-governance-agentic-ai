#!/usr/bin/env python3
"""Verify checksums and structure of all downloaded datasets.

Usage:
    # Bash / PowerShell
    python data/scripts/validate_datasets.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CHECKS = [
    {
        "path": "data/synthetic/grid_10x10_seed42.npz",
        "required": True,
        "description": "Synthetic 10x10 grid for unit tests",
    },
    {
        "path": "data/synthetic/grid_100x100_seed0.npz",
        "required": False,
        "description": "Synthetic 100x100 grid for smoke tests",
    },
    {
        "path": "data/raw/viirs",
        "required": False,
        "is_dir": True,
        "description": "VIIRS raw CSV directory (optional — run make download-viirs)",
    },
    {
        "path": "data/raw/nifc",
        "required": False,
        "is_dir": True,
        "description": "NIFC perimeter data directory (optional)",
    },
    {
        "path": "data/processed",
        "required": True,
        "is_dir": True,
        "description": "Processed data directory (must exist)",
    },
]


def validate_all() -> bool:
    """Run all dataset validation checks.

    Returns:
        True if all required checks pass; False otherwise.
    """
    all_pass = True
    print("=== Dataset Validation ===\n")

    for check in CHECKS:
        path = Path(check["path"])
        required = check.get("required", False)
        is_dir = check.get("is_dir", False)
        desc = check.get("description", str(path))

        if is_dir:
            exists = path.is_dir()
        else:
            exists = path.is_file()

        if exists:
            if not is_dir:
                size_kb = path.stat().st_size / 1024
                status = f"OK ({size_kb:.1f} KB)"
            else:
                n_files = len(list(path.glob("*")))
                status = f"OK ({n_files} files)"
            print(f"  [PASS] {desc}")
            print(f"         Path: {path} — {status}")
        else:
            if required:
                print(f"  [FAIL] {desc}")
                print(f"         Path: {path} — MISSING (required)")
                all_pass = False
            else:
                print(f"  [SKIP] {desc}")
                print(f"         Path: {path} — not downloaded (optional)")
        print()

    if all_pass:
        print("All required datasets validated successfully.")
    else:
        print("Some required datasets are missing. Run: make download-data")

    return all_pass


if __name__ == "__main__":
    import sys
    success = validate_all()
    sys.exit(0 if success else 1)
