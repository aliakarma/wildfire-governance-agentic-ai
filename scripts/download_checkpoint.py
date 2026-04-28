#!/usr/bin/env python3
"""Download the pre-trained PPO checkpoint used for evaluation/reproduction."""

import hashlib
import os
import urllib.request
from pathlib import Path

URL = os.getenv("WILDFIRE_PPO_CHECKPOINT_URL", "REPLACE_WITH_ACTUAL_LINK")
EXPECTED_SHA256 = os.getenv("WILDFIRE_PPO_CHECKPOINT_SHA256", "REPLACE_WITH_ACTUAL_SHA256")
DEST = Path("src/wildfire_governance/rl/checkpoints/ppo_gomdp_best.pt")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checksum(path: Path) -> None:
    if EXPECTED_SHA256 == "REPLACE_WITH_ACTUAL_SHA256":
        print("WARNING: Checksum placeholder is not set. Skipping checksum validation.")
        return

    actual = sha256_file(path)
    if actual.lower() != EXPECTED_SHA256.lower():
        raise RuntimeError(
            "Checkpoint checksum mismatch. "
            f"expected={EXPECTED_SHA256} actual={actual}"
        )


DEST.parent.mkdir(parents=True, exist_ok=True)

if URL == "REPLACE_WITH_ACTUAL_LINK":
    raise RuntimeError(
        "Checkpoint URL is not configured. "
        "Set WILDFIRE_PPO_CHECKPOINT_URL and optionally "
        "WILDFIRE_PPO_CHECKPOINT_SHA256 before running this script."
    )

if not DEST.exists():
    print("Downloading PPO checkpoint...")
    urllib.request.urlretrieve(URL, DEST)
    validate_checksum(DEST)
    print("Done.")
else:
    print("Checkpoint already exists.")
    validate_checksum(DEST)
