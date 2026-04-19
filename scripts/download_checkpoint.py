#!/usr/bin/env python3
"""Download the pre-trained PPO checkpoint used for evaluation/reproduction."""

import urllib.request
from pathlib import Path

URL = "REPLACE_WITH_ACTUAL_LINK"
DEST = Path("src/wildfire_governance/rl/checkpoints/ppo_gomdp_best.pt")

DEST.parent.mkdir(parents=True, exist_ok=True)

if not DEST.exists():
    print("Downloading PPO checkpoint...")
    urllib.request.urlretrieve(URL, DEST)
    print("Done.")
else:
    print("Checkpoint already exists.")
