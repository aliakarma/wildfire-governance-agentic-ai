"""Global seed management and run-hash generation for reproducible experiments."""
from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set all random seeds deterministically.

    Args:
        seed: Integer seed value. Applied to Python random, NumPy, and (if
              available) PyTorch CPU and CUDA generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def generate_run_hash(config: Any) -> str:
    """Generate a unique, deterministic run identifier from a config object.

    Args:
        config: Any JSON-serialisable object (dict, OmegaConf DictConfig, etc.).

    Returns:
        String of the form ``YYYYMMDD_HHMMSS_<8-char-hash>``.
    """
    from datetime import datetime

    try:
        from omegaconf import OmegaConf
        config_dict = OmegaConf.to_container(config, resolve=True)
    except (ImportError, Exception):
        config_dict = dict(config) if not isinstance(config, dict) else config

    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    digest = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{digest}"


def make_results_dir(base: Path, run_hash: str) -> Path:
    """Create and return a timestamped results directory.

    Args:
        base: Root results directory (e.g., ``Path("results/runs")``).
        run_hash: Run identifier from :func:`generate_run_hash`.

    Returns:
        Path to the newly created directory.
    """
    results_dir = base / run_hash
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_rng(seed: int) -> np.random.Generator:
    """Return a seeded NumPy Generator (preferred over legacy RandomState).

    Args:
        seed: Integer seed.

    Returns:
        ``np.random.default_rng(seed)`` instance.
    """
    return np.random.default_rng(seed)
