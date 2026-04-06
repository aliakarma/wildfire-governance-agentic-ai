"""Global seed management and run-hash generation."""
from __future__ import annotations
import hashlib, json, os, random
from pathlib import Path
from typing import Any
import numpy as np

def set_global_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch; torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    except ImportError: pass

def generate_run_hash(config: Any) -> str:
    from datetime import datetime
    try:
        from omegaconf import OmegaConf; config_dict = OmegaConf.to_container(config, resolve=True)
    except Exception: config_dict = dict(config) if not isinstance(config, dict) else config
    digest = hashlib.sha256(json.dumps(config_dict, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest}"

def make_results_dir(base: Path, run_hash: str) -> Path:
    d = base / run_hash; d.mkdir(parents=True, exist_ok=True); return d

def get_rng(seed: int): return np.random.default_rng(seed)
