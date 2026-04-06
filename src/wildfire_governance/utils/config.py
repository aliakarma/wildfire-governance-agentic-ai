"""YAML configuration loader."""
from __future__ import annotations
from pathlib import Path
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf; _OMEGACONF = True
except ImportError:
    _OMEGACONF = False

try:
    import yaml; _YAML = True
except ImportError:
    _YAML = False

_REQUIRED_KEYS = ["simulation","decision","verification","blockchain","governance","logging","reproducibility"]

class ConfigValidationError(ValueError): pass


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mapping values from override into base."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_config(path, overrides=None):
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Config not found: {path}")
    if _OMEGACONF:
        base_cfg = OmegaConf.load(path)
        if hasattr(base_cfg, "defaults"):
            base_path = path.parent.parent / "base.yaml"
            if base_path.exists():
                base_cfg = OmegaConf.merge(OmegaConf.load(base_path), base_cfg)
        if overrides:
            base_cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(overrides))
        _validate(OmegaConf.to_container(base_cfg, resolve=True)); return base_cfg
    if not _YAML: raise ImportError("Neither omegaconf nor pyyaml is installed.")
    with open(path) as fh: cfg = yaml.safe_load(fh)
    if isinstance(cfg, dict) and "defaults" in cfg:
        base_path = path.parent.parent / "base.yaml"
        if base_path.exists():
            with open(base_path) as fh:
                base_cfg = yaml.safe_load(fh)
            if isinstance(base_cfg, dict):
                cfg = _merge_dicts(base_cfg, cfg)
    _validate(cfg); return cfg

def _validate(cfg):
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing: raise ConfigValidationError(f"Config missing required keys: {missing}")
