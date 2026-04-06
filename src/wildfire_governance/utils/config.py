"""YAML configuration loader with OmegaConf and CLI-override support."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf
    _OMEGACONF = True
except ImportError:
    _OMEGACONF = False

try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False

_REQUIRED_KEYS = [
    "simulation",
    "decision",
    "verification",
    "blockchain",
    "governance",
    "logging",
    "reproducibility",
]


class ConfigValidationError(ValueError):
    """Raised when a required config key is missing."""


def load_config(path: str | Path, overrides: list[str] | None = None) -> Any:
    """Load and validate a YAML configuration file.

    Supports dot-notation CLI overrides, e.g.
    ``["simulation.n_uavs=40", "blockchain.n_validators=7"]``.

    Args:
        path: Path to the YAML config file.
        overrides: Optional list of ``key=value`` override strings.

    Returns:
        OmegaConf ``DictConfig`` if available, else a plain ``dict``.

    Raises:
        ConfigValidationError: If any required top-level key is missing.
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if _OMEGACONF:
        base_cfg = OmegaConf.load(path)
        # Merge base.yaml if the loaded file has a ``defaults`` key
        if hasattr(base_cfg, "defaults"):
            base_path = path.parent.parent / "base.yaml"
            if base_path.exists():
                base = OmegaConf.load(base_path)
                base_cfg = OmegaConf.merge(base, base_cfg)

        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            base_cfg = OmegaConf.merge(base_cfg, override_cfg)

        _validate(OmegaConf.to_container(base_cfg, resolve=True))
        return base_cfg
    else:
        if not _YAML:
            raise ImportError("Neither omegaconf nor pyyaml is installed.")
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        if overrides:
            for ov in overrides:
                key, _, value = ov.partition("=")
                parts = key.split(".")
                node = cfg
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = _cast(value)
        _validate(cfg)
        return cfg


def _validate(cfg: dict) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigValidationError(f"Config missing required keys: {missing}")


def _cast(value: str) -> Any:
    """Attempt to cast a CLI string to int, float, bool, or leave as str."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value
