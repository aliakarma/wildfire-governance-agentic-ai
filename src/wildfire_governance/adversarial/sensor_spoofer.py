"""Sensor spoofing attack simulator.

Injects fabricated heat readings into the sensor stream to stress-test
the false-alert suppression pipeline. Corresponds to the spoofing rows
in Table V (p_spoof ∈ {0.05, 0.10, 0.20}).

Attack model: with probability p_spoof, each non-fire grid cell has its
heat reading replaced by a spoofed value drawn from Uniform(tau_1, 1.0),
i.e., above the stage-1 detection threshold, maximising false-positive rate.
"""
from __future__ import annotations

import numpy as np


class SensorSpoofer:
    """Injects fabricated heat readings to simulate adversarial sensor spoofing.

    With probability ``p_spoof``, replaces non-fire cell heat readings with
    values above the stage-1 detection threshold (tau_1=0.60) to maximise
    false-positive alerts.

    Args:
        p_spoof: Per-cell spoofing probability per timestep.
                 Paper values tested: {0.0, 0.05, 0.10, 0.20}.
        tau1: Stage-1 detection threshold (spoof values are drawn above this).
        rng: Seeded NumPy Generator.
    """

    def __init__(
        self,
        p_spoof: float = 0.10,
        tau1: float = 0.60,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 0.0 <= p_spoof <= 1.0:
            raise ValueError(f"p_spoof must be in [0,1]; got {p_spoof}")
        self.p_spoof = p_spoof
        self.tau1 = tau1
        self._rng = rng or np.random.default_rng(42)
        self._n_spoofed: int = 0
        self._n_total: int = 0

    def inject(
        self,
        heat_map: np.ndarray,
        fire_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply spoofing to non-fire cells in *heat_map*.

        Args:
            heat_map: Current heat map, shape (H, W), values in [0, 1].
            fire_mask: Binary ground-truth fire mask, shape (H, W).

        Returns:
            Spoofed heat map of shape (H, W). Original array is NOT modified.
        """
        spoofed = heat_map.copy()
        non_fire = (fire_mask < 0.5)
        candidates = non_fire & (self._rng.random(heat_map.shape) < self.p_spoof)
        n_candidates = int(candidates.sum())
        self._n_total += n_candidates

        if n_candidates > 0:
            # Spoof values: Uniform(tau1, 1.0)
            spoof_values = self._rng.uniform(self.tau1, 1.0, size=n_candidates).astype(np.float32)
            spoofed[candidates] = spoof_values
            self._n_spoofed += n_candidates

        return spoofed

    @property
    def spoof_rate(self) -> float:
        """Actual fraction of cells spoofed so far."""
        return self._n_spoofed / max(1, self._n_total)

    def reset(self) -> None:
        """Reset per-episode counters."""
        self._n_spoofed = 0
        self._n_total = 0
