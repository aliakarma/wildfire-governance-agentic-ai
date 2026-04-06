"""Sensor spoofing attack simulator."""
from __future__ import annotations
import numpy as np

class SensorSpoofer:
    def __init__(self, p_spoof=0.10, tau1=0.60, rng=None):
        if not 0.0 <= p_spoof <= 1.0: raise ValueError(f"p_spoof must be in [0,1]")
        self.p_spoof = p_spoof; self.tau1 = tau1
        self._rng = rng or np.random.default_rng(42)
        self._n_spoofed = 0; self._n_total = 0

    def inject(self, heat_map, fire_mask):
        spoofed = heat_map.copy(); non_fire = (fire_mask < 0.5)
        candidates = non_fire & (self._rng.random(heat_map.shape) < self.p_spoof)
        n = int(candidates.sum()); self._n_total += n
        if n > 0:
            spoofed[candidates] = self._rng.uniform(self.tau1, 1.0, size=n).astype(np.float32)
            self._n_spoofed += n
        return spoofed

    @property
    def spoof_rate(self): return self._n_spoofed / max(1, self._n_total)
    def reset(self): self._n_spoofed = 0; self._n_total = 0
