"""Stage-1 cross-modal confidence fusion (Eq. 5)."""
from __future__ import annotations
import numpy as np

class CrossModalFusion:
    def __init__(self, w_h=0.65, w_w=0.35):
        if abs(w_h + w_w - 1.0) > 1e-6:
            raise ValueError(f"Fusion weights must sum to 1.0; got {w_h+w_w}")
        self.w_h = w_h; self.w_w = w_w

    def compute_stage1_confidence(self, heat_anomaly_index, weather_index):
        for name, val in [("heat_anomaly_index", heat_anomaly_index), ("weather_index", weather_index)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0,1]; got {val:.4f}")
        return float(self.w_h * heat_anomaly_index + self.w_w * weather_index)

    def batch_compute(self, heat_map, weather_map):
        return (self.w_h * np.clip(heat_map,0,1) + self.w_w * np.clip(weather_map,0,1)).astype(np.float32)
