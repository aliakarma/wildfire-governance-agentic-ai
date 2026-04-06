"""Adapter: convert real VIIRS/NIFC datasets to simulation grid format."""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np

class RealWorldAdapter:
    def __init__(self, grid_size=100): self.grid_size = grid_size

    def load_viirs_grid(self, path):
        path = Path(path)
        if not path.exists(): raise FileNotFoundError(f"VIIRS file not found: {path}")
        data = np.load(path)
        if "heat_map" not in data: raise KeyError(f"heat_map not in {path}")
        raw = data["heat_map"].astype(np.float32)
        if raw.ndim == 2: return self.align_to_grid(raw)
        return np.stack([self.align_to_grid(raw[t]) for t in range(len(raw))])

    def load_nifc_mask(self, path):
        path = Path(path)
        if not path.exists(): raise FileNotFoundError(f"NIFC mask not found: {path}")
        data = np.load(path); raw = data["fire_mask"].astype(bool)
        return self.align_to_grid(raw.astype(np.float32)) > 0.5

    def align_to_grid(self, data):
        from scipy.ndimage import zoom
        h, w = data.shape
        return zoom(data.astype(np.float32), (self.grid_size/h, self.grid_size/w), order=1).astype(np.float32)

    def validate_alignment(self, simulated, real):
        sim = simulated.astype(float); gt = real.astype(float)
        intersection = (sim*gt).sum(); union = ((sim+gt)>0).sum()
        iou = float(intersection/union) if union > 0 else 0.0
        rmse = float(np.sqrt(np.mean((sim-gt)**2)))
        corr = float(np.corrcoef(sim.ravel(), gt.ravel())[0,1]) if gt.std() > 0 else 0.0
        return {"iou": iou, "rmse": rmse, "spatial_correlation": corr}
