"""Adapter: convert real VIIRS/NIFC/ERA5 datasets to simulation grid format."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class RealWorldAdapter:
    """Convert geospatial real-world data into wildfire simulation grids.

    Supports VIIRS active fire hotspots, NIFC fire perimeter masks, and
    ERA5 meteorological fields. All outputs are aligned to the simulation
    grid at ``grid_size × grid_size`` resolution via bilinear interpolation.

    Args:
        grid_size: Target simulation grid side length (default 100).
    """

    def __init__(self, grid_size: int = 100) -> None:
        self.grid_size = grid_size

    def load_viirs_grid(self, path: Path) -> np.ndarray:
        """Load preprocessed VIIRS heat map from .npz file.

        The .npz must contain ``heat_map`` of shape (T, H, W) or (H, W).

        Args:
            path: Path to ``data/processed/viirs_grid_<region>.npz``.

        Returns:
            Float32 array of shape (T, grid_size, grid_size) or (grid_size, grid_size).

        Raises:
            FileNotFoundError: If the preprocessed file does not exist.
            KeyError: If the .npz lacks the ``heat_map`` key.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessed VIIRS file not found: {path}\n"
                "Run: make download-viirs && python data/scripts/preprocess_viirs.py"
            )
        data = np.load(path)
        if "heat_map" not in data:
            raise KeyError(f"'heat_map' array not found in {path}. Keys: {list(data)}")
        raw = data["heat_map"].astype(np.float32)
        if raw.ndim == 2:
            return self.align_to_grid(raw)
        return np.stack([self.align_to_grid(raw[t]) for t in range(len(raw))])

    def load_nifc_mask(self, path: Path) -> np.ndarray:
        """Load NIFC fire perimeter as binary ground-truth mask.

        Args:
            path: Path to ``data/processed/nifc_masks_<year>_<region>.npz``.

        Returns:
            Boolean array of shape (grid_size, grid_size).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NIFC mask not found: {path}")
        data = np.load(path)
        raw = data["fire_mask"].astype(bool)
        return self.align_to_grid(raw.astype(np.float32)) > 0.5

    def align_to_grid(self, data: np.ndarray) -> np.ndarray:
        """Bilinear interpolation to simulation grid resolution.

        Args:
            data: 2-D array of arbitrary spatial resolution.

        Returns:
            Float32 array of shape (grid_size, grid_size).
        """
        from scipy.ndimage import zoom  # type: ignore[import]

        h, w = data.shape
        zoom_r = self.grid_size / h
        zoom_c = self.grid_size / w
        resampled = zoom(data.astype(np.float32), (zoom_r, zoom_c), order=1)
        return resampled.astype(np.float32)

    def validate_alignment(
        self, simulated: np.ndarray, real: np.ndarray
    ) -> Dict[str, float]:
        """Compute spatial agreement between a simulated and real binary mask.

        Args:
            simulated: Simulated binary fire mask, shape (H, W).
            real: Real-world binary fire mask, shape (H, W).

        Returns:
            Dict with keys ``iou``, ``rmse``, ``spatial_correlation``.
        """
        sim = simulated.astype(float)
        gt = real.astype(float)
        intersection = (sim * gt).sum()
        union = ((sim + gt) > 0).sum()
        iou = float(intersection / union) if union > 0 else 0.0
        rmse = float(np.sqrt(np.mean((sim - gt) ** 2)))
        corr = float(np.corrcoef(sim.ravel(), gt.ravel())[0, 1]) if gt.std() > 0 else 0.0
        return {"iou": iou, "rmse": rmse, "spatial_correlation": corr}
