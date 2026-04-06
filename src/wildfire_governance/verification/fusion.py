"""Stage-1 cross-modal confidence fusion — Eq. (5) from the paper.

Conf^(1)_t = w_H * hat_H_t + w_W * hat_W_t

where hat_H_t is the standardised heat anomaly index and hat_W_t is the
standardised adverse weather index derived from meteorological conditions.
Both inputs are normalised to [0, 1] before fusion.

Fusion weights w_H=0.65, w_W=0.35 were calibrated via grid search on
held-out simulation validation seeds (see paper Section VI-A).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class CrossModalFusion:
    """Weighted linear cross-modal fusion for stage-1 anomaly detection.

    Args:
        w_h: Weight for the heat anomaly index (default 0.65).
        w_w: Weight for the weather anomaly index (default 0.35).

    Raises:
        ValueError: If ``w_h + w_w`` is not approximately 1.0.
    """

    def __init__(self, w_h: float = 0.65, w_w: float = 0.35) -> None:
        if abs(w_h + w_w - 1.0) > 1e-6:
            raise ValueError(
                f"Fusion weights must sum to 1.0; got w_h={w_h}, w_w={w_w}, sum={w_h+w_w}"
            )
        self.w_h = w_h
        self.w_w = w_w

    def compute_stage1_confidence(
        self,
        heat_anomaly_index: float,
        weather_index: float,
    ) -> float:
        """Compute Conf^(1)_t via weighted linear fusion (Eq. 5).

        Args:
            heat_anomaly_index: Normalised heat anomaly hat_H_t in [0, 1].
                Typical derivation: (observed_heat - baseline_mean) / baseline_std,
                then clipped and normalised to [0, 1].
            weather_index: Normalised adverse weather index hat_W_t in [0, 1].
                High values indicate dry, windy, fire-prone conditions.

        Returns:
            Stage-1 confidence score in [0, 1].

        Raises:
            ValueError: If either input is outside [0, 1].
        """
        for name, val in [("heat_anomaly_index", heat_anomaly_index), ("weather_index", weather_index)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]; got {val:.4f}")
        return float(self.w_h * heat_anomaly_index + self.w_w * weather_index)

    def batch_compute(
        self,
        heat_map: np.ndarray,
        weather_map: np.ndarray,
    ) -> np.ndarray:
        """Vectorised confidence computation over a full grid.

        Args:
            heat_map: Normalised heat anomaly map, shape (H, W), values in [0, 1].
            weather_map: Normalised weather index map, shape (H, W), values in [0, 1].

        Returns:
            Stage-1 confidence map of shape (H, W).
        """
        heat_map = np.clip(heat_map, 0.0, 1.0)
        weather_map = np.clip(weather_map, 0.0, 1.0)
        return (self.w_h * heat_map + self.w_w * weather_map).astype(np.float32)

    def calibrate_weights(
        self,
        validation_data: pd.DataFrame,
        metric: str = "f1",
    ) -> tuple[float, float]:
        """Grid search for optimal fusion weights on validation data.

        Args:
            validation_data: DataFrame with columns ``heat``, ``weather``, ``label``
                (1 = fire, 0 = no fire) and ``tau`` threshold for detection.
            metric: Optimisation metric — ``"f1"``, ``"precision"``, or ``"recall"``.

        Returns:
            Tuple (w_h, w_w) with best weights found.
        """
        best_score = -1.0
        best_wh = self.w_h
        best_ww = self.w_w

        for w_h_candidate in np.arange(0.1, 1.0, 0.05):
            w_w_candidate = 1.0 - w_h_candidate
            fusion = CrossModalFusion(w_h=float(w_h_candidate), w_w=float(w_w_candidate))
            preds = [
                int(fusion.compute_stage1_confidence(row["heat"], row["weather"]) > 0.5)
                for _, row in validation_data.iterrows()
            ]
            labels = validation_data["label"].tolist()
            score = _compute_metric(preds, labels, metric)
            if score > best_score:
                best_score = score
                best_wh = float(w_h_candidate)
                best_ww = float(w_w_candidate)

        self.w_h = best_wh
        self.w_w = best_ww
        return best_wh, best_ww


def _compute_metric(preds: list, labels: list, metric: str) -> float:
    """Compute precision, recall, or F1 from binary predictions and labels."""
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}.get(metric, f1)
