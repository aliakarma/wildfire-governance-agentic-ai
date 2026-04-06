"""Detection performance metrics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class EpisodeMetrics:
    detection_latency: float = float("inf"); false_alert_rate: float = 0.0
    end_to_end_latency: float = float("inf"); n_alerts_broadcast: int = 0
    n_true_alerts: int = 0; n_false_alerts: int = 0
    n_fires_detected: int = 0; n_fires_total: int = 0; governance_compliant: bool = True

@dataclass
class AggregatedMetrics:
    ld_mean: float = 0.0; ld_std: float = 0.0; fp_mean: float = 0.0; fp_std: float = 0.0
    le2e_mean: float = 0.0; le2e_std: float = 0.0
    governance_compliance_pct: float = 100.0; n_seeds: int = 0

def aggregate_metrics(episode_metrics):
    if not episode_metrics: return AggregatedMetrics()
    lds = [m.detection_latency for m in episode_metrics if m.detection_latency < float("inf")]
    fps = [m.false_alert_rate for m in episode_metrics]
    compliant = sum(1 for m in episode_metrics if m.governance_compliant)
    return AggregatedMetrics(
        ld_mean=float(np.mean(lds)) if lds else float("inf"),
        ld_std=float(np.std(lds)) if lds else 0.0,
        fp_mean=float(np.mean(fps)), fp_std=float(np.std(fps)),
        governance_compliance_pct=100.0*compliant/len(episode_metrics),
        n_seeds=len(episode_metrics))
