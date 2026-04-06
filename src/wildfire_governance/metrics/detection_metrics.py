"""Detection performance metrics: Ld, Fp, Le2e.

Implements the metrics reported in Tables II–VI of the paper.
All metrics are computed per-episode and then aggregated across seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EpisodeMetrics:
    """Per-episode detection metrics.

    Attributes:
        detection_latency: Ld — steps from fire ignition to first detection.
        false_alert_rate: Fp — fraction of alerts that were not true fires.
        end_to_end_latency: Le2e — steps from ignition to public broadcast.
        n_alerts_broadcast: Total alerts broadcast this episode.
        n_true_alerts: Alerts corresponding to genuine fire events.
        n_false_alerts: Alerts not corresponding to genuine fire events.
        n_fires_detected: Genuine fire events detected (for recall).
        n_fires_total: Total genuine fire events (for recall).
        governance_compliant: Whether all broadcasts had valid governance certificates.
    """

    detection_latency: float = float("inf")
    false_alert_rate: float = 0.0
    end_to_end_latency: float = float("inf")
    n_alerts_broadcast: int = 0
    n_true_alerts: int = 0
    n_false_alerts: int = 0
    n_fires_detected: int = 0
    n_fires_total: int = 0
    governance_compliant: bool = True


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across N seeds (mean ± std).

    Attributes:
        ld_mean: Mean detection latency across seeds.
        ld_std: Standard deviation of detection latency.
        fp_mean: Mean false alert rate (percentage).
        fp_std: Standard deviation of false alert rate.
        le2e_mean: Mean end-to-end latency.
        le2e_std: Standard deviation of end-to-end latency.
        governance_compliance_pct: Fraction of episodes with zero violations × 100.
        n_seeds: Number of seeds aggregated.
    """

    ld_mean: float = 0.0
    ld_std: float = 0.0
    fp_mean: float = 0.0
    fp_std: float = 0.0
    le2e_mean: float = 0.0
    le2e_std: float = 0.0
    governance_compliance_pct: float = 100.0
    n_seeds: int = 0


class DetectionMetricsTracker:
    """Accumulates and computes detection metrics over a single episode.

    Args:
        ignition_timestep: Timestep at which fire ignition was placed.
        n_fires_total: Total number of genuine fire events in this episode.
    """

    def __init__(self, ignition_timestep: int = 0, n_fires_total: int = 1) -> None:
        self._ignition_timestep = ignition_timestep
        self._n_fires_total = n_fires_total
        self._first_detection_time: Optional[int] = None
        self._first_broadcast_time: Optional[int] = None
        self._alerts: list[dict] = []
        self._governance_compliant = True

    def record_detection(self, timestep: int) -> None:
        """Record a UAV detection event (stage-1 trigger).

        Args:
            timestep: Current simulation timestep.
        """
        if self._first_detection_time is None:
            self._first_detection_time = timestep

    def record_alert_broadcast(
        self,
        timestep: int,
        is_true_fire: bool,
        governance_cert: Optional[str] = None,
    ) -> None:
        """Record a public alert broadcast event.

        Args:
            timestep: Current simulation timestep.
            is_true_fire: Whether this alert corresponds to a real fire.
            governance_cert: Governance certificate hash; None = violation.
        """
        if self._first_broadcast_time is None:
            self._first_broadcast_time = timestep
        self._alerts.append(
            {
                "timestep": timestep,
                "is_true_fire": is_true_fire,
                "cert": governance_cert,
            }
        )
        if governance_cert is None:
            self._governance_compliant = False

    def compute(self) -> EpisodeMetrics:
        """Compute all metrics for this episode.

        Returns:
            EpisodeMetrics dataclass with all computed values.
        """
        n_total = len(self._alerts)
        n_true = sum(1 for a in self._alerts if a["is_true_fire"])
        n_false = n_total - n_true
        fp_rate = (n_false / n_total * 100.0) if n_total > 0 else 0.0

        ld = (
            float(self._first_detection_time - self._ignition_timestep)
            if self._first_detection_time is not None
            else float("inf")
        )
        le2e = (
            float(self._first_broadcast_time - self._ignition_timestep)
            if self._first_broadcast_time is not None
            else float("inf")
        )

        return EpisodeMetrics(
            detection_latency=max(0.0, ld),
            false_alert_rate=fp_rate,
            end_to_end_latency=max(0.0, le2e),
            n_alerts_broadcast=n_total,
            n_true_alerts=n_true,
            n_false_alerts=n_false,
            n_fires_detected=1 if self._first_detection_time is not None else 0,
            n_fires_total=self._n_fires_total,
            governance_compliant=self._governance_compliant,
        )


def aggregate_metrics(episode_metrics: List[EpisodeMetrics]) -> AggregatedMetrics:
    """Aggregate a list of per-episode metrics across seeds.

    Args:
        episode_metrics: List of EpisodeMetrics, one per seed.

    Returns:
        AggregatedMetrics with mean and std for each metric.
    """
    import numpy as np

    if not episode_metrics:
        return AggregatedMetrics()

    lds = [m.detection_latency for m in episode_metrics if m.detection_latency < float("inf")]
    fps = [m.false_alert_rate for m in episode_metrics]
    le2es = [m.end_to_end_latency for m in episode_metrics if m.end_to_end_latency < float("inf")]
    compliant = sum(1 for m in episode_metrics if m.governance_compliant)

    return AggregatedMetrics(
        ld_mean=float(np.mean(lds)) if lds else float("inf"),
        ld_std=float(np.std(lds)) if lds else 0.0,
        fp_mean=float(np.mean(fps)),
        fp_std=float(np.std(fps)),
        le2e_mean=float(np.mean(le2es)) if le2es else float("inf"),
        le2e_std=float(np.std(le2es)) if le2es else 0.0,
        governance_compliance_pct=100.0 * compliant / len(episode_metrics),
        n_seeds=len(episode_metrics),
    )
