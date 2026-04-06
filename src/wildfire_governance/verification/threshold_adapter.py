"""Online EMA-based threshold adaptation."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ThresholdHistory:
    tau1_history: list; tau2_history: list; precision_history: list; recall_history: list

class OnlineThresholdAdapter:
    def __init__(self, tau1_init=0.60, tau2_init=0.80, alpha_ema=0.10,
                 target_f1=0.85, tau1_min=0.40, tau2_max=0.95):
        self._tau1 = tau1_init; self._tau2 = tau2_init; self._alpha = alpha_ema
        self._target_f1 = target_f1; self._tau1_min = tau1_min; self._tau2_max = tau2_max
        self._history = ThresholdHistory([], [], [], [])

    def update(self, episode_precision, episode_recall):
        self._history.precision_history.append(episode_precision)
        self._history.recall_history.append(episode_recall)
        p = episode_precision; r = episode_recall
        f1 = (2*p*r/(p+r)) if (p+r) > 1e-9 else 0.0
        delta = -(self._target_f1 - f1) * 0.1
        target_tau1 = float(min(max(self._tau1+delta, self._tau1_min), self._tau2-0.05))
        target_tau2 = float(min(max(self._tau2+delta, target_tau1+0.05), self._tau2_max))
        self._tau1 = (1-self._alpha)*self._tau1 + self._alpha*target_tau1
        self._tau2 = (1-self._alpha)*self._tau2 + self._alpha*target_tau2
        if self._tau1 >= self._tau2: self._tau1 = self._tau2 - 0.05
        self._history.tau1_history.append(self._tau1); self._history.tau2_history.append(self._tau2)
        return self._tau1, self._tau2

    def get_thresholds(self): return self._tau1, self._tau2
    def reset(self, tau1=0.60, tau2=0.80):
        self._tau1 = tau1; self._tau2 = tau2; self._history = ThresholdHistory([], [], [], [])
