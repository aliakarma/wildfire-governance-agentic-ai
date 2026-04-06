"""Concrete UAV agent with battery management and step-wise movement."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from wildfire_governance.agents.base_agent import BaseUAVAgent, InsufficientBatteryError

_MIN_BATTERY_FRACTION = 0.05


class UAVAgent(BaseUAVAgent):
    """Autonomous UAV with position, battery, and thermal sensor.

    Moves one cell per step (velocity=1). Battery decreases by 1 per step.
    When battery fraction falls below 5%, the UAV must recharge before
    further movement.

    Args:
        agent_id: Unique identifier string.
        initial_position: Starting (row, col) on the simulation grid.
        battery_capacity: Total battery steps (default 500, ≈1.4h at 10s/step).
        detection_probability: Thermal sensor P(detect | fire) (default 0.85).
        grid_size: Simulation grid side length (for bounds checking).
    """

    def __init__(
        self,
        agent_id: str,
        initial_position: Tuple[int, int],
        battery_capacity: int = 500,
        detection_probability: float = 0.85,
        grid_size: int = 100,
    ) -> None:
        super().__init__(agent_id, initial_position, battery_capacity, detection_probability)
        self._grid_size = grid_size

    def move_to(
        self,
        target: Tuple[int, int],
        rng: np.random.Generator,
    ) -> float:
        """Move one step toward *target* using Manhattan-distance routing.

        Battery decreases by 1 per step. If already at target, no battery
        is consumed.

        Args:
            target: (row, col) destination cell.
            rng: Seeded NumPy Generator (reserved for future stochastic movement).

        Returns:
            Battery consumed this step (0 if already at target).

        Raises:
            InsufficientBatteryError: If battery fraction < 5%.
        """
        if self.battery_fraction < _MIN_BATTERY_FRACTION:
            raise InsufficientBatteryError(
                f"UAV {self.agent_id} battery too low ({self.battery_fraction:.1%}). "
                "Call agent.recharge() before moving."
            )

        row, col = self._position
        t_row, t_col = target

        if row == t_row and col == t_col:
            return 0.0

        # Move one step along the dominant axis
        d_row = t_row - row
        d_col = t_col - col
        if abs(d_row) >= abs(d_col):
            row += int(np.sign(d_row))
        else:
            col += int(np.sign(d_col))

        # Clamp to grid bounds
        row = int(np.clip(row, 0, self._grid_size - 1))
        col = int(np.clip(col, 0, self._grid_size - 1))
        self._position = (row, col)
        self._battery_level -= 1.0
        return 1.0

    def distance_to(self, target: Tuple[int, int]) -> int:
        """Manhattan distance from current position to *target*.

        Args:
            target: (row, col) target cell.

        Returns:
            Integer Manhattan distance.
        """
        return abs(self._position[0] - target[0]) + abs(self._position[1] - target[1])
