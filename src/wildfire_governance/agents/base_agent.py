"""Abstract base class for UAV agents."""
from __future__ import annotations

import abc
from typing import Optional, Tuple

import numpy as np

from wildfire_governance.simulation.sensor_models import SensorReading, ThermalUAVSensor


class InsufficientBatteryError(RuntimeError):
    """Raised when a UAV cannot complete a move due to insufficient battery."""


class BaseUAVAgent(abc.ABC):
    """Abstract base for all UAV agents.

    Args:
        agent_id: Unique string identifier.
        initial_position: Starting (row, col) grid position.
        battery_capacity: Maximum battery in steps (default 500).
        detection_probability: Thermal sensor detection probability.
    """

    def __init__(
        self,
        agent_id: str,
        initial_position: Tuple[int, int],
        battery_capacity: int = 500,
        detection_probability: float = 0.85,
    ) -> None:
        self.agent_id = agent_id
        self._position: Tuple[int, int] = initial_position
        self._battery_capacity = battery_capacity
        self._battery_level: float = float(battery_capacity)
        self._sensor = ThermalUAVSensor(detection_probability=detection_probability)
        self._patrol_sector: Optional[int] = None
        self._status: str = "idle"

    @property
    def position(self) -> Tuple[int, int]:
        """Current (row, col) grid position."""
        return self._position

    @property
    def battery_level(self) -> float:
        """Remaining battery in steps."""
        return self._battery_level

    @property
    def battery_fraction(self) -> float:
        """Battery as a fraction [0, 1] of capacity."""
        return self._battery_level / self._battery_capacity

    @property
    def patrol_sector(self) -> Optional[int]:
        """Assigned patrol sector ID, or None if unassigned."""
        return self._patrol_sector

    @property
    def status(self) -> str:
        """Current agent status string."""
        return self._status

    @abc.abstractmethod
    def move_to(
        self,
        target: Tuple[int, int],
        rng: np.random.Generator,
    ) -> float:
        """Move toward *target*; return battery consumed.

        Raises:
            InsufficientBatteryError: If battery < minimum safe level.
        """

    def observe(
        self,
        grid: np.ndarray,
        rng: np.random.Generator,
    ) -> SensorReading:
        """Take a thermal observation at the current position.

        Args:
            grid: Current heat map (H, W).
            rng: Seeded NumPy Generator.

        Returns:
            SensorReading at the current position.
        """
        return self._sensor.observe(grid, self._position, rng)

    def assign_sector(self, sector_id: int) -> None:
        """Assign this UAV to a patrol sector.

        Args:
            sector_id: Integer sector identifier.
        """
        self._patrol_sector = sector_id
        self._status = "patrolling"

    def recharge(self) -> None:
        """Fully recharge the UAV battery (called when returning to base)."""
        self._battery_level = float(self._battery_capacity)
        self._status = "idle"

    def get_remaining_range(self) -> int:
        """Number of move steps possible at current battery level."""
        return int(self._battery_level)
