"""Concrete UAV agent with battery management and step-wise movement."""
from __future__ import annotations
from typing import Tuple
import numpy as np
from wildfire_governance.simulation.sensor_models import SensorReading, ThermalUAVSensor

_MIN_BATTERY_FRACTION = 0.05

class InsufficientBatteryError(RuntimeError): pass

class UAVAgent:
    def __init__(self, agent_id, initial_position, battery_capacity=500,
                 detection_probability=0.85, grid_size=100):
        self.agent_id = agent_id; self._position = initial_position
        self._battery_capacity = battery_capacity; self._battery_level = float(battery_capacity)
        self._sensor = ThermalUAVSensor(detection_probability=detection_probability)
        self._grid_size = grid_size; self._patrol_sector = None; self._status = "idle"

    @property
    def position(self): return self._position
    @property
    def battery_level(self): return self._battery_level
    @property
    def battery_fraction(self): return self._battery_level / self._battery_capacity
    @property
    def patrol_sector(self): return self._patrol_sector
    @property
    def status(self): return self._status

    def move_to(self, target, rng):
        if self.battery_fraction < _MIN_BATTERY_FRACTION:
            raise InsufficientBatteryError(f"UAV {self.agent_id} battery too low")
        row, col = self._position; t_row, t_col = target
        if row == t_row and col == t_col: return 0.0
        d_row = t_row - row; d_col = t_col - col
        if abs(d_row) >= abs(d_col): row += int(np.sign(d_row))
        else: col += int(np.sign(d_col))
        row = int(np.clip(row, 0, self._grid_size-1)); col = int(np.clip(col, 0, self._grid_size-1))
        self._position = (row, col); self._battery_level -= 1.0; return 1.0

    def observe(self, grid, rng): return self._sensor.observe(grid, self._position, rng)
    def assign_sector(self, sector_id): self._patrol_sector = sector_id; self._status = "patrolling"
    def recharge(self): self._battery_level = float(self._battery_capacity); self._status = "idle"
    def distance_to(self, target): return abs(self._position[0]-target[0]) + abs(self._position[1]-target[1])
