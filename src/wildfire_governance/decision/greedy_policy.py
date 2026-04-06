"""Risk-weighted greedy UAV coordination policy."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

class RiskWeightedGreedyPolicy:
    def __init__(self, n_sectors=25, grid_size=100):
        self._n_sectors = n_sectors; self._grid_size = grid_size
        self._sector_cells = self._build_sector_cells()

    def _build_sector_cells(self):
        gs = self._grid_size; side = int(np.ceil(np.sqrt(self._n_sectors)))
        cell_size = gs // side; sectors = {}; sid = 0
        for sr in range(side):
            for sc in range(side):
                if sid >= self._n_sectors: break
                r0, r1 = sr*cell_size, min((sr+1)*cell_size, gs)
                c0, c1 = sc*cell_size, min((sc+1)*cell_size, gs)
                sectors[sid] = [(r,c) for r in range(r0,r1) for c in range(c0,c1)]
                sid += 1
        return sectors

    def sector_centroid(self, sector_id):
        cells = self._sector_cells.get(sector_id, [])
        if not cells: return (0, 0)
        return (int(np.mean([c[0] for c in cells])), int(np.mean([c[1] for c in cells])))

    def compute_sector_risk(self, risk_map, sector_id):
        cells = self._sector_cells.get(sector_id, [])
        if not cells: return 0.0
        return float(np.mean([float(risk_map[r,c]) for r,c in cells]))

    def select_actions(self, risk_map, uav_positions, battery_levels=None):
        n_uavs = len(uav_positions)
        if battery_levels is None: battery_levels = [1.0] * n_uavs
        scores = {s: self.compute_sector_risk(risk_map, s) * (
            1.0 - sum(1 for p in uav_positions if p in self._sector_cells.get(s, []))/max(1, len(self._sector_cells.get(s,[1])))
        ) for s in self._sector_cells}
        sorted_sectors = sorted(scores, key=lambda s: scores[s], reverse=True)
        top_centroid = self.sector_centroid(sorted_sectors[0]) if sorted_sectors else (0,0)
        uav_order = sorted(range(n_uavs), key=lambda i: (uav_positions[i][0]-top_centroid[0])**2+(uav_positions[i][1]-top_centroid[1])**2)
        allocation = {}; assigned = set()
        for idx in uav_order:
            if battery_levels[idx] < 0.10: continue
            for s in sorted_sectors:
                if s not in assigned: allocation[idx] = s; assigned.add(s); break
        return allocation
