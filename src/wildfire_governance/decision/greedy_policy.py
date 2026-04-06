"""Risk-weighted greedy UAV coordination policy (Algorithm 1 in paper).

O(NZ) approximation to the CPOMDP allocation objective.
Provides a tractable, training-free baseline within the GOMDP environment.
Any policy — including this one — satisfies Theorem 1 (Policy-Agnostic Safety)
when operating inside the GovernanceInvariantMDP environment.

See ``rl/ppo_agent.py`` for the deep RL alternative that reduces Ld by 17.5%.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


AllocationDict = Dict[int, int]  # uav_id -> sector_id


def _compute_coverage(
    uav_positions: List[Tuple[int, int]],
    sector_cells: List[Tuple[int, int]],
) -> float:
    """Fraction of sector cells NOT currently patrolled by any UAV.

    Coverage(z) ∈ [0, 1]; higher means more of the sector is unobserved.
    Used in the allocation objective: max sum_i E[ΔR^{p_i}_t].

    Args:
        uav_positions: Current (row, col) positions of all UAVs.
        sector_cells: List of (row, col) cells belonging to the sector.

    Returns:
        Float in [0, 1]; 1.0 means no UAV is currently in this sector.
    """
    if not sector_cells:
        return 0.0
    uav_set = set(uav_positions)
    covered = sum(1 for cell in sector_cells if cell in uav_set)
    return 1.0 - covered / len(sector_cells)


class RiskWeightedGreedyPolicy:
    """Greedy O(NZ) approximation to the CPOMDP allocation objective.

    Implements Algorithm 1 from the paper: sorts sectors by descending risk,
    assigns UAVs greedily to the highest-unassigned sector within energy budget.

    This heuristic approximates ``max_{A_t} sum_i E[ΔR^{p_i}_t]`` but does
    NOT solve the CPOMDP optimally. The optimality gap is uncharacterised;
    see ``decision/cpomdp.py`` for the theoretical target.

    Args:
        n_sectors: Number of patrol sectors (Z). Sectors are square tiles.
        grid_size: Environment grid side length.
    """

    def __init__(self, n_sectors: int = 25, grid_size: int = 100) -> None:
        self._n_sectors = n_sectors
        self._grid_size = grid_size
        self._sector_cells = self._build_sector_cells()

    def _build_sector_cells(self) -> Dict[int, List[Tuple[int, int]]]:
        """Partition the grid into n_sectors square tiles."""
        gs = self._grid_size
        n = self._n_sectors
        side = int(np.ceil(np.sqrt(n)))
        cell_size = gs // side
        sectors: Dict[int, List[Tuple[int, int]]] = {}
        sector_id = 0
        for sr in range(side):
            for sc in range(side):
                if sector_id >= n:
                    break
                r0, r1 = sr * cell_size, min((sr + 1) * cell_size, gs)
                c0, c1 = sc * cell_size, min((sc + 1) * cell_size, gs)
                cells = [
                    (r, c)
                    for r in range(r0, r1)
                    for c in range(c0, c1)
                ]
                sectors[sector_id] = cells
                sector_id += 1
        return sectors

    def sector_centroid(self, sector_id: int) -> Tuple[int, int]:
        """Return the (row, col) centroid of a sector.

        Args:
            sector_id: Integer sector identifier.

        Returns:
            (row, col) centroid position.
        """
        cells = self._sector_cells.get(sector_id, [])
        if not cells:
            return (0, 0)
        rows = [c[0] for c in cells]
        cols = [c[1] for c in cells]
        return (int(np.mean(rows)), int(np.mean(cols)))

    def compute_sector_risk(
        self, risk_map: np.ndarray, sector_id: int
    ) -> float:
        """Mean risk estimate over all cells in a sector.

        Args:
            risk_map: Float array of shape (H, W) with values in [0, 1].
            sector_id: Integer sector identifier.

        Returns:
            Mean risk value in [0, 1].
        """
        cells = self._sector_cells.get(sector_id, [])
        if not cells:
            return 0.0
        values = [float(risk_map[r, c]) for r, c in cells]
        return float(np.mean(values))

    def select_actions(
        self,
        risk_map: np.ndarray,
        uav_positions: List[Tuple[int, int]],
        battery_levels: Optional[List[float]] = None,
    ) -> AllocationDict:
        """Greedily assign UAVs to sectors (Algorithm 1).

        Sectors are sorted by descending risk; each UAV is assigned to the
        highest-unassigned sector within its energy budget.

        Args:
            risk_map: Current belief-derived risk map R_t, shape (H, W).
            uav_positions: Current (row, col) position of each UAV.
            battery_levels: Optional battery level in [0, 1] per UAV.
                            UAVs with battery < 0.1 are skipped (recharging).

        Returns:
            AllocationDict mapping uav_index → assigned sector_id.
        """
        n_uavs = len(uav_positions)
        if battery_levels is None:
            battery_levels = [1.0] * n_uavs

        # Score each sector by risk × coverage
        scores = {}
        for s_id in self._sector_cells:
            risk = self.compute_sector_risk(risk_map, s_id)
            coverage = _compute_coverage(uav_positions, self._sector_cells[s_id])
            scores[s_id] = risk * coverage

        sorted_sectors = sorted(scores, key=lambda s: scores[s], reverse=True)

        # Sort UAVs by proximity to the top-risk sector (if any)
        top_sector = sorted_sectors[0] if sorted_sectors else 0
        top_centroid = self.sector_centroid(top_sector)

        def proximity(i: int) -> float:
            r, c = uav_positions[i]
            tr, tc = top_centroid
            return float((r - tr) ** 2 + (c - tc) ** 2)

        uav_order = sorted(range(n_uavs), key=proximity)

        allocation: AllocationDict = {}
        assigned_sectors = set()

        for uav_idx in uav_order:
            if battery_levels[uav_idx] < 0.10:
                continue  # UAV is recharging
            for s_id in sorted_sectors:
                if s_id not in assigned_sectors:
                    allocation[uav_idx] = s_id
                    assigned_sectors.add(s_id)
                    break

        return allocation
