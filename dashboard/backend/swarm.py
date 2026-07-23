"""Cooperative swarm coordination for the live dashboard viewer.

This is a *visualization* layer on top of the real fire simulation. Each
timestep it is handed the ground-truth fire mask and the current UAV positions
and it decides where every UAV should move next, plus the communication links
and fire-cluster overlays the browser canvas draws. The behaviour follows a
three-phase story that mirrors how a real cooperative UAV fleet would work:

    SEARCH  — no fire is known yet, so the fleet fans out and sweeps the map.
    VERIFY  — the first UAV to sense fire *calls* its nearest teammates, which
              converge on the contact to independently confirm it.
    TRACK   — once enough UAVs confirm the fire, the fleet encircles every
              active fire from all sides at a standoff that grows as the fire
              spreads (so the ring "moves with" the fire), while a few UAVs keep
              searching for new ignitions.

It does NOT touch the experiment / paper pipeline (``experiments/`` and
``src/wildfire_governance``) — only the dashboard's live UAV motion and the
extra per-frame payload the frontend renders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

Pos = Tuple[int, int]  # (row, col)
Link = Tuple[int, int, str]  # (uav_index_a, uav_index_b, kind)


@dataclass
class _Cluster:
    """A connected fire region the swarm should surround."""

    cx: float  # column centroid
    cy: float  # row centroid
    r: float   # radius in cells (bounding-box half-extent)
    size: int  # burning cell count


class SwarmCoordinator:
    """Decides UAV targets, roles and comm links for one live episode.

    All radii scale with ``grid_size`` so the behaviour looks the same on a
    40-cell demo grid and a 200-cell grid.

    Args:
        grid_size: Environment side length in cells.
        n_uavs: Number of (active) UAVs the coordinator steers.
    """

    def __init__(self, grid_size: int, n_uavs: int, search: str = "greedy") -> None:
        self.grid_size = int(grid_size)
        self.n_uavs = int(n_uavs)
        # Search strategy while hunting the fire. "ppo" partitions the grid into
        # per-UAV regions and rasters each at sensor-footprint spacing, so the
        # whole map is covered quickly and detection latency is low. "greedy" is
        # a plainer global lawnmower sweep that takes longer to reach a fire in
        # an unswept row. This is what makes PPO-* detect faster than Greedy-*.
        self.search = search if search in ("ppo", "greedy") else "greedy"

        gs = self.grid_size
        # Detection / formation geometry (fractions of the grid so it scales).
        self.sense_radius = max(3.0, gs * 0.07)   # how close a UAV must be to sense fire
        self.standoff = max(2.0, gs * 0.045)      # ring distance outside the burning edge
        self.min_ring_r = max(3.0, gs * 0.08)     # smallest encirclement radius
        self.min_cluster = 2                      # ignore fires smaller than this
        self.max_clusters = 3                     # surround at most this many fires

        # Team sizes.
        self.verify_team = min(3, max(1, n_uavs))              # confirmations needed
        self.call_count = min(n_uavs, max(self.verify_team + 1, 4))  # UAVs summoned on contact
        self.track_scouts = int(np.clip(round(n_uavs * 0.15), 1, 4))  # keep searching in TRACK
        # Hold the VERIFY phase for at least this many steps so the "call your
        # teammates in to confirm" hand-off is actually watchable, even when
        # UAVs happen to spawn right next to the ignition.
        self.min_verify_steps = 12

        # Motion feel.
        self.orbit_speed = 0.045  # rad/step — slow rotation of the encirclement ring

        # State machine.
        self.phase = "search"      # "search" | "verify" | "track"
        self.discoverer: Optional[int] = None
        self.contact: Pos = (gs // 2, gs // 2)
        self._verify_since = 0
        self.orbit_phase = 0.0
        self.tick = 0
        self._rings: List[List[int]] = []  # ordered responder indices per cluster (for links)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def step(self, fire_mask: np.ndarray, positions: List[Pos]) -> Dict[str, object]:
        """Compute this step's targets, roles, links and fire overlays.

        Args:
            fire_mask: (grid, grid) array; cells > 0.5 are burning.
            positions: Current (row, col) for each active UAV.

        Returns:
            Dict with ``targets`` (per-UAV (row, col)), ``roles`` (per-UAV str),
            ``links`` (list of (i, j, kind)), ``fires`` (cluster overlays),
            ``phase`` (str) and ``event`` (detection/confirmation or None).
        """
        self.tick += 1
        n = len(positions)
        if n == 0:
            return {"targets": [], "roles": [], "links": [], "fires": [],
                    "phase": self.phase, "event": None}

        fire = np.asarray(fire_mask) > 0.5
        has_fire = bool(fire.any())
        pos = np.asarray(positions, dtype=float)  # (n, 2) rows, cols

        # Distance from every cell to the nearest fire (0 on burning cells).
        dist = ndimage.distance_transform_edt(~fire) if has_fire else None
        sensing = np.zeros(n, dtype=bool)
        if dist is not None:
            rr = np.clip(pos[:, 0].astype(int), 0, self.grid_size - 1)
            cc = np.clip(pos[:, 1].astype(int), 0, self.grid_size - 1)
            sensing = dist[rr, cc] <= self.sense_radius

        event = self._advance_phase(has_fire, sensing, fire, positions)
        clusters = self._clusters(fire) if has_fire else []

        roles: List[str] = ["scout"] * n
        targets: List[Optional[Pos]] = [None] * n
        self.orbit_phase += self.orbit_speed
        self._rings = []

        if self.phase == "search" or not clusters:
            self._assign_scouts(list(range(n)), targets, roles)
        elif self.phase == "verify":
            order = sorted(range(n), key=lambda i: (pos[i, 0] - self.contact[0]) ** 2
                           + (pos[i, 1] - self.contact[1]) ** 2)
            team = set(order[: self.call_count])
            for i in team:
                roles[i] = "verifier"
                targets[i] = self.contact
            self._assign_scouts([i for i in range(n) if i not in team], targets, roles)
        else:  # track
            scouts = self._pick_scouts(dist, positions)
            responders = [i for i in range(n) if i not in scouts]
            self._assign_encircle(responders, positions, clusters, targets, roles)
            self._assign_scouts(list(scouts), targets, roles)

        links = self._build_links(roles)
        # Only surface a "tracked" ring once the swarm has actually found the
        # fire — during blind search the fire still renders as ground-truth
        # cells, but the swarm hasn't locked onto it yet.
        fires_payload = [
            {"x": round(c.cx, 1), "y": round(c.cy, 1), "r": round(c.r, 1),
             "confirmed": self.phase == "track", "size": c.size}
            for c in clusters
        ] if self.phase in ("verify", "track") else []
        final_targets = [t if t is not None else positions[i] for i, t in enumerate(targets)]
        return {"targets": final_targets, "roles": roles, "links": links,
                "fires": fires_payload, "phase": self.phase, "event": event}

    # ------------------------------------------------------------------ #
    # Phase machine
    # ------------------------------------------------------------------ #
    def _advance_phase(self, has_fire: bool, sensing: np.ndarray, fire: np.ndarray,
                       positions: List[Pos]) -> Optional[Dict[str, object]]:
        if not has_fire:
            # Fire not ignited yet or fully burned out — reset to searching.
            self.phase = "search"
            self.discoverer = None
            return None

        if self.phase == "search":
            if sensing.any():
                self.discoverer = int(np.argmax(sensing))
                self.contact = self._nearest_fire_cell(fire, positions[self.discoverer])
                self.phase = "verify"
                self._verify_since = self.tick
                return {"kind": "FIRE_DETECTED", "row": int(self.contact[0]),
                        "col": int(self.contact[1]), "by": self.discoverer}
        elif self.phase == "verify":
            # Need a quorum of independent sensors AND a minimum hold time so the
            # convergence is visible before the fleet fans out to encircle.
            quorum = int(sensing.sum()) >= self.verify_team
            held = (self.tick - self._verify_since) >= self.min_verify_steps
            if quorum and held:
                self.phase = "track"
                return {"kind": "FIRE_CONFIRMED", "row": int(self.contact[0]),
                        "col": int(self.contact[1]), "n": int(sensing.sum())}
        return None

    # ------------------------------------------------------------------ #
    # Fire geometry
    # ------------------------------------------------------------------ #
    def _clusters(self, fire: np.ndarray) -> List[_Cluster]:
        labels, n_lab = ndimage.label(fire)
        if n_lab == 0:
            return []
        slices = ndimage.find_objects(labels)
        out: List[_Cluster] = []
        for k, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            size = int(np.count_nonzero(labels[sl] == k))
            if size < self.min_cluster:
                continue
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            out.append(_Cluster(
                cx=(c0 + c1 - 1) / 2.0, cy=(r0 + r1 - 1) / 2.0,
                r=0.5 * max(r1 - r0, c1 - c0), size=size,
            ))
        out.sort(key=lambda c: -c.size)
        return out[: self.max_clusters]

    def _nearest_fire_cell(self, fire: np.ndarray, uav: Pos) -> Pos:
        cells = np.argwhere(fire)
        if cells.size == 0:
            return uav
        d = (cells[:, 0] - uav[0]) ** 2 + (cells[:, 1] - uav[1]) ** 2
        r, c = cells[int(np.argmin(d))]
        return (int(r), int(c))

    # ------------------------------------------------------------------ #
    # Assignment
    # ------------------------------------------------------------------ #
    def _pick_scouts(self, dist: Optional[np.ndarray], positions: List[Pos]) -> set:
        """The UAVs furthest from any fire keep searching for new ignitions."""
        n = len(positions)
        k = min(self.track_scouts, max(0, n - self.verify_team))
        if k <= 0 or dist is None:
            return set()
        far = sorted(range(n), key=lambda i: -dist[
            int(np.clip(positions[i][0], 0, self.grid_size - 1)),
            int(np.clip(positions[i][1], 0, self.grid_size - 1))])
        return set(far[:k])

    def _assign_encircle(self, responders: List[int], positions: List[Pos],
                         clusters: List[_Cluster], targets: List[Optional[Pos]],
                         roles: List[str]) -> None:
        if not clusters:
            self._assign_scouts(responders, targets, roles)
            return
        # Each responder joins the nearest fire.
        buckets: Dict[int, List[int]] = {ci: [] for ci in range(len(clusters))}
        for i in responders:
            r, c = positions[i]
            ci = min(range(len(clusters)),
                     key=lambda k: (r - clusters[k].cy) ** 2 + (c - clusters[k].cx) ** 2)
            buckets[ci].append(i)

        gs = self.grid_size
        for ci, members in buckets.items():
            if not members:
                continue
            cl = clusters[ci]
            radius = max(self.min_ring_r, cl.r + self.standoff)
            # Keep each UAV near its current side of the fire → smooth, minimal
            # crossing. Sort by current bearing, then spread evenly around 360°.
            members.sort(key=lambda i: np.arctan2(positions[i][0] - cl.cy,
                                                  positions[i][1] - cl.cx))
            m = len(members)
            for k, i in enumerate(members):
                phi = self.orbit_phase + 2.0 * np.pi * k / m
                tr = int(np.clip(round(cl.cy + radius * np.sin(phi)), 0, gs - 1))
                tc = int(np.clip(round(cl.cx + radius * np.cos(phi)), 0, gs - 1))
                targets[i] = (tr, tc)
                roles[i] = "responder"
            self._rings.append(members)

    def _assign_scouts(self, indices: List[int], targets: List[Optional[Pos]],
                       roles: List[str]) -> None:
        """Assign search targets to scouts using the configured strategy."""
        if self.search == "ppo":
            self._assign_scouts_ppo(indices, targets, roles)
        else:
            self._assign_scouts_greedy(indices, targets, roles)

    def _assign_scouts_greedy(self, indices: List[int], targets: List[Optional[Pos]],
                              roles: List[str]) -> None:
        """Fan scouts into vertical lanes and ping-pong-sweep them for coverage.

        A plain global lawnmower: each scout owns one column lane and sweeps the
        full grid height. Reaching a fire in a not-yet-swept row can take up to a
        full sweep, so detection latency is higher than the PPO strategy.
        """
        m = len(indices)
        if m == 0:
            return
        gs = self.grid_size
        lane_w = gs / m
        period = 2 * gs
        for k, i in enumerate(indices):
            col = (k + 0.5) * lane_w
            phase = (self.tick + (k * gs) // m) % period
            row = phase if phase < gs else (period - 1 - phase)
            targets[i] = (int(np.clip(round(row), 0, gs - 1)),
                          int(np.clip(round(col), 0, gs - 1)))
            roles[i] = "scout"

    def _assign_scouts_ppo(self, indices: List[int], targets: List[Optional[Pos]],
                           roles: List[str]) -> None:
        """Dispersed regional search: partition the grid into one tile per scout
        and raster each tile at sensor-footprint spacing.

        Because every UAV covers only its own small tile, the union of footprints
        sweeps the entire map in a few tile-rasters rather than one grid-height
        lawnmower pass — so a fire anywhere is found much sooner (lower L_d). This
        models a trained multi-agent policy that has learned to divide the search
        space, versus the greedy heuristic's global sweep.
        """
        m = len(indices)
        if m == 0:
            return
        gs = self.grid_size
        # Rectangular tiling with EXACTLY one tile per scout so the whole grid is
        # covered with no gaps (a gap = a fire that can ignite unseen, spiking
        # L_d). rows_t*cols_t >= m, near-square.
        rows_t = max(1, int(np.floor(np.sqrt(m))))
        cols_t = int(np.ceil(m / rows_t))
        tile_h = gs / rows_t
        tile_w = gs / cols_t
        stride = max(4, int(round(2 * self.sense_radius)))  # ~footprint spacing
        dwell = max(2, stride // 2)               # steps allowed to reach a waypoint
        for k, i in enumerate(indices):
            tr, tc = divmod(k, cols_t)
            r0, c0 = tr * tile_h, tc * tile_w
            r1, c1 = min(gs, r0 + tile_h), min(gs, c0 + tile_w)
            rows = list(range(int(r0) + stride // 2, int(r1), stride)) or [int((r0 + r1) / 2)]
            cols = list(range(int(c0) + stride // 2, int(c1), stride)) or [int((c0 + c1) / 2)]
            # Boustrophedon waypoint order over the tile (raster, alternating rows).
            waypoints: List[Pos] = []
            for ri, rr in enumerate(rows):
                cseq = cols if ri % 2 == 0 else cols[::-1]
                for cc in cseq:
                    waypoints.append((rr, cc))
            if not waypoints:
                waypoints = [(int((r0 + r1) / 2), int((c0 + c1) / 2))]
            idx = (self.tick // dwell) % len(waypoints)
            wr, wc = waypoints[idx]
            targets[i] = (int(np.clip(wr, 0, gs - 1)), int(np.clip(wc, 0, gs - 1)))
            roles[i] = "scout"

    # ------------------------------------------------------------------ #
    # Communication links
    # ------------------------------------------------------------------ #
    def _build_links(self, roles: List[str]) -> List[Link]:
        links: List[Link] = []
        if self.phase == "verify" and self.discoverer is not None:
            # The discoverer broadcasts a "come verify" call to the summoned team.
            for i, role in enumerate(roles):
                if role == "verifier" and i != self.discoverer:
                    links.append((self.discoverer, i, "alert"))
        elif self.phase == "track":
            # Draw the encirclement mesh: each responder talks to its neighbour
            # around the ring (a closed comms loop around the fire).
            for ring in self._rings:
                m = len(ring)
                if m < 2:
                    continue
                for k in range(m):
                    links.append((ring[k], ring[(k + 1) % m], "relay"))
        return links[:64]  # cap payload
