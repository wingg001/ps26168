"""HMM/Viterbi trajectory-level map matching.

Selects one road-edge candidate per trajectory point using:
- **Emission**: Gaussian distance penalty from point to candidate edge.
- **Transition**: Network travel distance between consecutive candidate
  edges compared to observed point-to-point distance.

This module is intentionally independent of the Phase 3/4 estimator.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import osmnx as ox
from shapely.geometry import Point as ShapelyPoint

from src.maps.map_match import CandidateResult


_UNMATCHED_EDGE = (-1, -1, -1)
_INF_COST = 1e18
_DISCONNECTED_PENALTY = 1e10


@dataclass
class HMMMatchResult:
    """Container for HMM map-matching results.

    Attributes
    ----------
    edges : numpy.ndarray of shape (N, 3)
        Selected edge ID ``(u, v, key)`` for each trajectory point.
        Unmatched points (zero candidates) have ``(-1, -1, -1)``.
    total_cost : float
        Sum of all per-point incremental costs.  ``inf`` if the path is
        infeasible.  Equals ``sum(costs)`` within floating-point tolerance.
    costs : numpy.ndarray of shape (N,)
        Per-point incremental (step) cost.  For matched points this is the
        emission cost plus the incoming transition cost from the previously
        matched point.  For unmatched (zero-candidate) points this is 0.0.
    """

    edges: npt.NDArray[np.int64]
    total_cost: float
    costs: npt.NDArray[np.float64]


def _network_distance(
    graph: nx.MultiDiGraph,
    edge_a: Tuple[int, int, int],
    edge_b: Tuple[int, int, int],
) -> float:
    """Return shortest directed network distance between two edges in metres.

    The distance is measured from the *end node* of *edge_a* to the
    *start node* of *edge_b* plus the length of *edge_b* itself.  This
    represents the travel distance if a vehicle exits *edge_a* and
    enters *edge_b*.

    Special cases
    -------------
    * **Same edge** ``(u, v, k) == (u, v, k)``: distance is 0 — the
      vehicle stays on the same road segment.
    * **Shared endpoint**  (end of A == start of B): distance is
      ``length(B)``.
    * **No directed path** from end-of-A to start-of-B: returns
      ``float('inf')``.
    """
    if edge_a == edge_b:
        return 0.0

    ua, va, _ = edge_a
    ub, vb, _ = edge_b

    # Length of the target edge
    edge_b_data = graph.get_edge_data(ub, vb, edge_b[2])
    if edge_b_data is None:
        return float("inf")
    len_b = edge_b_data.get("length", 0.0)

    # BFS from end-of-A to start-of-B along directed edges
    if va == ub:
        return len_b

    visited = {va}
    queue: deque[Tuple[int, float]] = deque([(va, 0.0)])

    while queue:
        node, dist_so_far = queue.popleft()
        for _, nbr, data in graph.out_edges(node, data=True):
            if nbr == ub:
                return dist_so_far + data.get("length", 0.0) + len_b
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, dist_so_far + data.get("length", 0.0)))

    return float("inf")


def match_trajectory_hmm(
    graph: nx.MultiDiGraph,
    latitudes: Union[Sequence[float], npt.NDArray[np.floating]],
    longitudes: Union[Sequence[float], npt.NDArray[np.floating]],
    candidate_result: CandidateResult,
    *,
    emission_sigma_m: float,
    transition_sigma_m: float,
) -> HMMMatchResult:
    """Select one candidate edge per point via Viterbi decoding.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Road graph with ``length`` edge attributes (metres).
    latitudes, longitudes : array-like
        Trajectory coordinates in EPSG:4326.  Must have the same length
        as ``candidate_result.candidates``.
    candidate_result : CandidateResult
        Output of :func:`find_edge_candidates`.
    emission_sigma_m : float
        Standard deviation of the Gaussian emission model (metres).
    transition_sigma_m : float
        Standard deviation of the Gaussian transition model (metres).

    Returns
    -------
    HMMMatchResult
        ``edges[i]`` is the selected ``(u, v, key)`` for point *i*, or
        ``(-1, -1, -1)`` if the point had no candidates.
        ``costs[i]`` is the incremental cost (emission + incoming
        transition) for matched points, or 0.0 for unmatched points.
        ``total_cost`` equals ``sum(costs)``.

    Notes
    -----
    * The first trajectory point uses only its emission cost (no
      transition penalty).
    * Points with zero candidates are assigned ``(-1, -1, -1)`` and
      zero cost; the Viterbi path bridges over them.
    * If no feasible path exists the total cost is ``inf``.
    """
    lat_arr = np.asarray(latitudes, dtype=np.float64)
    lon_arr = np.asarray(longitudes, dtype=np.float64)

    if lat_arr.shape != lon_arr.shape:
        raise ValueError(
            f"latitudes length {lat_arr.size} does not match "
            f"longitudes length {lon_arr.size}"
        )

    n_points = lat_arr.size
    if n_points == 0:
        return HMMMatchResult(
            edges=np.empty((0, 3), dtype=np.int64),
            total_cost=0.0,
            costs=np.empty(0, dtype=np.float64),
        )

    cand_lists = candidate_result.candidates
    assert len(cand_lists) == n_points

    # ------------------------------------------------------------------
    # Observed point-to-point distances in local UTM (metres)
    # ------------------------------------------------------------------
    pts_gdf = gpd.GeoDataFrame(
        geometry=[ShapelyPoint(lon, lat) for lon, lat in zip(lon_arr, lat_arr)],
        crs="EPSG:4326",
    )
    # Derive UTM from the graph edges (consistent with find_edge_candidates)
    _n, edges_gdf_raw = ox.graph_to_gdfs(graph)
    utm_crs = edges_gdf_raw.estimate_utm_crs()
    pts_proj = pts_gdf.to_crs(utm_crs)

    observed_dist = np.zeros(n_points, dtype=np.float64)
    for i in range(1, n_points):
        observed_dist[i] = pts_proj.geometry.iloc[i].distance(
            pts_proj.geometry.iloc[i - 1]
        )

    # ------------------------------------------------------------------
    # Pre-compute network distances between all candidate pairs
    # ------------------------------------------------------------------
    # Key: (prev_idx, curr_idx) -> network distance in metres
    _net_cache: Dict[Tuple[int, int], float] = {}

    def _net_dist(ci: int, cj: int) -> float:
        key = (ci, cj)
        if key not in _net_cache:
            e_prev = all_candidates[ci].edge
            e_curr = all_candidates[cj].edge
            _net_cache[key] = _network_distance(graph, e_prev, e_curr)
        return _net_cache[key]

    # Flatten candidate list with global indices
    all_candidates = []
    point_slices: List[Tuple[int, int]] = []  # (start, end) in all_candidates
    idx = 0
    for cl in cand_lists:
        start = idx
        for c in cl:
            all_candidates.append(c)
            idx += 1
        point_slices.append((start, idx))

    n_total = len(all_candidates)
    if n_total == 0:
        return HMMMatchResult(
            edges=np.full((n_points, 3), _UNMATCHED_EDGE, dtype=np.int64),
            total_cost=0.0,
            costs=np.zeros(n_points, dtype=np.float64),
        )

    # Pre-compute emission costs
    emit = np.empty(n_total, dtype=np.float64)
    for i, c in enumerate(all_candidates):
        emit[i] = (c.distance_m / emission_sigma_m) ** 2

    # ------------------------------------------------------------------
    # Viterbi
    # ------------------------------------------------------------------
    cum = np.full(n_total, _INF_COST, dtype=np.float64)
    back = np.full((n_points, n_total), -1, dtype=np.int64)
    back_time = np.full((n_points, n_total), -1, dtype=np.int64)

    # Find first candidate-bearing point for initialisation
    t_start = 0
    while t_start < n_points:
        s_ts, e_ts = point_slices[t_start]
        if s_ts < e_ts:
            break
        t_start += 1

    if t_start >= n_points:
        return HMMMatchResult(
            edges=np.full((n_points, 3), _UNMATCHED_EDGE, dtype=np.int64),
            total_cost=0.0,
            costs=np.zeros(n_points, dtype=np.float64),
        )

    s0, e0 = point_slices[t_start]
    for ci in range(s0, e0):
        cum[ci] = emit[ci]

    # Track the last time step that had candidates (for bridging gaps)
    last_valid_t = t_start

    # Forward pass
    for t in range(t_start + 1, n_points):
        s_curr, e_curr = point_slices[t]

        if s_curr == e_curr:
            # No candidates — skip, the next valid point will bridge
            continue

        # Transition from the last point that had candidates
        s_prev, e_prev = point_slices[last_valid_t]

        new_cum = np.full(e_curr - s_curr, _INF_COST, dtype=np.float64)
        new_back = np.full(e_curr - s_curr, -1, dtype=np.int64)

        for local_ci, ci in enumerate(range(s_curr, e_curr)):
            for pi in range(s_prev, e_prev):
                if cum[pi] >= _INF_COST:
                    continue
                nd = _net_dist(pi, ci)
                if nd == float("inf"):
                    trans = _DISCONNECTED_PENALTY
                else:
                    # Use accumulated observed distance over skipped points
                    gap_dist = observed_dist[last_valid_t + 1 : t + 1].sum()
                    trans = ((nd - gap_dist) / transition_sigma_m) ** 2
                cost = cum[pi] + trans + emit[ci]
                if cost < new_cum[local_ci]:
                    new_cum[local_ci] = cost
                    new_back[local_ci] = pi

        # Store results
        for local_ci, ci in enumerate(range(s_curr, e_curr)):
            cum[ci] = new_cum[local_ci]
            back[t][ci] = new_back[local_ci]
            back_time[t][ci] = last_valid_t

        last_valid_t = t

    # ------------------------------------------------------------------
    # Back-trace
    # ------------------------------------------------------------------
    selected_global = np.full(n_points, -1, dtype=np.int64)

    # Find best final candidate among reachable candidate-bearing points
    best_final = -1
    best_cost = _INF_COST
    t_end = -1
    for t_scan in range(n_points - 1, -1, -1):
        s_te, e_te = point_slices[t_scan]
        for ci in range(s_te, e_te):
            if cum[ci] < best_cost:
                best_cost = cum[ci]
                best_final = ci
                t_end = t_scan
        if best_final >= 0:
            break

    if best_final >= 0:
        selected_global[t_end] = best_final
        cur = best_final
        t = t_end
        while t > t_start:
            s_t, e_t = point_slices[t]
            if s_t == e_t:
                t -= 1
                continue
            prev_t = int(back_time[t][cur])
            cur = int(back[t][cur])
            selected_global[prev_t] = cur
            t = prev_t

    # ------------------------------------------------------------------
    # Assemble output  (incremental step costs)
    # ------------------------------------------------------------------
    edges_out = np.full((n_points, 3), _UNMATCHED_EDGE, dtype=np.int64)
    costs_out = np.zeros(n_points, dtype=np.float64)

    has_feasible = False
    prev_matched_t: int = -1

    for t in range(n_points):
        gi = selected_global[t]
        if gi >= 0:
            c = all_candidates[gi]
            edges_out[t] = [c.edge[0], c.edge[1], c.edge[2]]
            if prev_matched_t < 0:
                costs_out[t] = cum[gi]
            else:
                costs_out[t] = cum[gi] - cum[selected_global[prev_matched_t]]
            prev_matched_t = t
            has_feasible = True

    if has_feasible:
        total = float(np.sum(costs_out))
    elif n_total > 0:
        total = float("inf")
    else:
        total = 0.0

    return HMMMatchResult(edges=edges_out, total_cost=total, costs=costs_out)
