"""Trajectory-level nearest-road matching baseline.

Matches each trajectory point independently to its nearest road-graph
edge using :func:`src.maps.road_graph.find_nearest_edges`.

This module is intentionally independent of the Phase 3/4 estimator
and does not perform HMM smoothing or routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import networkx as nx
import numpy as np
import numpy.typing as npt

from src.maps.road_graph import find_nearest_edges


@dataclass
class MatchResult:
    """Container for nearest-road matching results.

    Attributes
    ----------
    edges : numpy.ndarray of shape (N, 3)
        Matched edge IDs for each trajectory point, with columns
        ``(u, v, k)``.
    distances : numpy.ndarray of shape (N,)
        Euclidean distance from each point to its matched edge.
    """

    edges: npt.NDArray[np.int64]
    distances: npt.NDArray[np.float64]


def match_trajectory_nearest(
    graph: nx.MultiDiGraph,
    latitudes: Union[Sequence[float], npt.NDArray[np.floating]],
    longitudes: Union[Sequence[float], npt.NDArray[np.floating]],
    *,
    return_distance: bool = True,
) -> MatchResult:
    """Match every trajectory point to its nearest road-graph edge.

    Thin wrapper around :func:`find_nearest_edges` that validates
    inputs and packages the result into a structured
    :class:`MatchResult`.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Road graph (e.g. from :func:`build_road_graph`).
    latitudes : array-like
        Latitude(s) of the trajectory points (degrees, EPSG:4326).
        Must have the same length as *longitudes*.
    longitudes : array-like
        Longitude(s) of the trajectory points (degrees, EPSG:4326).
        Must have the same length as *latitudes*.
    return_distance : bool, optional
        If *True* (default), include per-point distances in the result.

    Returns
    -------
    MatchResult
        ``edges`` : ndarray of shape (N, 3)
            Columns ``(u, v, k)`` for each matched edge.
        ``distances`` : ndarray of shape (N,)
            Distance from each point to its matched edge.  If
            ``return_distance=False``, this array is filled with zeros.

    Raises
    ------
    ValueError
        If *latitudes* and *longitudes* have different lengths.

    Notes
    -----
    * Each point is matched independently — no smoothing, no HMM, no
      routing continuity.
    * Input coordinates are not modified, snapped, or reordered.
    """
    lat_arr = np.asarray(latitudes, dtype=np.float64)
    lon_arr = np.asarray(longitudes, dtype=np.float64)

    if lat_arr.shape != lon_arr.shape:
        raise ValueError(
            f"latitudes length {lat_arr.size} does not match "
            f"longitudes length {lon_arr.size}"
        )

    # Empty trajectory — return empty arrays
    if lat_arr.size == 0:
        return MatchResult(
            edges=np.empty((0, 3), dtype=np.int64),
            distances=np.empty(0, dtype=np.float64),
        )

    # Flatten to 1-D in case user passed (N,1) or similar
    lon_flat = lon_arr.ravel()
    lat_flat = lat_arr.ravel()

    # find_nearest_edges expects X=longitude, Y=latitude
    ne, dist = find_nearest_edges(graph, lon_flat, lat_flat, return_distance=True)

    # ne may be a tuple of 3 arrays (us, vs, ks) or a single (u,v,k) tuple
    if isinstance(ne, tuple) and len(ne) == 3:
        edges = np.column_stack([np.asarray(a) for a in ne])
    else:
        edges = np.array([ne], dtype=np.int64)

    if return_distance:
        dist_arr = np.asarray(dist, dtype=np.float64).ravel()
    else:
        dist_arr = np.zeros(edges.shape[0], dtype=np.float64)

    return MatchResult(edges=edges, distances=dist_arr)
