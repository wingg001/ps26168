"""Trajectory-level nearest-road matching baseline.

Matches each trajectory point independently to its nearest road-graph
edge using :func:`src.maps.road_graph.find_nearest_edges`.

This module is intentionally independent of the Phase 3/4 estimator
and does not perform HMM smoothing or routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import osmnx as ox
from shapely.geometry import Point

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

    # OSMnx returns ndarray[N] dtype=object, each element a (u, v, k) tuple
    edges = np.array(list(ne), dtype=np.int64)

    if return_distance:
        dist_arr = np.asarray(dist, dtype=np.float64).ravel()
    else:
        dist_arr = np.zeros(edges.shape[0], dtype=np.float64)

    return MatchResult(edges=edges, distances=dist_arr)


@dataclass
class EdgeCandidate:
    """A single road-edge candidate for a trajectory point.

    Attributes
    ----------
    edge : tuple[int, int, int]
        Edge ID as ``(u, v, key)``.
    distance_m : float
        Perpendicular distance from the trajectory point to the edge
        geometry, in metres.
    """

    edge: tuple[int, int, int]
    distance_m: float


@dataclass
class CandidateResult:
    """Container for per-point edge-candidate lists.

    Attributes
    ----------
    candidates : list[list[EdgeCandidate]]
        One inner list per trajectory point.  Each inner list contains
        :class:`EdgeCandidate` instances sorted nearest-first, with at
        most *max_candidates* entries.  Points with no nearby edges
        have an empty inner list.
    """

    candidates: List[List[EdgeCandidate]] = field(default_factory=list)


def find_edge_candidates(
    graph: nx.MultiDiGraph,
    latitudes: Union[Sequence[float], npt.NDArray[np.floating]],
    longitudes: Union[Sequence[float], npt.NDArray[np.floating]],
    *,
    radius_m: float,
    max_candidates: int,
) -> CandidateResult:
    """Find nearby road-edge candidates for each trajectory point.

    Projects both the road-graph edges and the trajectory points to
    EPSG:3857 (Web Mercator) so that all distance calculations are
    performed in metres.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Road graph (e.g. from :func:`build_road_graph`).
    latitudes : array-like
        Latitude(s) of the trajectory points (degrees, EPSG:4326).
    longitudes : array-like
        Longitude(s) of the trajectory points (degrees, EPSG:4326).
    radius_m : float
        Search radius in metres.  Only edges whose geometry comes
        within this distance of a trajectory point are returned as
        candidates.
    max_candidates : int
        Maximum number of candidates retained per point (sorted
        nearest-first).

    Returns
    -------
    CandidateResult
        ``candidates[i]`` is a list of :class:`EdgeCandidate` for
        trajectory point *i*, sorted by ascending distance.

    Raises
    ------
    ValueError
        If *latitudes* and *longitudes* have different lengths.
    """
    lat_arr = np.asarray(latitudes, dtype=np.float64)
    lon_arr = np.asarray(longitudes, dtype=np.float64)

    if lat_arr.shape != lon_arr.shape:
        raise ValueError(
            f"latitudes length {lat_arr.size} does not match "
            f"longitudes length {lon_arr.size}"
        )

    if lat_arr.size == 0:
        return CandidateResult(candidates=[])

    # Extract edge GeoDataFrame from the graph (EPSG:4326 by default)
    _nodes, edges_gdf = ox.graph_to_gdfs(graph)

    # Derive a local UTM CRS from the edge geometries for accurate
    # metre-scale distance calculations (avoids EPSG:3857 distortion).
    utm_crs = edges_gdf.estimate_utm_crs()
    edges_proj = edges_gdf.to_crs(utm_crs)

    # Build trajectory-point GeoDataFrame and project to the same UTM CRS
    pts_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(lon_arr, lat_arr)],
        crs="EPSG:4326",
    )
    pts_proj = pts_gdf.to_crs(utm_crs)

    # Spatial index for fast radius queries
    sindex = edges_proj.sindex

    candidates: List[List[EdgeCandidate]] = []

    for pt in pts_proj.geometry:
        # Buffer the point and query the spatial index
        buf = pt.buffer(radius_m)
        possible_idx = list(sindex.intersection(buf.bounds))

        # Collect candidates that truly intersect the buffer
        nearby = []
        for idx in possible_idx:
            edge_geom = edges_proj.geometry.iloc[idx]
            dist = pt.distance(edge_geom)
            if dist <= radius_m:
                u, v, k = edges_proj.index[idx]
                nearby.append(EdgeCandidate(edge=(u, v, k), distance_m=dist))

        # Sort nearest-first and cap at max_candidates
        nearby.sort(key=lambda c: c.distance_m)
        candidates.append(nearby[:max_candidates])

    return CandidateResult(candidates=candidates)
