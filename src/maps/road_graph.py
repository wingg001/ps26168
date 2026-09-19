"""Road-network foundation for Phase 5 map matching.

Downloads a drivable-road graph from OpenStreetMap via OSMnx and returns
it as a NetworkX MultiDiGraph.  This module is intentionally independent
of the Phase 3/4 estimator (src/filters/).

Typical usage
-------------
>>> G = build_road_graph(lat_bounds=(51.40, 51.55), lon_bounds=(-0.30, -0.10))
>>> len(G.nodes), len(G.edges)  # doctest: +SKIP
>>> edges, dists = find_nearest_edges(G, X=-0.15, Y=51.47, return_distance=True)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Sequence, Union

import networkx as nx
import numpy as np
import numpy.typing as npt
import osmnx as ox


def find_nearest_edges(
    graph: nx.MultiDiGraph,
    X: Union[float, Sequence[float], npt.NDArray[np.floating]],
    Y: Union[float, Sequence[float], npt.NDArray[np.floating]],
    *,
    return_distance: bool = False,
) -> Union[
    tuple[int, int, int],
    tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
    tuple[tuple[int, int, int], float],
    tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]],
]:
    """Find the nearest road-graph edge to each coordinate pair.

    Thin wrapper around :func:`osmnx.distance.nearest_edges` that
    normalises scalar and array inputs into a consistent return type.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Road graph (e.g. from :func:`build_road_graph`).
    X : float or array-like
        Longitude(s) of the query point(s).  Accepts a single float,
        a Python sequence, or a NumPy array.
    Y : float or array-like
        Latitude(s) of the query point(s).  Same shape rules as *X*.
    return_distance : bool, optional
        If *True*, also return the Euclidean distance(s) from each
        query point to its nearest edge (in the same units as the
        graph CRS — typically metres for a projected graph, or
        approximate degrees for a raw WGS-84 graph).

    Returns
    -------
    edges : tuple of arrays or single tuple
        For a single point, ``(u, v, k)`` — the nearest edge ID.
        For multiple points, three NumPy arrays ``(us, vs, ks)`` each
        of length *N*.
    distances : numpy.ndarray, optional
        Only returned when ``return_distance=True``.  A float64 array
        of per-point distances.

    Notes
    -----
    * The function passes ``X=longitude`` and ``Y=latitude`` to OSMnx,
      matching its coordinate convention.
    * No input coordinates are modified or snapped.
    """
    # Ensure numpy arrays for consistent downstream handling
    X_arr = np.atleast_1d(np.asarray(X, dtype=np.float64))
    Y_arr = np.atleast_1d(np.asarray(Y, dtype=np.float64))

    ne, dist = ox.distance.nearest_edges(graph, X_arr, Y_arr, return_dist=True)

    if return_distance:
        return ne, dist
    else:
        return ne


def build_road_graph(
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    *,
    network_type: str = "drive",
    simplify: bool = True,
) -> nx.MultiDiGraph:
    """Download and return a drivable-road graph for a bounding box.

    Parameters
    ----------
    lat_bounds : tuple[float, float]
        ``(lat_min, lat_max)`` — southern and northern latitude bounds
        in degrees (EPSG:4326).
    lon_bounds : tuple[float, float]
        ``(lon_min, lon_max)`` — western and eastern longitude bounds
        in degrees (EPSG:4326).
    network_type : str, optional
        OSMnx network filter.  Default ``"drive"`` retrieves drivable
        roads only.  alternatives: ``"bike"``, ``"walk"``, ``"all"``.
    simplify : bool, optional
        If *True* (default), simplify graph topology by removing
        interstitial nodes that do not represent intersections.

    Returns
    -------
    networkx.MultiDiGraph
        An undirected-at-scale, directed-at-edge-level graph whose nodes
        carry ``lat``/``lon`` attributes and whose edges carry ``length``
        (metres) and ``geometry`` attributes.  This is the standard OSMnx
        road graph used as foundation for later map-matching (Phase 5).

    Notes
    -----
    * The bounding box is converted internally to the OSMnx convention
      ``(west, south, east, north)`` before calling ``osmnx.graph_from_bbox``.
    * The function performs a live Overpass API query — it requires
      internet access and is *not* suitable for deterministic unit tests.
    * The returned graph is the largest weakly connected component
      (the OSMnx default when ``retain_all=False``).
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    # OSMnx expects bbox as (left, bottom, right, top) = (west, south, east, north)
    bbox = (lon_min, lat_min, lon_max, lat_max)

    G = ox.graph_from_bbox(
        bbox,
        network_type=network_type,
        simplify=simplify,
    )
    return G
