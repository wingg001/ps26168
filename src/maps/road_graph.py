"""Road-network foundation for Phase 5 map matching.

Downloads a drivable-road graph from OpenStreetMap via OSMnx and returns
it as a NetworkX MultiDiGraph.  This module is intentionally independent
of the Phase 3/4 estimator (src/filters/).

Typical usage
-------------
>>> G = build_road_graph(lat_bounds=(51.40, 51.55), lon_bounds=(-0.30, -0.10))
>>> len(G.nodes), len(G.edges)  # doctest: +SKIP
"""

from __future__ import annotations

import networkx as nx
import osmnx as ox


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
