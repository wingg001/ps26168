"""Unit tests for src.maps.road_graph — no internet required.

All tests mock osmnx.graph_from_bbox so the normal pytest suite never
downloads data from OpenStreetMap / Overpass.
"""

import unittest
from unittest.mock import patch

import networkx as nx

from src.maps.road_graph import build_road_graph


def _make_mock_graph() -> nx.MultiDiGraph:
    """Return a small deterministic MultiDiGraph that mimics OSMnx output."""
    G = nx.MultiDiGraph()
    # Four nodes forming a tiny square
    for nid, lat, lon in [
        (1, 51.45, -0.20),
        (2, 51.45, -0.15),
        (3, 51.50, -0.15),
        (4, 51.50, -0.20),
    ]:
        G.add_node(nid, x=lon, y=lat, lat=lat, lon=lon)
    # Four directed edges (clockwise loop)
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    for u, v in edges:
        G.add_edge(
            u, v,
            length=500.0,
            highway="residential",
            geometry=None,
        )
    return G


class TestBuildRoadGraph(unittest.TestCase):
    """Tests for build_road_graph with mocked OSMnx calls."""

    @patch("src.maps.road_graph.ox.graph_from_bbox")
    def test_returns_networkx_multidigraph(self, mock_gfb):
        """Function must return a networkx.MultiDiGraph."""
        mock_gfb.return_value = _make_mock_graph()
        G = build_road_graph((51.40, 51.55), (-0.30, -0.10))
        self.assertIsInstance(G, nx.MultiDiGraph)

    @patch("src.maps.road_graph.ox.graph_from_bbox")
    def test_graph_has_nodes_and_edges(self, mock_gfb):
        """Returned graph must have nodes and edges."""
        mock_gfb.return_value = _make_mock_graph()
        G = build_road_graph((51.40, 51.55), (-0.30, -0.10))
        self.assertGreater(len(G.nodes), 0)
        self.assertGreater(len(G.edges), 0)

    @patch("src.maps.road_graph.ox.graph_from_bbox")
    def test_bbox_conversion(self, mock_gfb):
        """lat_bounds/lon_bounds must be converted to (west, south, east, north)."""
        mock_gfb.return_value = _make_mock_graph()
        build_road_graph((51.40, 51.55), (-0.30, -0.10))
        args, kwargs = mock_gfb.call_args
        # First positional arg is the bbox tuple
        bbox = args[0]
        self.assertEqual(bbox, (-0.30, 51.40, -0.10, 51.55))

    @patch("src.maps.road_graph.ox.graph_from_bbox")
    def test_network_type_drive(self, mock_gfb):
        """Default network_type must be 'drive'."""
        mock_gfb.return_value = _make_mock_graph()
        build_road_graph((51.40, 51.55), (-0.30, -0.10))
        _, kwargs = mock_gfb.call_args
        self.assertEqual(kwargs.get("network_type", "drive"), "drive")

    @patch("src.maps.road_graph.ox.graph_from_bbox")
    def test_simplify_default_true(self, mock_gfb):
        """Default simplify must be True."""
        mock_gfb.return_value = _make_mock_graph()
        build_road_graph((51.40, 51.55), (-0.30, -0.10))
        _, kwargs = mock_gfb.call_args
        self.assertTrue(kwargs.get("simplify", True))

    @patch("src.maps.road_graph.ox.graph_from_bbox")
    def test_nodes_have_coordinates(self, mock_gfb):
        """Nodes must carry lat/lon attributes."""
        mock_gfb.return_value = _make_mock_graph()
        G = build_road_graph((51.40, 51.55), (-0.30, -0.10))
        for nid in G.nodes:
            data = G.nodes[nid]
            self.assertIn("lat", data)
            self.assertIn("lon", data)

    @patch("src.maps.road_graph.ox.graph_from_bbox")
    def test_edges_have_length(self, mock_gfb):
        """Edges must carry a length attribute."""
        mock_gfb.return_value = _make_mock_graph()
        G = build_road_graph((51.40, 51.55), (-0.30, -0.10))
        for u, v, k in G.edges:
            self.assertIn("length", G.edges[u, v, k])


if __name__ == "__main__":
    unittest.main()
