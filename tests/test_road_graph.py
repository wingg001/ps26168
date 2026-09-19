"""Unit tests for src.maps.road_graph — no internet required.

All tests mock osmnx calls so the normal pytest suite never downloads
data from OpenStreetMap / Overpass.
"""

import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from src.maps.road_graph import build_road_graph, find_nearest_edges


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


class TestFindNearestEdges(unittest.TestCase):
    """Tests for find_nearest_edges with mocked OSMnx calls."""

    @patch("src.maps.road_graph.ox.distance.nearest_edges")
    def test_single_point_returns_edge(self, mock_ne):
        """A single coordinate pair must return the OSMnx edge result."""
        ne = np.empty(1, dtype=object)
        ne[0] = (1, 2, 0)
        mock_ne.return_value = (ne, np.array([0.003]))
        graph = _make_mock_graph()
        result = find_nearest_edges(graph, -0.175, 51.475)
        # Real OSMnx returns ndarray shape (N,) dtype=object
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.dtype, object)
        self.assertEqual(tuple(result[0]), (1, 2, 0))

    @patch("src.maps.road_graph.ox.distance.nearest_edges")
    def test_multiple_coordinates(self, mock_ne):
        """Arrays of coordinates must be handled without error."""
        ne = np.empty(2, dtype=object)
        ne[0] = (1, 2, 0)
        ne[1] = (3, 4, 0)
        mock_ne.return_value = (ne, np.array([0.003, 0.005]))
        graph = _make_mock_graph()
        xs = np.array([-0.175, -0.175])
        ys = np.array([51.475, 51.475])
        edges, dists = find_nearest_edges(graph, xs, ys, return_distance=True)
        self.assertEqual(edges.shape, (2,))
        self.assertEqual(dists.shape, (2,))

    @patch("src.maps.road_graph.ox.distance.nearest_edges")
    def test_longitude_passed_as_X(self, mock_ne):
        """The first coordinate argument must be passed as X (longitude)."""
        ne = np.empty(1, dtype=object)
        ne[0] = (1, 2, 0)
        mock_ne.return_value = (ne, np.array([0.0]))
        graph = _make_mock_graph()
        find_nearest_edges(graph, -0.175, 51.475)
        args, kwargs = mock_ne.call_args
        # args = (graph, X, Y)
        np.testing.assert_array_equal(args[1], np.array([-0.175]))

    @patch("src.maps.road_graph.ox.distance.nearest_edges")
    def test_latitude_passed_as_Y(self, mock_ne):
        """The second coordinate argument must be passed as Y (latitude)."""
        ne = np.empty(1, dtype=object)
        ne[0] = (1, 2, 0)
        mock_ne.return_value = (ne, np.array([0.0]))
        graph = _make_mock_graph()
        find_nearest_edges(graph, -0.175, 51.475)
        args, kwargs = mock_ne.call_args
        np.testing.assert_array_equal(args[2], np.array([51.475]))

    @patch("src.maps.road_graph.ox.distance.nearest_edges")
    def test_distance_returned_when_requested(self, mock_ne):
        """With return_distance=True the distance array must be returned."""
        ne = np.empty(2, dtype=object)
        ne[0] = (1, 2, 0)
        ne[1] = (3, 4, 0)
        dist_array = np.array([0.003, 0.007])
        mock_ne.return_value = (ne, dist_array)
        graph = _make_mock_graph()
        edges, dists = find_nearest_edges(
            graph, [-0.175, -0.180], [51.475, 51.480], return_distance=True
        )
        np.testing.assert_array_equal(dists, dist_array)

    @patch("src.maps.road_graph.ox.distance.nearest_edges")
    def test_return_distance_false_omits_dist(self, mock_ne):
        """With return_distance=False only the edge IDs are returned."""
        ne = np.empty(1, dtype=object)
        ne[0] = (1, 2, 0)
        mock_ne.return_value = (ne, np.array([0.003]))
        graph = _make_mock_graph()
        result = find_nearest_edges(graph, [-0.175], [51.475], return_distance=False)
        # result should be the edge ndarray directly (no distance array)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.dtype, object)
        self.assertEqual(tuple(result[0]), (1, 2, 0))

    @patch("src.maps.road_graph.ox.distance.nearest_edges")
    def test_empty_input_handled(self, mock_ne):
        """Empty coordinate arrays must be handled without raising."""
        ne = np.empty(0, dtype=object)
        mock_ne.return_value = (ne, np.array([], dtype=np.float64))
        graph = _make_mock_graph()
        edges, dists = find_nearest_edges(
            graph, np.array([]), np.array([]), return_distance=True
        )
        self.assertEqual(edges.shape, (0,))
        self.assertEqual(dists.shape, (0,))

    @patch("src.maps.road_graph.ox.distance.nearest_edges")
    def test_scalar_input_wrapped_as_array(self, mock_ne):
        """Scalar float inputs must be wrapped into arrays for OSMnx."""
        ne = np.empty(1, dtype=object)
        ne[0] = (1, 2, 0)
        mock_ne.return_value = (ne, np.array([0.0]))
        graph = _make_mock_graph()
        find_nearest_edges(graph, -0.175, 51.475)
        args, _ = mock_ne.call_args
        self.assertIsInstance(args[1], np.ndarray)
        self.assertIsInstance(args[2], np.ndarray)


if __name__ == "__main__":
    unittest.main()
