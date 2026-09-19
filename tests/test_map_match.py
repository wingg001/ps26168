"""Unit tests for src.maps.map_match — no internet required.

All tests mock find_nearest_edges so the normal pytest suite never
downloads data from OpenStreetMap / Overpass.
"""

import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from src.maps.map_match import MatchResult, match_trajectory_nearest


def _make_mock_graph() -> nx.MultiDiGraph:
    """Return a small deterministic MultiDiGraph that mimics OSMnx output."""
    G = nx.MultiDiGraph()
    for nid, lat, lon in [
        (1, 51.45, -0.20),
        (2, 51.45, -0.15),
        (3, 51.50, -0.15),
        (4, 51.50, -0.20),
    ]:
        G.add_node(nid, x=lon, y=lat, lat=lat, lon=lon)
    for u, v in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        G.add_edge(u, v, length=500.0, highway="residential", geometry=None)
    return G


def _mock_ne_single(graph, lon, lat, **kwargs):
    """Helper: mock return for a single-point query."""
    return ((1, 2, 0), np.array([0.003]))


def _mock_ne_multi(graph, lon, lat, **kwargs):
    """Helper: mock return for a two-point query."""
    n = len(lon)
    return (
        (np.array([1] * n), np.array([2] * n), np.array([0] * n)),
        np.array([0.003, 0.005])[:n],
    )


class TestMatchTrajectoryNearest(unittest.TestCase):
    """Tests for match_trajectory_nearest with mocked OSMnx calls."""

    @patch("src.maps.map_match.find_nearest_edges", side_effect=_mock_ne_single)
    def test_single_point_trajectory(self, mock_fne):
        """A single coordinate pair must produce a MatchResult with one entry."""
        graph = _make_mock_graph()
        result = match_trajectory_nearest(graph, [51.475], [-0.175])
        self.assertIsInstance(result, MatchResult)
        self.assertEqual(result.edges.shape, (1, 3))
        self.assertEqual(result.distances.shape, (1,))

    @patch("src.maps.map_match.find_nearest_edges", side_effect=_mock_ne_multi)
    def test_multi_point_trajectory(self, mock_fne):
        """Multiple coordinates must produce N entries in the result."""
        graph = _make_mock_graph()
        lats = [51.475, 51.480]
        lons = [-0.175, -0.180]
        result = match_trajectory_nearest(graph, lats, lons)
        self.assertEqual(result.edges.shape, (2, 3))
        self.assertEqual(result.distances.shape, (2,))

    @patch("src.maps.map_match.find_nearest_edges", side_effect=_mock_ne_multi)
    def test_output_preserves_point_order(self, mock_fne):
        """The order of edges/distances must match the input point order."""
        graph = _make_mock_graph()
        lats = [51.475, 51.480]
        lons = [-0.175, -0.180]
        result = match_trajectory_nearest(graph, lats, lons)
        # Mock returns same edge (1,2,0) for both points
        np.testing.assert_array_equal(result.edges[:, 0], [1, 1])
        np.testing.assert_array_equal(result.edges[:, 1], [2, 2])
        np.testing.assert_array_equal(result.edges[:, 2], [0, 0])

    @patch("src.maps.map_match.find_nearest_edges", side_effect=_mock_ne_multi)
    def test_distances_are_returned(self, mock_fne):
        """Distances must be non-negative and match the mock values."""
        graph = _make_mock_graph()
        result = match_trajectory_nearest(graph, [51.475, 51.480], [-0.175, -0.180])
        np.testing.assert_array_almost_equal(result.distances, [0.003, 0.005])

    def test_empty_input(self):
        """Empty arrays must return empty MatchResult without calling OSMnx."""
        graph = _make_mock_graph()
        result = match_trajectory_nearest(graph, [], [])
        self.assertEqual(result.edges.shape, (0, 3))
        self.assertEqual(result.distances.shape, (0,))

    def test_mismatched_lengths_raises_value_error(self):
        """Mismatched latitude/longitude lengths must raise ValueError."""
        graph = _make_mock_graph()
        with self.assertRaises(ValueError):
            match_trajectory_nearest(graph, [51.475, 51.480], [-0.175])

    @patch("src.maps.map_match.find_nearest_edges", side_effect=_mock_ne_single)
    def test_one_point_result_edge_tuple_format(self, mock_fne):
        """Single-point result must contain exactly 3 columns (u, v, k)."""
        graph = _make_mock_graph()
        result = match_trajectory_nearest(graph, [51.475], [-0.175])
        self.assertEqual(result.edges.ndim, 2)
        self.assertEqual(result.edges.shape[1], 3)
        # Check it's an integer array
        self.assertTrue(np.issubdtype(result.edges.dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
