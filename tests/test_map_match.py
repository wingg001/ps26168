"""Unit tests for src.maps.map_match — no internet required.

All tests mock find_nearest_edges and ox.graph_to_gdfs so the normal
pytest suite never downloads data from OpenStreetMap / Overpass.
"""

import unittest
from unittest.mock import patch

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from src.maps.map_match import (
    CandidateResult,
    EdgeCandidate,
    MatchResult,
    find_edge_candidates,
    match_trajectory_nearest,
)


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
    """Helper: mock return for a single-point query (real OSMnx shape)."""
    ne = np.empty(1, dtype=object)
    ne[0] = (1, 2, 0)
    return (ne, np.array([0.003]))


def _mock_ne_multi(graph, lon, lat, **kwargs):
    """Helper: mock return for a two-point query (real OSMnx shape)."""
    n = len(lon)
    ne = np.empty(n, dtype=object)
    for i in range(n):
        ne[i] = (1, 2, 0)
    return (ne, np.array([0.003, 0.005])[:n])


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


# ---------------------------------------------------------------------------
# Helpers for find_edge_candidates tests
# ---------------------------------------------------------------------------

def _make_edges_gdf():
    """Return a small GeoDataFrame of edge geometries in EPSG:4326.

    Three edges:
      (10, 20, 0): horizontal line at lat 51.475,  lon -0.180 to -0.170
      (30, 40, 0): vertical   line at lon -0.175, lat  51.470 to 51.480
      (50, 60, 0): diagonal   line far away at    lat ~52.0, lon ~-1.5
    """
    idx = pd.MultiIndex.from_tuples(
        [(10, 20, 0), (30, 40, 0), (50, 60, 0)],
        names=["u", "v", "key"],
    )
    geoms = [
        LineString([(-0.180, 51.475), (-0.170, 51.475)]),  # horizontal
        LineString([(-0.175, 51.470), (-0.175, 51.480)]),  # vertical
        LineString([(-10.000, 55.000), (-9.990, 55.010)]),  # far away
    ]
    return gpd.GeoDataFrame(geometry=geoms, index=idx, crs="EPSG:4326")


class TestFindEdgeCandidates(unittest.TestCase):
    """Tests for find_edge_candidates with mocked graph_to_gdfs."""

    def _call(self, lats, lons, radius_m=200.0, max_candidates=10):
        """Helper: call find_edge_candidates with mocked graph_to_gdfs."""
        edges_gdf = _make_edges_gdf()
        with patch("src.maps.map_match.ox.graph_to_gdfs") as mock_gtg:
            mock_gtg.return_value = (None, edges_gdf)
            graph = _make_mock_graph()
            return find_edge_candidates(
                graph, lats, lons, radius_m=radius_m, max_candidates=max_candidates
            )

    def test_one_point_multiple_nearby_edges(self):
        """A point near two crossing edges must return both candidates."""
        result = self._call([51.475], [-0.175])
        self.assertEqual(len(result.candidates), 1)
        # Point at intersection of horizontal and vertical edges
        self.assertEqual(len(result.candidates[0]), 2)
        edges = {c.edge for c in result.candidates[0]}
        self.assertIn((10, 20, 0), edges)
        self.assertIn((30, 40, 0), edges)

    def test_candidates_sorted_by_distance(self):
        """Candidates must be sorted nearest-first."""
        result = self._call([51.475], [-0.175])
        dists = [c.distance_m for c in result.candidates[0]]
        self.assertEqual(dists, sorted(dists))

    def test_max_candidates_limits_results(self):
        """At most max_candidates must be returned per point."""
        result = self._call([51.475], [-0.175], max_candidates=1)
        self.assertEqual(len(result.candidates[0]), 1)

    def test_radius_excludes_distant_edges(self):
        """Edges beyond radius_m must not appear as candidates."""
        result = self._call([51.475], [-0.175], radius_m=100.0)
        # Only edges within 100 m should be returned
        for c in result.candidates[0]:
            self.assertLessEqual(c.distance_m, 100.0)

    def test_point_with_zero_candidates(self):
        """A point far from all edges must have an empty candidate list."""
        result = self._call([0.000], [0.000], radius_m=50.0)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.candidates[0]), 0)

    def test_multiple_points_preserve_order(self):
        """Candidate list order must match input point order."""
        lats = [51.475, 0.000]
        lons = [-0.175, 0.000]
        result = self._call(lats, lons)
        self.assertEqual(len(result.candidates), 2)
        # First point has candidates, second does not
        self.assertGreater(len(result.candidates[0]), 0)
        self.assertEqual(len(result.candidates[1]), 0)

    def test_empty_trajectory(self):
        """Empty input must return empty CandidateResult."""
        result = self._call([], [])
        self.assertEqual(result.candidates, [])

    def test_mismatched_lengths_raises_value_error(self):
        """Mismatched latitude/longitude lengths must raise ValueError."""
        with self.assertRaises(ValueError):
            self._call([51.475, 51.480], [-0.175])

    def test_distances_are_in_meters(self):
        """Returned distances must be in metres (positive, reasonable)."""
        result = self._call([51.475], [-0.175])
        for c in result.candidates[0]:
            self.assertGreaterEqual(c.distance_m, 0.0)
            # Edge at same lat should have ~0 distance for the horizontal edge
            # and ~0 distance for the vertical edge at the crossing point
            self.assertLess(c.distance_m, 1000.0)

    def test_edge_ids_are_u_v_key_tuples(self):
        """Edge IDs must be (u, v, key) tuples."""
        result = self._call([51.475], [-0.175])
        for c in result.candidates[0]:
            self.assertIsInstance(c.edge, tuple)
            self.assertEqual(len(c.edge), 3)
            self.assertIsInstance(c.edge[0], (int, np.integer))
            self.assertIsInstance(c.edge[1], (int, np.integer))
            self.assertIsInstance(c.edge[2], (int, np.integer))

    def test_utm_crs_used_not_3857(self):
        """Both edges and points must be projected to the same local UTM CRS."""
        edges_gdf = _make_edges_gdf()
        expected_utm = edges_gdf.estimate_utm_crs()
        self.assertNotEqual(expected_utm, "EPSG:3857")

        with patch("src.maps.map_match.ox.graph_to_gdfs") as mock_gtg:
            mock_gtg.return_value = (None, edges_gdf)
            # Patch estimate_utm_crs to record it is called and capture the value
            with patch.object(edges_gdf, "estimate_utm_crs", wraps=edges_gdf.estimate_utm_crs) as mock_utm:
                graph = _make_mock_graph()
                result = find_edge_candidates(graph, [51.475], [-0.175],
                                              radius_m=200.0, max_candidates=10)
                mock_utm.assert_called_once()
                # Verify the returned distances are reasonable local metres
                self.assertGreater(len(result.candidates[0]), 0)
                for c in result.candidates[0]:
                    self.assertGreaterEqual(c.distance_m, 0.0)
                    self.assertLess(c.distance_m, 2000.0)


if __name__ == "__main__":
    unittest.main()
