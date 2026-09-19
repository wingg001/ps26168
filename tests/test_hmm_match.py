"""Unit tests for src.maps.hmm_match -- no internet required.

Uses a small synthetic NetworkX MultiDiGraph with deterministic edge
lengths and mocked OSMnx calls.
"""

import unittest
from unittest.mock import patch

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from src.maps.map_match import CandidateResult, EdgeCandidate
from src.maps.hmm_match import HMMMatchResult, _network_distance, match_trajectory_hmm


# ---------------------------------------------------------------------------
# Synthetic graph helpers
# ---------------------------------------------------------------------------

def _build_test_graph():
    """Build a small directed graph for testing.

    Layout (x = lon, y = lat):
        A(0) ---e1(100m)--> B(1) ---e2(100m)--> C(2)
                                        |
                                     e3(100m)
                                        |
                                        v
                                       D(3)

    Node coords (EPSG:4326):
        A = (0.0000, 0.0000)
        B = (0.0010, 0.0000)   ~ 100 m east of A
        C = (0.0020, 0.0000)   ~ 100 m east of B
        D = (0.0010, -0.0010)  ~ 100 m south of B
    """
    G = nx.MultiDiGraph()
    coords = {
        0: (0.0, 0.0),
        1: (0.001, 0.0),
        2: (0.002, 0.0),
        3: (0.001, -0.001),
    }
    for nid, (lon, lat) in coords.items():
        G.add_node(nid, x=lon, y=lat, lat=lat, lon=lon)
    G.add_edge(0, 1, length=100.0, highway="residential", geometry=None)
    G.add_edge(1, 2, length=100.0, highway="residential", geometry=None)
    G.add_edge(1, 3, length=100.0, highway="residential", geometry=None)
    return G


def _make_edges_gdf():
    """Return a GeoDataFrame matching the test graph edges in EPSG:4326."""
    idx = pd.MultiIndex.from_tuples(
        [(0, 1, 0), (1, 2, 0), (1, 3, 0)], names=["u", "v", "key"]
    )
    geoms = [
        LineString([(0.0, 0.0), (0.001, 0.0)]),
        LineString([(0.001, 0.0), (0.002, 0.0)]),
        LineString([(0.001, 0.0), (0.001, -0.001)]),
    ]
    return gpd.GeoDataFrame(geometry=geoms, index=idx, crs="EPSG:4326")


def _make_candidates(lists):
    """Build a CandidateResult from a list-of-lists of (edge, dist) pairs."""
    cands = []
    for lst in lists:
        cands.append([EdgeCandidate(edge=e, distance_m=d) for e, d in lst])
    return CandidateResult(candidates=cands)


def _mock_hmm(lats, lons, cand_result, **kwargs):
    """Call match_trajectory_hmm with mocked graph_to_gdfs."""
    edges_gdf = _make_edges_gdf()
    graph = _build_test_graph()
    with patch("src.maps.hmm_match.ox.graph_to_gdfs") as mock_g2g:
        mock_g2g.return_value = (None, edges_gdf)
        return match_trajectory_hmm(
            graph, lats, lons, cand_result, **kwargs
        )


class TestNetworkDistance(unittest.TestCase):
    """Tests for the _network_distance helper."""

    def test_same_edge_returns_zero(self):
        G = _build_test_graph()
        self.assertAlmostEqual(_network_distance(G, (0, 1, 0), (0, 1, 0)), 0.0)

    def test_directly_connected(self):
        G = _build_test_graph()
        dist = _network_distance(G, (0, 1, 0), (1, 2, 0))
        self.assertAlmostEqual(dist, 100.0)

    def test_two_hop(self):
        G = _build_test_graph()
        dist = _network_distance(G, (0, 1, 0), (1, 3, 0))
        self.assertAlmostEqual(dist, 100.0)

    def test_disconnected_returns_inf(self):
        G = _build_test_graph()
        dist = _network_distance(G, (1, 2, 0), (0, 1, 0))
        self.assertEqual(dist, float("inf"))


class TestEmissionCost(unittest.TestCase):
    """Emission cost must favour the closer candidate."""

    def test_closer_candidate_preferred(self):
        """Point at 5 m from edge A and 20 m from edge B selects A."""
        result = _mock_hmm(
            [0.00005], [0.0005],
            _make_candidates([
                [((0, 1, 0), 20.0), ((1, 2, 0), 5.0)],
            ]),
            emission_sigma_m=10.0,
            transition_sigma_m=50.0,
        )
        # With single point, Viterbi picks lowest emission
        self.assertEqual(result.edges[0].tolist(), [1, 2, 0])


class TestTransitionPreference(unittest.TestCase):
    """Connected candidates should be preferred over disconnected ones."""

    def test_connected_beats_disconnected(self):
        """Two points: both have a close-but-disconnected and a slightly-far-but-connected candidate.
        Viterbi should prefer the connected path."""
        # Point 0 near edge (0,1,0), Point 1 near edge (1,2,0)
        # Candidate for P0: (0,1,0) at 2m, (1,3,0) at 2m
        # Candidate for P1: (1,2,0) at 2m, (1,3,0) at 20m
        # Edge (0,1,0)->(1,2,0) is connected, (1,3,0)->(1,3,0) is same-edge
        # But (0,1,0)->(1,3,0) is also connected (via node 1)
        # The real test: (1,3,0)->(1,2,0) is disconnected
        result = _mock_hmm(
            [0.00002, 0.00102], [0.0005, 0.0015],
            _make_candidates([
                [((0, 1, 0), 2.0), ((1, 3, 0), 2.0)],
                [((1, 2, 0), 2.0), ((1, 3, 0), 20.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        # Path (0,1,0)->(1,2,0) has small emission + small transition
        # Path (1,3,0)->(1,3,0) has small+large emission
        # Path (1,3,0)->(1,2,0) has large transition penalty (disconnected)
        self.assertEqual(result.edges[0].tolist(), [0, 1, 0])
        self.assertEqual(result.edges[1].tolist(), [1, 2, 0])


class TestViterbiCoherence(unittest.TestCase):
    """Viterbi must select a coherent path rather than independent nearest edges."""

    def test_path_coherence(self):
        """Three points along edge (0,1,0): nearest edge is always (0,1,0),
        but a slight detour to (1,2,0) at point 2 tests that Viterbi considers
        the full path cost."""
        # All points near (0,1,0), one also near (1,2,0)
        result = _mock_hmm(
            [0.00002, 0.00004, 0.00006],
            [0.0002, 0.0004, 0.0006],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [((0, 1, 0), 2.0)],
                [((0, 1, 0), 3.0), ((1, 2, 0), 8.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        # Viterbi should prefer staying on (0,1,0) for all three
        for i in range(3):
            self.assertEqual(result.edges[i].tolist(), [0, 1, 0])


class TestSameEdgeConsecutive(unittest.TestCase):
    """Consecutive points on the same edge should work correctly."""

    def test_same_edge(self):
        result = _mock_hmm(
            [0.00001, 0.00003],
            [0.0001, 0.0003],
            _make_candidates([
                [((0, 1, 0), 1.0)],
                [((0, 1, 0), 1.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        self.assertEqual(result.edges[0].tolist(), [0, 1, 0])
        self.assertEqual(result.edges[1].tolist(), [0, 1, 0])
        self.assertLess(result.total_cost, 10.0)


class TestDisconnectedPenalty(unittest.TestCase):
    """Disconnected transitions receive a finite large penalty."""

    def test_disconnected_finite_penalty(self):
        """Two candidates where the transition is impossible (no path).
        Cost should be finite but very large."""
        result_connected = _mock_hmm(
            [0.00002, 0.00102], [0.0005, 0.0015],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [((1, 2, 0), 2.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        result_disconnected = _mock_hmm(
            [0.00002, 0.00102], [0.0005, 0.0015],
            _make_candidates([
                [((1, 2, 0), 2.0)],
                [((0, 1, 0), 2.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        # Connected path should have much lower cost
        self.assertLess(result_connected.total_cost, result_disconnected.total_cost)


class TestZeroCandidates(unittest.TestCase):
    """Points with zero candidates get (-1,-1,-1) and zero cost."""

    def test_zero_candidate_point(self):
        result = _mock_hmm(
            [0.00002, 0.00102, 0.00202],
            [0.0005, 0.0015, 0.0025],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [],
                [((1, 2, 0), 2.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        self.assertEqual(result.edges[0].tolist(), [0, 1, 0])
        self.assertEqual(result.edges[1].tolist(), [-1, -1, -1])
        self.assertEqual(result.edges[2].tolist(), [1, 2, 0])
        self.assertEqual(result.costs[1], 0.0)


class TestEmptyTrajectory(unittest.TestCase):
    """Empty input returns empty result."""

    def test_empty(self):
        result = _mock_hmm(
            [], [],
            _make_candidates([]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        self.assertEqual(result.edges.shape, (0, 3))
        self.assertEqual(result.total_cost, 0.0)


class TestMismatchedLengths(unittest.TestCase):
    """Mismatched lat/lon lengths must raise ValueError."""

    def test_raises(self):
        with self.assertRaises(ValueError):
            _mock_hmm(
                [0.0, 0.1], [0.0],
                _make_candidates([[((0, 1, 0), 1.0)], [((1, 2, 0), 1.0)]]),
                emission_sigma_m=5.0,
                transition_sigma_m=30.0,
            )


class TestOutputShape(unittest.TestCase):
    """Output must be shape (N, 3)."""

    def test_shape(self):
        result = _mock_hmm(
            [0.00002, 0.00102], [0.0005, 0.0015],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [((1, 2, 0), 2.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        self.assertEqual(result.edges.shape, (2, 3))
        self.assertEqual(result.costs.shape, (2,))
        self.assertIsInstance(result.total_cost, float)


class TestTotalCostConsistency(unittest.TestCase):
    """total_cost must equal sum of per-point costs."""

    def test_consistency(self):
        result = _mock_hmm(
            [0.00002, 0.00102], [0.0005, 0.0015],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [((1, 2, 0), 2.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        self.assertAlmostEqual(result.total_cost, result.costs.sum(), places=6)


class TestStepCostFirstPoint(unittest.TestCase):
    """First matched point step cost must equal emission only (no transition)."""

    def test_first_point_is_emission(self):
        sigma_e = 5.0
        result = _mock_hmm(
            [0.00002], [0.0005],
            _make_candidates([
                [((0, 1, 0), 2.0)],
            ]),
            emission_sigma_m=sigma_e,
            transition_sigma_m=30.0,
        )
        expected_emission = (2.0 / sigma_e) ** 2
        self.assertAlmostEqual(result.costs[0], expected_emission, places=10)
        self.assertAlmostEqual(result.total_cost, expected_emission, places=10)


class TestStepCostLaterPoint(unittest.TestCase):
    """Later matched point step cost = emission + incoming transition."""

    def test_later_point_step_cost(self):
        sigma_e, sigma_t = 5.0, 30.0
        result = _mock_hmm(
            [0.00002, 0.00102], [0.0005, 0.0015],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [((1, 2, 0), 2.0)],
            ]),
            emission_sigma_m=sigma_e,
            transition_sigma_m=sigma_t,
        )
        emission = (2.0 / sigma_e) ** 2
        # Edge (0,1,0) -> (1,2,0) directly connected: nd = length(1,2,0) = 100
        nd = 100.0
        obs = result.costs[0]  # first cost is emission only
        # observed distance between the two points in UTM
        from unittest.mock import patch as _patch
        # We compute it the same way the function does
        import geopandas as _gpd
        from shapely.geometry import Point as _Pt
        pts = _gpd.GeoDataFrame(
            geometry=[_Pt(0.0005, 0.00002), _Pt(0.0015, 0.00102)],
            crs="EPSG:4326",
        )
        # need the UTM CRS from the mock
        edges_gdf = _make_edges_gdf()
        with _patch("src.maps.hmm_match.ox.graph_to_gdfs") as m:
            m.return_value = (None, edges_gdf)
            import osmnx as _ox
            utm = edges_gdf.estimate_utm_crs()
        pts_utm = pts.to_crs(utm)
        gap_dist = pts_utm.geometry.iloc[1].distance(pts_utm.geometry.iloc[0])
        transition = ((nd - gap_dist) / sigma_t) ** 2
        expected_step1 = emission + transition
        self.assertAlmostEqual(result.costs[1], expected_step1, places=6)
        self.assertAlmostEqual(
            result.total_cost,
            result.costs[0] + result.costs[1],
            places=6,
        )


class TestStepCostZeroCandidate(unittest.TestCase):
    """Zero-candidate point must have step cost = 0."""

    def test_zero_candidate_step_cost(self):
        result = _mock_hmm(
            [0.00002, 0.00102, 0.00202], [0.0005, 0.0015, 0.0025],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [],
                [((1, 2, 0), 2.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        self.assertEqual(result.costs[1], 0.0)
        self.assertEqual(result.edges[1].tolist(), [-1, -1, -1])


class TestStepCostTotalWithGap(unittest.TestCase):
    """total_cost == sum(costs) when a zero-candidate gap exists."""

    def test_consistency_with_gap(self):
        result = _mock_hmm(
            [0.00002, 0.00102, 0.00202], [0.0005, 0.0015, 0.0025],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [],
                [((1, 2, 0), 2.0)],
            ]),
            emission_sigma_m=5.0,
            transition_sigma_m=30.0,
        )
        self.assertAlmostEqual(result.total_cost, result.costs.sum(), places=6)
        # step[0] = emission only
        self.assertGreater(result.costs[0], 0.0)
        # step[1] = 0 (unmatched)
        self.assertEqual(result.costs[1], 0.0)
        # step[2] = emission + transition from point 0
        self.assertGreater(result.costs[2], 0.0)
        # total = step[0] + step[2] (gap contributes nothing)
        self.assertAlmostEqual(
            result.total_cost,
            result.costs[0] + result.costs[2],
            places=6,
        )


class TestViterbiPathUnchanged(unittest.TestCase):
    """Step-cost fix must not alter selected edges."""

    def test_path_unchanged_normal(self):
        r = _mock_hmm(
            [0.00002, 0.00102], [0.0005, 0.0015],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [((1, 2, 0), 2.0)],
            ]),
            emission_sigma_m=5.0, transition_sigma_m=30.0,
        )
        self.assertEqual(r.edges[0].tolist(), [0, 1, 0])
        self.assertEqual(r.edges[1].tolist(), [1, 2, 0])

    def test_path_unchanged_with_gap(self):
        r = _mock_hmm(
            [0.00002, 0.00102, 0.00202], [0.0005, 0.0015, 0.0025],
            _make_candidates([
                [((0, 1, 0), 2.0)],
                [],
                [((1, 2, 0), 2.0)],
            ]),
            emission_sigma_m=5.0, transition_sigma_m=30.0,
        )
        self.assertEqual(r.edges[0].tolist(), [0, 1, 0])
        self.assertEqual(r.edges[1].tolist(), [-1, -1, -1])
        self.assertEqual(r.edges[2].tolist(), [1, 2, 0])


if __name__ == "__main__":
    unittest.main()
