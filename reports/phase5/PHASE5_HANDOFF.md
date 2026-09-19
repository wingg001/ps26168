# Phase 5 — Map Matching: Technical Handoff

**Status:** Complete  
**Date:** 2026-09-19  
**Test suite:** 95/95 passed  

---

## 1. Overview

Phase 5 implements trajectory-level map matching that snaps phone-GNSS trajectory points onto the OpenStreetMap road network. The pipeline has two stages: (1) per-point candidate edge generation, and (2) Viterbi path selection across the full trajectory.

No reference/VBOX trajectory data is used as estimator input. Offline ground-truth accuracy evaluation is a separate task.

---

## 2. Components

### 2.1 Road Graph (`src/maps/road_graph.py`)

- **`build_road_graph(lat_bounds, lon_bounds)`**: Downloads a directed drivable-road graph from OpenStreetMap via OSMnx `graph_from_bbox`.
- Returns a `networkx.MultiDiGraph` with nodes carrying `lat`/`lon` attributes and edges carrying `length` (metres) and `geometry` attributes.
- **`find_nearest_edges(graph, X, Y)`**: Thin wrapper around `osmnx.distance.nearest_edges` for single-point or batch nearest-edge queries.

### 2.2 Nearest-Edge Baseline (`src/maps/map_match.py`)

- **`match_trajectory_nearest(graph, latitudes, longitudes)`**: Matches each trajectory point independently to its nearest graph edge.
- No path continuity — each point is matched in isolation.
- Serves as a baseline for comparing the HMM matcher.

### 2.3 Multi-Candidate Generation (`src/maps/map_match.py`)

- **`find_edge_candidates(graph, latitudes, longitudes, *, radius_m, max_candidates)`**: For each trajectory point, finds all edges within `radius_m` metres and returns up to `max_candidates` nearest candidates.
- Both edges and points are projected to local UTM CRS via `estimate_utm_crs()` for metre-scale distance accuracy.
- Returns a `CandidateResult` containing per-point lists of `EdgeCandidate(edge, distance_m)`.

### 2.4 HMM/Viterbi Matcher (`src/maps/hmm_match.py`)

- **`match_trajectory_hmm(graph, latitudes, longitudes, candidate_result, *, emission_sigma_m, transition_sigma_m)`**: Selects one candidate edge per point by minimising total cost using Viterbi decoding.
- Returns `HMMMatchResult(edges, total_cost, costs)`.

---

## 3. Models

### 3.1 Emission Model

```
emission_cost = (distance_m / emission_sigma_m)^2
```

Gaussian penalty: closer candidate → lower cost. `distance_m` is the perpendicular distance from the trajectory point to the candidate edge, computed in local UTM (metres).

### 3.2 Transition Model

```
transition_cost = ((network_distance - observed_distance) / transition_sigma_m)^2
```

- **`network_distance`**: Shortest directed path from end-node of previous edge to start-node of current edge, plus length of current edge. Computed via BFS on the directed graph using edge `length` attributes (metres).
- **`observed_distance`**: Euclidean distance between consecutive trajectory points, computed in local UTM (metres).
- **Same edge**: `network_distance = 0` (standard HMM map-matching approximation).
- **Directly connected** (end of A == start of B): `network_distance = length(B)`.
- **Disconnected**: Large finite penalty (`1e10`), not infinity.
- **First matched point**: Only emission cost (no transition).

### 3.3 Viterbi Decoding

- Forward pass: initialises at the first candidate-bearing point (handles leading zero-candidate points). Iterates through subsequent points, computing cumulative cost = emission + transition + predecessor cost.
- Zero-candidate gaps: points with no candidates get `(-1, -1, -1)` edges and zero cost; the Viterbi path bridges over them by accumulating observed distance across the gap.
- Back-tracing: finds the best reachable candidate-bearing point (not necessarily the last point), traces back pointers to reconstruct the optimal path.
- Output: incremental step costs per point, so `total_cost == sum(costs)`.

### 3.4 Local UTM Projection

All distance calculations (emission, transition, observed) use a local UTM CRS derived from the graph edges via `estimate_utm_crs()`. This avoids Web Mercator (EPSG:3857) distortion at the cost of a per-graph projection step.

---

## 4. Real S1 Integration Results

**Trajectory:** 530 deduplicated phone-GNSS points from `reports/phase3/gnss_diag_s1.csv`.  
**Graph:** 2094 nodes, 4497 edges.  
**Configuration:** `radius_m=30`, `max_candidates=5`, `emission_sigma_m=10.0`, `transition_sigma_m=20.0`.

| Metric | Nearest-edge | HMM/Viterbi |
|--------|:--:|:--:|
| Matched points | 530/530 | 520/530 |
| Unmatched points | 0 | 10 |
| Unique matched edges | 325 | 326 |
| Edge changes | 529 | 518 |
| Same-edge continuity | 25.9% | 28.4% |
| Connected transitions | 74.1% | 71.6% |
| Disconnected transitions | 0 | 0 |
| Mean snap distance | 3.43 m | 4.82 m |
| Median snap distance | 2.08 m | 2.91 m |
| Max snap distance | 71.13 m | 29.98 m |
| Total HMM cost | N/A | 20170.45 |
| `total_cost == sum(costs)` | N/A | True |

### Interpretation

The HMM trades some per-point proximity for trajectory and network consistency. Its mean snap distance is slightly higher than nearest-edge (4.82 m vs 3.43 m) because the Viterbi path constraint occasionally selects a connected-but-slightly-farther edge over a closer-but-isolated one. However, the HMM's maximum snap distance is substantially lower (29.98 m vs 71.13 m), indicating that the path consistency constraint prevents the worst-case single-point outliers that affect nearest-edge matching.

Both methods produce zero disconnected transitions — all transitions follow directed road connectivity. The 10 unmatched HMM points consist of the trajectory's leading point (index 0), a mid-trajectory gap (indices 336-340), and trailing points (indices 526-529). These are genuine zero-candidate points where no road edge was found within 30 m. The cause of these gaps (e.g., open areas, GPS drift) has not been independently verified.

---

## 5. Test Suite

95 tests across four test files:

| File | Tests | Coverage |
|------|:--:|---|
| `tests/test_phase3.py` | 37 | Phase 3/4 filters, mechanization, constraints, UKF |
| `tests/test_road_graph.py` | 15 | Graph construction, nearest-edge lookup, real S1 graph |
| `tests/test_map_match.py` | 18 | Candidate generation, nearest-edge baseline, UTM projection |
| `tests/test_hmm_match.py` | 25 | Network distance, emission, transition, Viterbi, step costs, leading/trailing zeros |

All 95 tests pass. No reference/V trajectory data is used in any test or integration run.

---

## 6. Files

| File | Lines | Role |
|------|:--:|---|
| `src/maps/__init__.py` | 0 | Package init |
| `src/maps/road_graph.py` | 136 | Road graph download + nearest-edge lookup |
| `src/maps/map_match.py` | 246 | Candidate generation + nearest-edge baseline |
| `src/maps/hmm_match.py` | 347 | HMM/Viterbi trajectory matcher |
| `tests/test_road_graph.py` | ~200 | Road graph tests |
| `tests/test_map_match.py` | ~250 | Map match tests |
| `tests/test_hmm_match.py` | ~550 | HMM matcher tests |

---

## 7. Limitations and Future Work

- **Offline only**: The pipeline runs offline with pre-downloaded OSM graphs. Real-time mobile deployment requires local graph storage (Phase 8).
- **No GNSS/INS fusion integration**: The current implementation matches phone-GNSS positions, not UKF-fused positions. Integrating the Phase 4 filter output is future work.
- **Single session validated**: Results are for S1 only. Cross-session generalisation has not been tested.
- **No ground-truth accuracy**: Phase 5 establishes the matching pipeline. Quantitative accuracy against VBOX reference data is a separate evaluation task.
- **No heading/velocity constraints**: The transition model uses network distance only. Incorporating heading from the IMU or velocity from the UKF could improve matching quality.

---

## 8. Commits

```
927d631 — Phase 5: use local UTM for candidate distances
71330d0 — Phase 5: add HMM Viterbi map matching
b89f419 — Phase 5: handle leading unmatched trajectory points
```
