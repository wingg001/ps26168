# PS26168 – Intelligent Dead Reckoning for GNSS-Denied Vehicle Navigation

## SIH 2026 · Problem Statement 26168

A smartphone-based vehicle navigation pipeline for maintaining a navigation
estimate when GNSS becomes unavailable or unreliable in tunnels, underpasses,
parking structures, dense urban canyons, forested roads, and other GNSS-denied
environments. The system combines smartphone IMU measurements with available
GNSS fixes through a 15-state Scaled UKF, motion constraints, GNSS health
monitoring, and road-network map matching.

> **Status:** Phases 1–6 complete · Phase 7 next
> **Validation:** 183/183 tests passing

---

## The Problem

GNSS (GPS) is the backbone of civilian vehicle navigation, but it has well-known failure modes:

- **Tunnels and underpasses** — complete signal loss for tens of seconds
- **Parking structures** — multi-level environments with no sky view
- **Urban canyons** — tall buildings cause multipath and signal attenuation
- **Dense foliage** — forested roads degrade satellite visibility
- **Intentional interference** — jammers and spoofers deny reliable positioning

When GNSS drops out, a dead-reckoning system must take over. The challenge is doing this with **smartphone-grade sensors** — MEMS IMUs with significant noise, bias drift, and no tactical-grade accuracy. This project builds that system: a full GNSS/INS fusion pipeline that degrades gracefully during outages and recovers cleanly when signals return.

---

## What We Built

| Capability | Description | Status |
|-----------|-------------|--------|
| **Data loading & calibration** | IO-VNBD CSV ingestion, schema validation, sensor normalization, time alignment | Complete |
| **Speed estimation** | Phase 2 CNN-based vehicle speed model from smartphone IMU | Complete |
| **INS mechanization** | Strapdown inertial navigation propagation (position, velocity, attitude) | Complete |
| **Causal delayed initialization** | Filter waits for ≥50 m GNSS baseline before starting — no future-data leakage | Complete |
| **GNSS + INS fusion** | 15-state Scaled UKF fusing smartphone IMU (10 Hz) with GNSS position fixes | Complete |
| **NIS chi-square gating** | 95% confidence gate accepts/rejects each GNSS measurement individually | Complete |
| **NHC constraint** | Non-holonomic constraint: zero lateral and vertical body-frame velocity | Complete |
| **ZUPT constraint** | Zero-velocity update when vehicle is stationary | Complete |
| **CNN speed constraint** | Forward velocity from Phase 2 speed model | Complete |
| **GNSS health tracking** | NORMAL / DEGRADED / OUTAGE / RECOVERY state machine | Complete |
| **Adaptive Q scaling** | Process noise scaled per health state (1.0 / 2.0 / 4.0 / 1.5) | Complete |
| **GNSS recovery probing** | Accepted/rejected fixes during OUTAGE tracked for recovery | Complete |
| **No-fix timeout** | 5.0 s timeout transitions to OUTAGE when no GNSS observation arrives | Complete |
| **Road graph** | OSMnx driving-road graph download with edge lengths and geometries | Complete |
| **Map matching** | Nearest-edge baseline + multi-candidate generation + HMM/Viterbi decoder | Complete |
| **Automated tests** | 183 tests covering filters, map matching, GNSS deficit, and integration | Complete |
| Mobile / real-time deployment | Android app for real-time navigation | Next |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SENSOR INPUTS                                  │
│                                                                     │
│   Smartphone IMU (10 Hz)              GNSS fixes (~1 Hz)           │
│   ├── Accelerometer (ax, ay, az)      ├── Lat / Lon                │
│   └── Gyroscope (wx, wy, wz)          └── GPS Accuracy (m)         │
│                                                                     │
│   CNN Speed Model (Phase 2)                                       │
│   └── Forward velocity estimate                                    │
└──────────────┬──────────────────────────────┬──────────────────────┘
               │                              │
               ▼                              │
┌──────────────────────────┐                  │
│   INS MECHANIZATION      │                  │
│   Strapdown propagation  │                  │
│   Position, velocity,    │                  │
│   attitude integration   │                  │
└──────────────┬───────────┘                  │
               │                              │
               ▼                              │
┌──────────────────────────┐                  │
│   UKF STATE PREDICTION   │                  │
│   15-state scaled UKF    │                  │
│   Q scaled by GNSS       │◄─── Adaptive Q ──┤
│   health state           │     (Phase 6)    │
└──────────────┬───────────┘                  │
               │                              │
               │    ┌─────────────────────────┤
               │    │                         │
               ▼    ▼                         ▼
┌──────────────────────────────────────────────────────────┐
│                    UKF UPDATE                             │
│                                                          │
│   GNSS measurements ──────────────────► Chi-square gate  │
│   NHC (zero lateral/vert velocity) ──►  (NIS 95%)        │
│   ZUPT (zero velocity when stopped) ──►                 │
│   CNN speed constraint ────────────────►                 │
│                                                          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    ESTIMATED    │
              │   TRAJECTORY    │
              │  (ENU position) │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  MAP MATCHING   │
              │  HMM/Viterbi    │
              │  road snapping  │
              └─────────────────┘

┌──────────────────────────────────────────────────────────┐
│              GNSS HEALTH MONITOR (Phase 6)                │
│                                                          │
│   Observes NIS, GPS accuracy, acceptance history         │
│   States: NORMAL ──► DEGRADED ──► OUTAGE ──► RECOVERY    │
│                                                │         │
│   No-fix timeout (5.0s) ──► OUTAGE             │         │
│   Recovery probe ──────────────────────────────┘         │
│   Adaptive Q: 1.0 / 2.0 / 4.0 / 1.5                    │
│   NIS gate remains authoritative                         │
└──────────────────────────────────────────────────────────┘
```

---

## Fusion Pipeline

| Stage | What It Does | Status |
|-------|-------------|--------|
| Sensor loading | IO-VNBD CSV ingestion with schema validation | Complete |
| Normalization | Unit conversion, header cleanup, time-series construction | Complete |
| Time alignment | Phone IMU ↔ GNSS ↔ VBOX timestamp alignment | Complete |
| Static calibration | Zero-rate bias estimation from stationary window | Complete |
| Speed estimation | CNN model predicts vehicle speed from phone IMU | Complete |
| INS mechanization | Strapdown inertial propagation: `pos += vel·dt + 0.5·acc·dt²` | Complete |
| Causal initialization | Two consecutive distinct fixes with ≥50 m baseline before filter start | Complete |
| UKF prediction | 15-state scaled UKF time update with adaptive Q | Complete |
| GNSS + INS fusion | UKF measurement update with GNSS position | Complete |
| NIS gating | Chi-square test (95%, dof=2) accepts/rejects each GNSS fix | Complete |
| NHC | Zero lateral and vertical velocity in body frame | Complete |
| ZUPT | Zero velocity when vehicle is stationary | Complete |
| CNN speed | Forward velocity constraint from Phase 2 model | Complete |
| GNSS health tracking | State machine monitors signal quality | Complete |
| Adaptive Q | Process noise scaled by health state | Complete |
| Recovery probing | Track GNSS attempts during outage | Complete |
| No-fix timeout | 5.0 s without observation → OUTAGE | Complete |
| Map matching | Road-graph snapping via HMM/Viterbi | Complete |

The estimator does **not** use V/reference trajectory data as runtime input. All GNSS noise parameters are derived from the smartphone's own GPS ACCURACY broadcasts.

---

## Map Matching

### Road Graph

- **`build_road_graph()`** — downloads a directed driving-road graph from OpenStreetMap via OSMnx
- Nodes carry lat/lon; edges carry `length` (metres) and `geometry`
- Projected to local UTM CRS for metre-scale distances

### Nearest-Edge Baseline

- **`find_nearest_edges()`** — matches each trajectory point independently to its nearest graph edge
- **`match_trajectory_nearest()`** — batch nearest-edge matching for an entire trajectory
- No path continuity; each point is matched in isolation

### Multi-Candidate Generation

- **`find_edge_candidates()`** — projects edges and points to local UTM CRS
- For each trajectory point, finds all edges within a configurable radius (default 30 m)
- Returns up to `max_candidates` (default 5) nearest candidates with perpendicular distances

### HMM / Viterbi Matcher

- **`match_trajectory_hmm()`** — selects one candidate edge per point by minimising combined cost
- **Emission cost**: point-to-edge perpendicular distance
- **Transition cost**: mismatch between network travel distance and observed inter-point distance
- **Directed connectivity**: transitions only valid along road network direction
- **Zero-candidate handling**: points with no nearby candidates are skipped

The HMM trades some per-point proximity for **trajectory and network consistency** — its maximum snap distance is typically lower than nearest-edge, and matched paths follow road connectivity.

---

## GNSS Deficit Handling

Phase 6 adds a deterministic health-state tracker that monitors GNSS signal quality and adjusts filter behavior accordingly.

### State Machine

```
NORMAL ──poor quality──► DEGRADED ──sustained deficit──► OUTAGE
   ▲                        │                               │
   └──good quality──────────┘                               │
   ▲                                                         │
   └──sustained good──► RECOVERY ◄──first accepted fix───────┘
```

### Adaptive Process-Noise Scaling

| State | Q Scale | Effect |
|-------|:-------:|--------|
| NORMAL | 1.0 | Baseline process noise |
| DEGRADED | 2.0 | 2× — allows faster state adaptation |
| OUTAGE | 4.0 | 4× — wider uncertainty during GNSS absence |
| RECOVERY | 1.5 | 1.5× — moderate uncertainty during re-acquisition |

### Recovery Probing

When the manager is in OUTAGE, each arriving GNSS fix is tracked as a recovery probe. An accepted fix transitions to RECOVERY; a rejected fix keeps OUTAGE. The existing NIS chi-square gate remains the sole acceptance authority.

### No-Fix Timeout

When no GNSS observation (accepted or rejected) arrives for **5.0 seconds**, the manager transitions NORMAL or DEGRADED to OUTAGE via `check_timeout()`. This provides a safety net for GNSS dropout detection independent of NIS/quality history.

---

## Validation / Measured Results

### Phase 4 Baseline (pre-Phase 6)

| Metric | S1 | VW12 |
|--------|:--:|:----:|
| Blackout MAE | 156.9 m | 931.4 m |
| Test suite | 37/37 | — |

### Phase 5 — Map Matching (S1)

| Metric | Nearest-Edge | HMM/Viterbi |
|--------|:--:|:--:|
| Matched points | 530/530 | 520/530 |
| Unique edges | 325 | 326 |
| Same-edge continuity | 25.9% | 28.4% |
| Connected transitions | 74.1% | 71.6% |
| Mean snap distance | 3.43 m | 4.82 m |
| Max snap distance | 71.13 m | 29.98 m |
| Test suite | 95/95 | — |

### Phase 6 — Final (VW12)

| Metric | Value |
|--------|-------|
| Overall MAE | 532.26 m |
| Overall RMSE | 603.65 m |
| Blackout MAE | 801.97 m |
| Blackout RMSE | 805.25 m |
| Blackout MAX | 917.88 m |
| No-fix timeout events | 1 |
| Test suite | 183/183 |

The earlier adaptive-Q experiment (without the 5.0 s timeout) reduced VW12 blackout MAE from 931.41 m to 674.85 m. The final Phase 6 configuration — which adds the no-fix timeout — reports 801.97 m blackout MAE. This is not an accuracy improvement caused by the timeout; it is the measured result of the combined Phase 6 configuration operating together. The timeout provides detection coverage, not accuracy.

### S1 — Initialization Failure

S1 reports 0.00 m across all metrics. This is **not** a navigation-performance result. The filter never initialized because no qualifying GNSS baseline (≥50 m displacement between consecutive distinct fixes before the blackout period) existed in the S1 session. The estimator remained dormant throughout.

---

## Development Roadmap

| Phase | Description | Status |
|:-----:|-------------|:------:|
| 0 | Project setup, schema inspection | Complete |
| 1 | Data loading, calibration, time alignment | Complete |
| 2 | CNN speed estimation | Complete |
| 3 | INS mechanization / dead reckoning | Complete |
| 4 | GNSS + INS fusion (UKF) | Complete |
| 5 | Map matching (HMM + road network) | Complete |
| 6 | GNSS deficit handling | Complete |
| 7 | Mobile / real-time deployment | **Next** |
| 8 | Real-world validation | Deferred |
| 9 | Final system integration | Deferred |

---

## Testing

```
183 / 183 TESTS PASSED
```

| Test File | Tests | Coverage |
|-----------|:-----:|----------|
| `test_phase3.py` | 37 | UKF internals, INS mechanization, constraints, filter initialization |
| `test_road_graph.py` | 15 | Road graph construction, nearest-edge lookup |
| `test_map_match.py` | 18 | Candidate generation, nearest-edge baseline |
| `test_hmm_match.py` | 25 | HMM/Viterbi matcher, network distance, emission/transition costs |
| `test_gnss_deficit.py` | 58 | State machine transitions, Q scale, no-fix timeout, validation |
| `test_gnss_deficit_integration.py` | 30 | Adaptive Q, recovery probes, timeout integration, metrics output |

Tests verify internal correctness of filters, data pipelines, and the GNSS deficit state machine. They do not prove real-world navigation accuracy.

---

## Tech Stack

**Languages & Runtime**
- Python 3.10+

**Numerical Computing**
- NumPy — array operations, linear algebra, matrix operations
- Pandas — time-series data loading, CSV handling
- SciPy — special functions (constraint computations)

**Sensor Fusion**
- Scaled UKF — 15-state (position, velocity, attitude, accel bias, gyro bias)
- ENU local tangent plane — East/North/Up coordinate frame
- WGS84 / UTM projections — geographic coordinate transforms

**Map Matching**
- OSMnx — OpenStreetMap road graph download and projection
- NetworkX — graph algorithms, shortest path, connectivity
- GeoPandas — spatial dataframes, UTM projection, geometric operations

**Visualization**
- Matplotlib — trajectory plots, evaluation figures

**Configuration**
- YAML — pipeline parameters, filter tuning, session config

**Testing**
- Pytest — automated test suite, 183 tests

---

## Repository Layout

```
ps26168-car-baseline/
├── src/
│   ├── filters/
│   │   ├── ukf.py                # 15-state Scaled UKF
│   │   ├── mechanization.py      # Strapdown INS propagation
│   │   ├── constraints.py        # NHC, ZUPT, CNN speed
│   │   ├── coordinates.py        # ENU, WGS84, UTM transforms
│   │   └── gnss_deficit.py       # GNSS health state machine
│   ├── maps/
│   │   ├── road_graph.py         # OSMnx road graph builder
│   │   ├── map_match.py          # Nearest-edge + candidate generation
│   │   └── hmm_match.py          # HMM/Viterbi map matcher
│   ├── calibration/
│   │   └── gnss_noise.py         # GNSS noise model calibration
│   ├── phase1/
│   │   ├── loaders.py            # IO-VNBD CSV loaders
│   │   ├── calibration.py        # Static IMU calibration
│   │   ├── alignment.py          # Time alignment
│   │   ├── normalization.py      # Unit conversion
│   │   └── pipeline.py           # Phase 1 pipeline
│   └── eval/
│       └── run_phase3.py         # Navigation runner + evaluation
├── tests/
│   ├── test_phase3.py            # Filter / mechanization tests
│   ├── test_road_graph.py        # Road graph tests
│   ├── test_map_match.py         # Map matching tests
│   ├── test_hmm_match.py         # HMM matcher tests
│   ├── test_gnss_deficit.py      # GNSS deficit state machine tests
│   └── test_gnss_deficit_integration.py  # Integration tests
├── configs/
│   ├── default.yaml              # Phase 0 defaults
│   ├── phase1_S1.yaml            # Phase 1 session config
│   └── phase3_navigation.yaml    # UKF + Phase 6 config
├── reports/
│   ├── phase3/                   # Evaluation metrics (JSON/CSV)
│   ├── phase4/                   # Phase 4 handoff
│   ├── phase5/                   # Phase 5 handoff
│   └── phase6/                   # Phase 6 handoff
├── eval/
│   └── run_phase0.py             # Phase 0 runner
├── requirements.txt
└── README.md
```

---

## How to Run

**Install dependencies:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Run the full test suite:**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: `183 passed`

**Run a specific test file:**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gnss_deficit.py -q
```

**Run navigation evaluation:**

```powershell
.\.venv\Scripts\python.exe -m src.eval.run_phase3 --session vw12
.\.venv\Scripts\python.exe -m src.eval.run_phase3 --session s1
```

---

## Known Limitations

- **Offline only** — the current filter and evaluation pipeline runs offline on recorded data
- **Heading drift** — smartphone MEMS gyro heading drift is the dominant error source during GNSS-denied propagation
- **S1 initialization** — no qualifying ≥50 m pre-blackout GNSS baseline exists in the S1 session, so the filter cannot initialize
- **Map matching is offline** — Phase 5 map matching is a post-processing capability, not real-time
- **No mobile deployment** — Phase 7 (Android real-time) is not yet implemented
- **No large-scale validation** — tested on two IO-VNBD sessions; broader validation is future work
