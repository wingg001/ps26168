# Phase 6 — GNSS Deficit Handling: Technical Handoff

**Status:** Observation/health tracking complete; adaptive handling pending  
**Date:** 2026-09-19  
**Test suite:** 139/139 passed  
**Last commit:** `efc2b6a` — Phase 6: add GNSS deficit state manager

---

## 1. Overview

Phase 6 adds a deterministic GNSS health-state tracker that monitors signal quality in real time. The manager is integrated into the Phase 3 navigation runner as **observation only** — it does not modify UKF parameters, block GNSS updates, or change filter behavior. All existing navigation logic remains numerically identical.

The manager tracks NIS values, GPS accuracy, and acceptance/rejection history through a four-state machine: NORMAL → DEGRADED → OUTAGE → RECOVERY → NORMAL.

**This handoff covers the completed observation/health-tracking milestone only.** Adaptive process-noise handling, GNSS blocking, re-acquisition logic, and map-matching aiding remain as future Phase 6 work.

---

## 2. State Machine

### States

| State | `gnss_allowed` | Description |
|-------|:-:|---|
| NORMAL | True | GNSS quality is acceptable |
| DEGRADED | True | GNSS quality is poor but not yet critical |
| OUTAGE | False | GNSS is unavailable or unreliable |
| RECOVERY | True | GNSS is returning after an outage |

### Transitions (evaluated in order each update)

1. **RECOVERY → NORMAL**: `accept_streak >= nis_window` (sustained good GNSS)
2. **OUTAGE → RECOVERY**: First accepted GNSS fix
3. **DEGRADED → OUTAGE**: `reject_streak >= nis_window` or `elapsed >= outage_min_s`
4. **DEGRADED → NORMAL**: Fewer than half of recent samples are poor
5. **NORMAL → DEGRADED**: More than half of recent samples are poor

---

## 3. Components

### 3.1 State Manager (`src/filters/gnss_deficit.py`)

- **`GnssDeficitManager`**: Deterministic state machine with sliding-window NIS/accuracy history
- Constructor parameters: `nis_window`, `nis_threshold`, `accuracy_threshold_m`, `outage_min_s`
- Methods: `update(t_s, gnss_accepted, nis, gps_accuracy_m) → GnssState`
- Properties: `state`, `gnss_allowed`, `q_scale` (always 1.0)

### 3.2 Navigation Runner Integration (`src/eval/run_phase3.py`)

- Manager instantiated before the filter loop with Phase 6 parameters
- Each GNSS epoch feeds the manager: timestamp, acceptance status, NIS, GPS accuracy
- State counts accumulated in `gnss_state_counts` dict
- GNSS health printed in summary and saved to metrics JSON/CSV

---

## 4. Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `nis_window` | 10 | Sliding window for NIS/accuracy history |
| `nis_threshold` | 5.99 | Chi-square 95% gate for dof=2 |
| `accuracy_threshold_m` | 30.0 | Phone GPS accuracy above which sample is "poor" |
| `outage_min_s` | 3.0 | Minimum seconds in DEGRADED before OUTAGE allowed |

---

## 5. Real Session Results

### VW12 (highway session)

| Metric | Value |
|--------|-------|
| Epochs | 918 |
| GNSS events attempted | 6 |
| GNSS accepts/rejects | 1 / 5 |
| **GNSS Health NORMAL** | **181 epochs** |
| **GNSS Health DEGRADED** | **10 epochs** |
| **GNSS Health OUTAGE** | **589 epochs** |
| **GNSS Health RECOVERY** | **0 epochs** |
| Final state | outage |
| Overall MAE | 613.90 m |
| Overall RMSE | 703.68 m |
| Blackout MAE | 931.41 m |

### S1 (stationary-then-drive session)

| Metric | Value |
|--------|-------|
| Epochs | 51,746 |
| GNSS events attempted | 0 |
| GNSS Health | All zero (filter never initialized) |
| Final state | normal |

S1 shows zero GNSS health activity because the filter never initialized — no qualifying GNSS baseline existed before the blackout period.

---

## 6. Numerical Identity Verification

All navigation metrics are **identical** to the pre-integration baseline:

| Metric | VW12 Pre-Integration | VW12 Post-Integration |
|--------|:--:|:--:|
| Overall MAE | 613.90 m | 613.90 m |
| Overall RMSE | 703.68 m | 703.68 m |
| Blackout MAE | 931.41 m | 931.41 m |
| Blackout RMSE | 939.09 m | 939.09 m |
| Blackout MAX | 1164.39 m | 1164.39 m |

The integration is purely observational. No filter parameters (Q, R, gating thresholds) are modified.

---

## 7. Test Suite

139 tests across five test files:

| File | Tests | Coverage |
|------|:--:|---|
| `tests/test_phase3.py` | 37 | Phase 3/4 filters, mechanization, constraints, UKF |
| `tests/test_road_graph.py` | 15 | Graph construction, nearest-edge lookup |
| `tests/test_map_match.py` | 18 | Candidate generation, nearest-edge baseline |
| `tests/test_hmm_match.py` | 25 | HMM/Viterbi matcher, network distance, step costs |
| `tests/test_gnss_deficit.py` | 33 | GNSS deficit state machine, transitions, validation |
| `tests/test_gnss_deficit_integration.py` | 11 | Integration wiring, state counts, metrics output |

All 139 tests pass. No reference/V trajectory data is used in any test.

---

## 8. Files

| File | Lines | Role |
|------|:--:|---|
| `src/filters/gnss_deficit.py` | 299 | GNSS deficit state machine |
| `src/eval/run_phase3.py` | 935 | Navigation runner (observation-only integration) |
| `tests/test_gnss_deficit.py` | 418 | State machine unit tests |
| `tests/test_gnss_deficit_integration.py` | ~170 | Integration behavior tests |

---

## 9. Completed Milestone

The following are complete and verified:

- GNSS health state machine with NORMAL/DEGRADED/OUTAGE/RECOVERY states
- Sliding-window NIS and GPS accuracy quality assessment
- Observation-only integration into `src/eval/run_phase3.py`
- GNSS health state counts in evaluation summary and metrics output
- Numerical identity with pre-integration baseline (no filter changes)
- 139/139 tests passing across all test files

---

## 10. Remaining Phase 6 Work

The following items are explicitly **not yet implemented**:

- **Adaptive process-noise handling**: Scale UKF Q based on GNSS health state (e.g., increase process noise during DEGRADED/OUTAGE)
- **GNSS blocking during OUTAGE**: Suppress GNSS updates when manager reports `gnss_allowed == False`
- **GNSS re-acquisition handling**: Logic for transitioning from OUTAGE back to normal operation, including potential re-initialization
- **Map-matching aiding during GNSS deficit**: Use road-network constraints to maintain position accuracy when GNSS is unavailable
- **R adaptation**: Modify measurement noise R based on health state
- **Cross-session validation**: Test generalization beyond VW12 and S1

---

## 11. Commits

```
efc2b6a — Phase 6: add GNSS deficit state manager
```
