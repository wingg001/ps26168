# Phase 6 — GNSS Deficit Handling: Technical Handoff

**Status:** Complete
**Date:** 2026-09-19
**Test suite:** 183/183 passed
**Last commit:** `38881c8` — Phase 6: add GNSS no-fix timeout handling

---

## 1. Overview

Phase 6 adds a deterministic GNSS health-state tracker that monitors signal quality in real time. The manager is integrated into the Phase 3 navigation runner and drives adaptive process-noise scaling. All existing navigation logic (UKF, NIS gating, constraints) remains unchanged — the NIS chi-square gate remains the authoritative per-update acceptance mechanism.

The manager tracks NIS values, GPS accuracy, and acceptance/rejection history through a four-state machine: NORMAL → DEGRADED → OUTAGE → RECOVERY → NORMAL.

Five capabilities are implemented and verified:

1. GNSS health state tracking (NORMAL / DEGRADED / OUTAGE / RECOVERY)
2. Observation-only integration
3. Adaptive process-noise (Q) scaling per state
4. GNSS recovery probing
5. No-fix timeout detection

---

## 2. State Machine

### States

| State | `gnss_allowed` | Q scale | Description |
|-------|:-:|:-:|---|
| NORMAL | True | 1.0 | GNSS quality is acceptable |
| DEGRADED | True | 2.0 | GNSS quality is poor but not yet critical |
| OUTAGE | False | 4.0 | GNSS is unavailable or unreliable |
| RECOVERY | True | 1.5 | GNSS is returning after an outage |

### Transitions (evaluated in order each update)

1. **RECOVERY → NORMAL**: `accept_streak >= nis_window` (sustained good GNSS)
2. **OUTAGE → RECOVERY**: First accepted GNSS fix
3. **DEGRADED → OUTAGE**: `reject_streak >= nis_window` or `elapsed >= outage_min_s`
4. **DEGRADED → NORMAL**: Fewer than half of recent samples are poor
5. **NORMAL → DEGRADED**: More than half of recent samples are poor

### No-fix timeout (checked via `check_timeout`)

6. **NORMAL/DEGRADED → OUTAGE**: Elapsed time since the last GNSS observation exceeds `no_fix_timeout_s`. An accepted OR rejected GNSS attempt counts as activity (a measurement was received). Only absence of a GNSS measurement advances the no-fix timer.

---

## 3. Components

### 3.1 State Manager (`src/filters/gnss_deficit.py`)

- **`GnssDeficitManager`**: Deterministic state machine with sliding-window NIS/accuracy history
- Constructor parameters: `nis_window`, `nis_threshold`, `accuracy_threshold_m`, `outage_min_s`, `q_scales`, `no_fix_timeout_s`
- Methods: `update(t_s, gnss_accepted, nis, gps_accuracy_m) → GnssState`, `check_timeout(t_s) → GnssState`
- Properties: `state`, `gnss_allowed`, `q_scale` (per-state multiplier)

### 3.2 Navigation Runner Integration (`src/eval/run_phase3.py`)

- Manager instantiated before the filter loop with Phase 6 parameters
- Each GNSS epoch feeds the manager: timestamp, acceptance status, NIS, GPS accuracy
- When no GNSS fix is available, `check_timeout()` is called instead of `update()`
- UKF Q is scaled by `q_scale` before each prediction and restored after
- State counts, recovery probe counters, and no-fix timeout events accumulated in counters dict
- All Phase 6 fields saved to metrics JSON/CSV

---

## 4. Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `nis_window` | 10 | Sliding window for NIS/accuracy history |
| `nis_threshold` | 5.99 | Chi-square 95% gate for dof=2 |
| `accuracy_threshold_m` | 30.0 | Phone GPS accuracy above which sample is "poor" |
| `outage_min_s` | 3.0 | Minimum seconds in DEGRADED before OUTAGE allowed |
| `q_scales.normal` | 1.0 | Baseline process noise (always forced to 1.0) |
| `q_scales.degraded` | 2.0 | 2× baseline during degraded GNSS |
| `q_scales.outage` | 4.0 | 4× baseline during GNSS outage |
| `q_scales.recovery` | 1.5 | 1.5× baseline during recovery |
| `no_fix_timeout_s` | 5.0 | Seconds without GNSS observation before timeout → OUTAGE |

---

## 5. Real Session Results

### VW12 (highway session)

| Metric | Value |
|--------|-------|
| Epochs | 918 |
| GNSS events attempted | 6 |
| GNSS accepts/rejects | 1 / 5 |
| GNSS Health NORMAL | 51 epochs |
| GNSS Health DEGRADED | 0 epochs |
| GNSS Health OUTAGE | 729 epochs |
| GNSS Health RECOVERY | 0 epochs |
| Final state | outage |
| Adaptive Q predictions | 728 |
| Recovery probes attempted | 5 (0 accepted, 5 rejected) |
| No-fix timeout events | 1 |
| **Overall MAE** | **532.26 m** |
| **Overall RMSE** | **603.65 m** |
| **Blackout MAE** | **801.97 m** |
| **Blackout RMSE** | **805.25 m** |
| **Blackout MAX** | **917.88 m** |

### S1 (stationary-then-drive session)

| Metric | Value |
|--------|-------|
| Epochs | 51,746 |
| GNSS events attempted | 0 |
| GNSS Health | All zero (filter never initialized) |
| Final state | normal |
| Overall/RMSE/Blackout metrics | 0.00 m (not a navigation result) |

S1 shows zero metrics because the filter never initialized — no qualifying GNSS baseline (≥50 m displacement between consecutive distinct fixes before blackout) existed. The 0.00 m values are not a navigation-performance result.

---

## 6. No-Fix Timeout Behavior

The no-fix timeout provides a safety net for GNSS dropout detection independent of NIS/quality history:

- When `check_timeout()` is called and no GNSS observation has arrived for ≥ `no_fix_timeout_s` seconds, NORMAL or DEGRADED transitions to OUTAGE
- An accepted OR rejected GNSS attempt (via `update()`) refreshes the activity timestamp — a rejected measurement still counts as "GNSS was received"
- `check_timeout()` does not fabricate GNSS observations — it only advances the state machine
- Recovery from a timeout-triggered OUTAGE still requires the existing recovery-probe mechanism (accepted GNSS fix)
- The 5.0 s timeout produced 1 timeout event in VW12; this contributed to the observed 801.97 m blackout MAE as part of the combined Phase 6 configuration

---

## 7. Test Suite

183 tests across six test files:

| File | Tests | Coverage |
|------|:--:|---|
| `tests/test_phase3.py` | 37 | Phase 3/4 filters, mechanization, constraints, UKF |
| `tests/test_road_graph.py` | 15 | Graph construction, nearest-edge lookup |
| `tests/test_map_match.py` | 18 | Candidate generation, nearest-edge baseline |
| `tests/test_hmm_match.py` | 25 | HMM/Viterbi matcher, network distance, step costs |
| `tests/test_gnss_deficit.py` | 50 | State machine transitions, Q scale, no-fix timeout, validation |
| `tests/test_gnss_deficit_integration.py` | 15 | Integration wiring, adaptive Q, recovery probes, timeout integration |

All 183 tests pass. No reference/V trajectory data is used in any test.

---

## 8. Files

| File | Lines | Role |
|------|:--:|---|
| `src/filters/gnss_deficit.py` | 393 | GNSS deficit state machine with timeout |
| `src/eval/run_phase3.py` | 985 | Navigation runner (adaptive Q + timeout integration) |
| `configs/phase3_navigation.yaml` | 50 | Configuration with gnss_deficit section |
| `tests/test_gnss_deficit.py` | 560 | State machine unit tests |
| `tests/test_gnss_deficit_integration.py` | 641 | Integration behavior tests |

---

## 9. Key Design Invariants

- **NIS gate remains authoritative**: `check_timeout()` never bypasses the chi-square gate
- **No reference/V data enters the estimator**: All GNSS noise is from smartphone GPS ACCURACY broadcasts
- **No filter parameter mutation**: Q scaling is applied before each prediction and restored after; baseline Q is never modified
- **Deterministic**: No randomness, no fabrication, no time-travel
- **Observation-only**: The manager reads GNSS observations; it does not create them

---

## 10. Commits

```
38881c8 — Phase 6: add GNSS no-fix timeout handling
9bc7c12 — Phase 6: add GNSS recovery probing
<commit> — Phase 6: adaptive Q scaling
<commit> — Phase 6: observation-only integration
efc2b6a — Phase 6: add GNSS deficit state manager
```
