# Phase 4: GNSS + INS Fusion — Completion Handoff

**Status:** Complete  
**Date:** 2026-09-19  
**Last commit:** `6eb9e33` — Phase 3: finalize causal UKF navigation baseline  
**Tests:** 37/37 passed

---

## 1. Architecture

15-state Scaled Unscented Kalman Filter (UKF) fusing smartphone IMU data with GNSS position fixes.

**State vector:** `[pos_E, pos_N, pos_U, vel_E, vel_N, vel_U, roll, pitch, yaw, accel_bias_x, accel_bias_y, accel_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]`

**Propagation:** Strapdown INS mechanization at IMU rate (10 Hz). Bias subtraction, body-to-navigation rotation via Euler integration, gravity compensation, velocity/position double integration.

**Updates:**
- GNSS position (when accepted by chi-square gate)
- NHC: zero lateral and vertical velocity in body frame
- ZUPT: zero velocity when stationary (detected via accel variance + gyro magnitude)
- CNN speed: forward velocity from Phase 2 speed model

---

## 2. Key Design Decisions

| Decision | Implementation |
|----------|---------------|
| GNSS noise model | Smartphone-accuracy mode: 75th percentile of phone GPS ACCURACY over calibration window, sigma floor = 3.0 m |
| GNSS gating | Chi-square NIS test, 95% confidence (dof=2, threshold ~5.99) |
| Initialization | Causal delayed: waits for two consecutive distinct GNSS fixes with >=50 m baseline before starting filter |
| Reference data at runtime | None. No V-file, no reference trajectory used by estimator |
| Process noise (Q) | pos=0.1, vel=0.05, euler=0.01, accel_bias=0.001, gyro_bias=0.0001 (std dev) |
| Measurement noise (R) | NHC=[0.5, 0.2] m/s, ZUPT=0.01 m/s, CNN speed=5.58 m/s, GNSS=floor(3.0 m or calibrated) |

---

## 3. File Inventory

### Core filter (newly created for Phase 4)

| File | Lines | Role |
|------|-------|------|
| `src/filters/ukf.py` | — | Scaled UKF: sigma-point generation, predict, update, circular-angle state mean, PSD repair |
| `src/filters/mechanization.py` | — | Strapdown INS propagation |
| `src/filters/constraints.py` | — | Measurement models: hx_gnss, hx_nhc, hx_speed, is_stationary, chi_square_gate, match_cnn_speed |
| `src/filters/coordinates.py` | — | WGS84/ECEF/ENU coordinate transforms |
| `src/calibration/gnss_noise.py` | 461 | Empirical + smartphone-only GNSS noise calibration |
| `src/eval/run_phase3.py` | 899 | End-to-end session runner |
| `configs/phase3_navigation.yaml` | 38 | Filter configuration |
| `tests/test_phase3.py` | 461 | 37 unit tests |

### Phase 1 infrastructure used (not modified)

| File | Role |
|------|------|
| `src/phase1/time_alignment.py` | Phone-vehicle timestamp offset |
| `src/phase1/alignment.py` | Phone-to-vehicle frame rotation |
| `src/phase1/calibration.py` | Static IMU bias calibration |
| `src/phase1/loaders.py` | CSV loading |

---

## 4. Test Coverage

37 tests in 3 classes:

- **TestPhase3** (18): coordinates, UKF internals, measurement models, gating, noise calibration, config, path resolution
- **TestSmartInit** (10): initialization logic, course estimation, velocity/heading fallback, data-leakage audit
- **TestSmartphoneAccuracy** (7): smartphone-only noise calibration, floor binding, invalid filtering

---

## 5. Session Results

Two IO-VNBD smartphone+vehicle sessions evaluated with synthetic 30 s GNSS blackout (t=60-90 s):

| | S1 | VW12 |
|---|---|---|
| Session type | Stationary-then-drive | Highway |
| GNSS distinct fixes | 532 | 11 |
| Blackout MAE | **156.9 m** | **931.4 m** |
| Blackout RMSE | 165.5 m | 939.1 m |
| GNSS accepts / rejects | 0 / 528 | 1 / 5 |
| NHC updates | 0 (dormant) | 780 |
| CNN updates | 0 (dormant) | 234 |
| Time sync quality | High | Fallback |

### Constraint ablation (S1 blackout MAE)

| Configuration | MAE (m) |
|---------------|---------|
| NHC + ZUPT + CNN (baseline) | 156.9 |
| NHC + ZUPT (no CNN) | 2514.0 |
| NHC + CNN (no ZUPT) | 451.6 |
| CNN only | 500.8 |
| ZUPT only | 1477.6 |

The S1 ablation showed substantial degradation when individual constraints were removed.

---

## 6. Known Limitations

1. **Heading drift during GNSS-denied propagation.** Initial heading from smartphone GNSS course is noisy. During blackout, heading drifts uncorrected, degrading velocity and position accuracy. This is the dominant error source for highway sessions (VW12).

2. **No map matching.** The fused trajectory floats in continuous ENU space with no road-network snapping. (Phase 5)

3. **Offline only.** The pipeline runs as a batch process. Real-time mobile deployment is deferred. (Phase 7)

4. **Two sessions only.** Results are from S1 and VW12. Broader validation on additional IO-VNBD drives is needed.

---

## 7. What Phase 5 Needs

- Road graph from OSM (via `osmnx` or `.osm.pbf` extract)
- HMM-based map matching (emission: INS/GNSS-to-road distance; transition: road-network path likelihood)
- Before/after plots showing raw fused trajectory vs. road-snapped result
- No modifications to the filter or estimator required
