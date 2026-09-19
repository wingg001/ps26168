# PS26168 car / IO-VNBD baseline

Intelligent Dead Reckoning for **SIH 2026 Problem Statement 26168**.

This repository is the **car / IO-VNBD baseline only**. Two-wheeler data collection is deferred.

GNSS/INS fusion pipeline for **SIH 2026 Problem Statement 26168**. Fuses smartphone IMU and GNSS via a 15-state UKF with NHC, ZUPT, and CNN speed constraints. Map matching and mobile deployment are planned (see roadmap below).

## Development Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| 0 | Project setup, schema inspection | Complete |
| 1 | Preprocessing, calibration, time alignment | Complete |
| 2 | CNN speed estimation model | Complete |
| 3 | INS mechanization / dead reckoning | Complete |
| 4 | GNSS + INS fusion (UKF) | **Complete** |
| 5 | Map matching (HMM + road network) | **Complete** |
| 6 | Seamless GNSS deficit handler | Deferred |
| 7 | Mobile app (Android) | Deferred |
| 8 | Edge-deployable engine | Deferred |
| 9 | Benchmarking & screening submission | Deferred |

## Phase 4: GNSS + INS Fusion

A 15-state Scaled UKF fuses smartphone IMU (10 Hz) with GNSS position fixes. The filter uses strapdown INS mechanization for propagation and applies three constraint types during updates:

- **NHC** (Non-Holonomic Constraint): zero lateral and vertical velocity in the body frame
- **ZUPT** (Zero-Velocity Update): zero velocity when the vehicle is stationary
- **CNN speed**: forward velocity from the Phase 2 speed estimation model

GNSS updates are accepted or rejected via chi-square NIS gating (95% confidence). The GNSS noise model uses the phone's own GPS ACCURACY broadcasts (75th percentile over a calibration window, 3.0 m sigma floor) — no reference trajectory data is used at runtime.

Initialization is causal: the filter waits for two consecutive distinct GNSS fixes with >=50 m baseline before starting. This prevents future-data leakage.

**Results** on two IO-VNBD sessions with a synthetic 30 s GNSS blackout:

| Session | Blackout MAE |
|---------|-------------|
| S1 (stationary-then-drive) | 156.9 m |
| VW12 (highway) | 931.4 m |

Constraint ablation on S1 showed substantial degradation when individual constraints were removed (e.g., removing CNN increased blackout MAE from 157 m to 2514 m).

**Known limitations:** heading drift during GNSS-denied propagation remains the dominant error source; no map matching is applied; the pipeline runs offline only.

See `reports/phase4/PHASE4_HANDOFF.md` for the full technical handoff.

## Phase 5: Map Matching — Complete

HMM/Viterbi trajectory-level map matching that snaps phone-GNSS trajectory points onto the OSM road network. Uses OpenStreetMap road graphs via OSMnx, UTM-projected metre-scale distances, and a two-stage pipeline: candidate generation then Viterbi decoding.

**Components:**

- **Road graph** (`src/maps/road_graph.py`): Downloads a directed drivable-road graph from OSM via OSMnx. Nodes carry lat/lon; edges carry `length` (metres) and `geometry`.
- **Nearest-edge baseline** (`src/maps/map_match.py`): Matches each point independently to its nearest graph edge via OSMnx `nearest_edges`. No path continuity.
- **Multi-candidate generation** (`src/maps/map_match.py`): Projects both edges and points to local UTM CRS. For each trajectory point, finds all edges within a radius (default 30 m) and returns up to `max_candidates` (default 5) nearest candidates with metre-scale perpendicular distances.
- **HMM/Viterbi matcher** (`src/maps/hmm_match.py`): Selects one candidate edge per point by minimising a combined emission + transition cost using Viterbi decoding. Emission cost penalises point-to-edge distance; transition cost penalises mismatch between network travel distance and observed inter-point distance.

**Real S1 results** (530 phone-GNSS points, `radius_m=30`, `emission_sigma_m=10.0`, `transition_sigma_m=20.0`):

| Metric | Nearest-edge | HMM/Viterbi |
|--------|:-:|:-:|
| Matched points | 530/530 | 520/530 |
| Unique matched edges | 325 | 326 |
| Same-edge continuity | 25.9% | 28.4% |
| Connected transitions | 74.1% | 71.6% |
| Mean snap distance | 3.43 m | 4.82 m |
| Max snap distance | 71.13 m | 29.98 m |

The HMM trades some per-point proximity for trajectory/network consistency — its max snap distance is 30 m versus 71 m for nearest-edge. 10 points (leading, gap, and trailing) are left unmatched. No reference/V trajectory data is used as estimator input. Offline ground-truth accuracy evaluation is separate.

See `reports/phase5/PHASE5_HANDOFF.md` for the full technical handoff.

## Run Phase 0

From this directory, using Miniconda (the Microsoft Store `python` stub is not used):

```powershell
cd "C:\Users\Suraj giri\ps26168-car-baseline"
& "C:\Users\Suraj giri\miniconda3\Scripts\conda.exe" env create -f environment.yml
conda activate ps26168-car-baseline
python eval/run_phase0.py
```

If `conda activate` is not hooked in PowerShell:

```powershell
& "C:\Users\Suraj giri\miniconda3\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe eval/run_phase0.py
```

Expected outputs:

- `outputs/logs/phase0.log`
- `outputs/phase0_schema_report.json`
- `outputs/plots/phase0_gps_trace.png`
- `outputs/plots/phase0_phone_imu.png`
- `data/fixtures/V-fixture_s1.csv` and `S-fixture_s1.csv`

Override duration, GNSS-denied window, and evaluation distance without editing code:

```powershell
python eval/run_phase0.py --duration-s 180 --gnss-denied-start-s 60 --gnss-denied-duration-s 90 --evaluation-distance-m 250
```

Later, for the SIH ~1 km GNSS-denied evaluation, change `configs/default.yaml` (`run.evaluation_distance_m: 1000` and a longer `duration_s`). Do not hard-code 1 km.

## IO-VNBD data

Official source: [github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)

CSV payloads are **Git LFS** objects. The zip files on GitHub are ~134-byte LFS pointers until `git lfs pull`.

1. Install Git and Git LFS.
2. Clone into `data/raw/IO-VNBD`.
3. `git lfs pull`.
4. Set `dataset.source: raw` and `dataset.drive_id` (for example `S1`) in `configs/default.yaml`.

Until that is done, Phase 0 uses **schema-accurate fixtures**. Those files are for pipeline smoke tests only. Do not quote them as IO-VNBD accuracy.

Confirmed from real LFS objects `V-S1.csv` and `S-S1.csv` (not the paper-only names):

**Vehicle / VBOX (`V-S1.csv`, 10 Hz, 29 fields):** `No of GPS Satellites Available`, `Time Since Start of Day (seconds)`, `Latitude (degrees)`, `Longitude (degrees)`, `Velocity (km/hr)`, `Heading (degrees)`, `Height (km)`, `Vertical velocity (km/hr)`, `Sample period (seconds)`, `Steering Angle (degrees)`, four `Wheel Speed * (rad/sec)`, `Yaw Rate (deg/sec)`, `Indicated Vehicle Speed (km/hr)`, indicated long/lat accel (g), `Handbrake (0 or 1)`, gear requested/gear (typo `fof` in header), `Engine Speed (rev/min)`, coolant, clutch, brake pressure/position, battery, air temp, `Accelerator Pedal Position (0 or 1)`.

Sample rows: lat/lon already decimal degrees (`52.4017, -1.5053`); `Sample period` = 0.1 s; wheel speeds ~20 rad/s. **Height values ~110 with a `km` header are almost certainly metres.**

**Smartphone / AndroSensor (`S-S1.csv`, IMU 10 Hz, GPS held at ~1 Hz, 24 fields):** `GPS LATITUDE/LONGITUDE (degrees)`, altitude m, speed Kmh, accuracy m, orientation, `GPS SATELLITES IN RANGE` (string like `27 / 28`), `TIME SINCE START (ms)`, date, accel/gravity XYZ, gyro yaw/pitch/roll rad/s, magnetometer, orientation yaw/pitch/roll. Headers contain a space after most commas.

Author scripts index vehicle lat/lon at columns 2 and 3 (0-based). Some scripts divide by 60 (minutes). **V-S1 does not need that conversion.** The loader only warns if latitude magnitude is > 90.

## Layout

| Path | Role |
| --- | --- |
| `configs/` | Test duration, GNSS-denied interval, evaluation distance |
| `data/` | Raw IO-VNBD, fixtures, processed outputs, OSM maps |
| `preprocessing/` | Schema, loader, fixture writer, Phase 1 skeleton |
| `models/` | Reserved for Phase 2 |
| `filters/` | Reserved for UKF / map matching |
| `mobile/` | Reserved for Android app |
| `eval/` | Phase 0 runner and plots |
| `utils/` | Config + logging |
| `outputs/` | Logs, reports, figures |

## Later phases

See the Development Roadmap table above. Detailed execution plan in `Downloads/PS26168_Execution_Roadmap.md`.
