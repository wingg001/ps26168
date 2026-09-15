# PS26168 car / IO-VNBD baseline

Intelligent Dead Reckoning for **SIH 2026 Problem Statement 26168**.

This repository is the **car / IO-VNBD baseline only**. Two-wheeler data collection is deferred.

Phase 0 (this checkout) is setup: layout, config, logging, dataset loading, preprocessing skeleton, and evaluation/visualization skeleton. It does **not** run dead reckoning, UKF, or ML yet, and it does **not** invent accuracy numbers.

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

Follow `Downloads/PS26168_Execution_Roadmap.md`. Do not start Phase 1 until this Phase 0 command completes without errors.
