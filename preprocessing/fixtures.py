"""Write schema-accurate IO-VNBD-like CSVs for Phase 0 smoke tests.

These rows are a pipeline fixture, not a substitute for the real dataset and
must not be used to report SIH accuracy or drift.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.schema import SMARTPHONE_COLUMNS, VEHICLE_COLUMNS


def write_fixture(fixture_dir: Path, duration_s: float = 180.0, hz: float = 10.0) -> dict[str, Path]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * hz)
    t = np.arange(n, dtype=float) / hz
    dt = 1.0 / hz

    # Slow eastbound motion near Coventry (IO-VNBD collection region), ~8 m/s.
    lat0, lon0 = 52.4068, -1.5197
    speed_mps = 8.0
    dlat = (speed_mps * t) / 111_320.0
    dlon = np.zeros_like(t)

    vehicle = pd.DataFrame(0.0, index=range(n), columns=VEHICLE_COLUMNS)
    vehicle["No of GPS satellites available"] = 10
    vehicle["Time since start of day"] = 12 * 3600 + t
    vehicle["GPS Latitude"] = lat0 + dlat
    vehicle["GPS Longitude"] = lon0 + dlon
    vehicle["GPS Velocity"] = speed_mps * 3.6
    vehicle["GPS Heading"] = 90.0
    vehicle["GPS Height"] = 0.08
    vehicle["GPS Vertical velocity"] = 0.0
    vehicle["Sample period"] = dt
    vehicle["Steering angle"] = 0.0
    vehicle["Wheel speed front left"] = speed_mps / 0.3
    vehicle["Wheel speed front right"] = speed_mps / 0.3
    vehicle["Wheel speed rear left"] = speed_mps / 0.3
    vehicle["Wheel speed rear right"] = speed_mps / 0.3
    vehicle["Yaw rate"] = 0.0
    vehicle["Indicated vehicle speed"] = speed_mps * 3.6
    vehicle["Indicated longitudinal acceleration"] = 0.0
    vehicle["Indicated lateral acceleration"] = 0.0
    vehicle["Handbrake activated or not"] = 0
    vehicle["Gear requested"] = 3
    vehicle["Gear number"] = 3
    vehicle["Engine speed"] = 1800
    vehicle["Coolant temperature"] = 90
    vehicle["Clutch position"] = 0
    vehicle["Brake pressure"] = 0
    vehicle["Brake position"] = 0
    vehicle["Battery voltage"] = 12.4
    vehicle["Air temperature"] = 15
    vehicle["Accelerator pedal position"] = 20

    start = datetime(2019, 9, 8, 12, 0, 0)
    dates = [(start + timedelta(milliseconds=int(x * 1000))).strftime("%Y-%m-%d %H-%M-%S_%f")[:-3] for x in t]

    phone = pd.DataFrame(0.0, index=range(n), columns=SMARTPHONE_COLUMNS)
    gps_hold = (np.floor(t)).astype(int)
    phone["GPS latitude"] = lat0 + (speed_mps * gps_hold) / 111_320.0
    phone["GPS longitude"] = lon0
    phone["GPS altitude"] = 80.0
    phone["GPS speed"] = speed_mps * 3.6
    phone["GPS accuracy"] = 4.0
    phone["GPS orientation"] = 90.0
    phone["GPS satellites In range"] = 12
    phone["Time since start"] = (t * 1000).astype(int)
    phone["Date"] = dates
    phone["Accelerometer X"] = 0.15 * np.sin(2 * np.pi * t)
    phone["Accelerometer Y"] = 0.05 * np.cos(2 * np.pi * t)
    phone["Accelerometer Z"] = 9.81
    phone["Gravity X"] = 0.0
    phone["Gravity Y"] = 0.0
    phone["Gravity Z"] = 9.81
    phone["Gyroscope (Yaw)"] = 0.01 * np.sin(0.2 * t)
    phone["Gyroscope (Pitch)"] = 0.0
    phone["Gyroscope (Roll)"] = 0.0
    phone["Magnetic field X"] = 20.0
    phone["Magnetic field Y"] = 5.0
    phone["Magnetic field Z"] = -40.0
    phone["Orientation (Yaw)"] = 90.0
    phone["Orientation (Pitch)"] = 0.0
    phone["Orientation (Roll)"] = 0.0

    v_path = fixture_dir / "V-fixture_s1.csv"
    s_path = fixture_dir / "S-fixture_s1.csv"
    vehicle.to_csv(v_path, index=False)
    phone.to_csv(s_path, index=False)
    readme = fixture_dir / "README.md"
    readme.write_text(
        "Schema-accurate smoke-test CSVs only. Not IO-VNBD recordings. "
        "Do not report SIH drift or accuracy from these files.\n",
        encoding="utf-8",
    )
    return {"vehicle": v_path, "smartphone": s_path}
