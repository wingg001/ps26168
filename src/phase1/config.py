"""
Phase 1 configuration.

Loads a small YAML file (configs/phase1_S1.yaml) that holds:
  - input file paths (V- and S- csv)
  - the confirmed column mapping (filled in after running inspect_schema.py)
  - windowing / calibration / segment parameters

Keeping this as plain YAML (not hardcoded in Python) means re-running Phase 1
on a different drive (S2, S3, ...) is just a new config file, not a code change.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import yaml


@dataclasses.dataclass
class ColumnMapping:
    """Logical field -> actual CSV column name. None = not found / not present."""
    time: Optional[str] = None
    accel_x: Optional[str] = None
    accel_y: Optional[str] = None
    accel_z: Optional[str] = None
    gyro_x: Optional[str] = None
    gyro_y: Optional[str] = None
    gyro_z: Optional[str] = None
    gravity_x: Optional[str] = None
    gravity_y: Optional[str] = None
    gravity_z: Optional[str] = None
    lat: Optional[str] = None
    lon: Optional[str] = None
    speed: Optional[str] = None
    heading: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "ColumnMapping":
        d = d or {}
        return cls(**{f.name: d.get(f.name) for f in dataclasses.fields(cls)})

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Phase1Config:
    # --- inputs ---
    v_csv: str                      # path to V-S<n>.csv (vehicle CAN + GPS)
    s_csv: str                      # path to S-S<n>.csv (phone IMU + GPS)
    drive_id: str = "S1"

    # --- short-segment test control (few minutes, not full drive) ---
    segment_start_s: float = 0.0
    segment_duration_s: Optional[float] = 180.0   # None = use full file

    # --- static calibration ---
    static_window_start_s: Optional[float] = 0.0
    static_window_duration_s: float = 5.0
    auto_detect_static_window: bool = True   # if the fixed window looks non-static, search for one

    # --- windowing ---
    window_seconds: float = 1.5
    overlap_fraction: float = 0.5     # 0.5 = 50% overlap

    # --- train/val split for normalization (chronological, no leakage) ---
    train_fraction: float = 0.8

    # --- V/S time-base alignment ---
    max_vs_time_offset_s: float = 60.0
    auto_clip_to_vs_overlap: bool = True

    # --- column mappings (filled after inspect_schema.py) ---
    s_columns: ColumnMapping = dataclasses.field(default_factory=ColumnMapping)
    v_columns: ColumnMapping = dataclasses.field(default_factory=ColumnMapping)

    # --- outputs ---
    output_dir: str = "data/processed/phase1"
    report_dir: str = "reports"

    @classmethod
    def load(cls, path: str | Path) -> "Phase1Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw = dict(raw)
        raw["s_columns"] = ColumnMapping.from_dict(raw.get("s_columns"))
        raw["v_columns"] = ColumnMapping.from_dict(raw.get("v_columns"))
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        d = dataclasses.asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, sort_keys=False)
