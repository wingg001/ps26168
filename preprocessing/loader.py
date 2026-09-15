"""IO-VNBD CSV loader and schema inspection skeleton."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.schema import (
    SMARTPHONE_ALIASES,
    SMARTPHONE_COLUMNS,
    SMARTPHONE_REQUIRED,
    VEHICLE_ALIASES,
    VEHICLE_COLUMNS,
    VEHICLE_REQUIRED,
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    return _NON_ALNUM.sub(" ", str(name).strip().lower()).strip()


def canonical_name(raw: str, stream_type: str) -> str:
    aliases = VEHICLE_ALIASES if stream_type == "vehicle" else SMARTPHONE_ALIASES
    parts = normalize_name(raw).split()
    while parts:
        key = " ".join(parts)
        if key in aliases:
            return aliases[key]
        parts = parts[:-1]
    return str(raw).strip()


@dataclass
class StreamReport:
    path: Path
    stream_type: str
    n_rows: int
    columns: list[str]
    mapped_columns: dict[str, str]
    missing_required: list[str]
    extra_columns: list[str]
    inferred_hz: float | None
    median_dt_s: float | None
    gap_count: int
    max_gap_s: float | None
    duration_s: float | None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "stream_type": self.stream_type,
            "n_rows": self.n_rows,
            "columns": self.columns,
            "mapped_columns": self.mapped_columns,
            "missing_required": self.missing_required,
            "extra_columns": self.extra_columns,
            "inferred_hz": self.inferred_hz,
            "median_dt_s": self.median_dt_s,
            "gap_count": self.gap_count,
            "max_gap_s": self.max_gap_s,
            "duration_s": self.duration_s,
            "notes": self.notes,
        }


def _read_csv(path: Path) -> pd.DataFrame:
    # Real IO-VNBD headers have spaces after commas.
    return pd.read_csv(path, skipinitialspace=True)


def _time_seconds(df: pd.DataFrame, stream_type: str) -> pd.Series | None:
    if stream_type == "vehicle":
        for col in ("Time since start of day",):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
    if stream_type == "smartphone":
        if "Time since start" in df.columns:
            ms = pd.to_numeric(df["Time since start"], errors="coerce")
            return ms / 1000.0
    return None


def _gap_stats(t_s: pd.Series, expected_hz: float) -> tuple[float | None, int, float | None, float | None]:
    t = t_s.dropna().to_numpy(dtype=float)
    if t.size < 2:
        return None, 0, None, None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt >= 0)]
    if dt.size == 0:
        return None, 0, None, float(t[-1] - t[0])
    median_dt = float(np.median(dt))
    gap_threshold = max(2.5 / expected_hz, 3.0 * median_dt)
    gaps = dt[dt > gap_threshold]
    duration = float(t[-1] - t[0])
    return median_dt, int(gaps.size), float(gaps.max()) if gaps.size else 0.0, duration


def inspect_dataframe(df: pd.DataFrame, path: Path, stream_type: str, expected_hz: float) -> StreamReport:
    mapped = {c: canonical_name(c, stream_type) for c in df.columns}
    renamed = df.rename(columns=mapped)
    columns = list(renamed.columns)
    expected = VEHICLE_COLUMNS if stream_type == "vehicle" else SMARTPHONE_COLUMNS
    required = VEHICLE_REQUIRED if stream_type == "vehicle" else SMARTPHONE_REQUIRED
    missing = [c for c in required if c not in renamed.columns]
    extra = [c for c in columns if c not in expected]

    notes: list[str] = []
    if stream_type == "vehicle" and "GPS Latitude" in renamed.columns:
        lat = pd.to_numeric(renamed["GPS Latitude"], errors="coerce")
        lat_med = lat.abs().median(skipna=True)
        if pd.notna(lat_med) and float(lat_med) > 90:
            notes.append(
                "Vehicle latitude magnitude > 90; some author scripts divide VBOX lat/lon by 60. "
                "V-S1.csv samples are already decimal degrees (~52.40). Do not convert unless magnitude requires it."
            )
        if "GPS Height" in renamed.columns:
            height = pd.to_numeric(renamed["GPS Height"], errors="coerce")
            h_med = height.median(skipna=True)
            if pd.notna(h_med) and 20 <= float(h_med) <= 2000:
                notes.append(
                    "Vehicle Height header says km but V-S1 values (~110) match metres. Treat as metres until proven otherwise."
                )
    if stream_type == "smartphone" and "GPS latitude" in renamed.columns:
        lat = pd.to_numeric(renamed["GPS latitude"], errors="coerce")
        n_unique = lat.nunique(dropna=True)
        if n_unique and n_unique < max(3, int(len(renamed) * 0.3)):
            notes.append(
                "Smartphone GPS appears held/repeated across IMU rows (paper: phone GPS ~1 Hz, IMU 10 Hz)."
            )

    t = _time_seconds(renamed, stream_type)
    median_dt = gap_count = max_gap = duration = inferred = None
    if t is not None:
        median_dt, gap_count, max_gap, duration = _gap_stats(t, expected_hz)
        if median_dt and median_dt > 0:
            inferred = 1.0 / median_dt

    return StreamReport(
        path=path,
        stream_type=stream_type,
        n_rows=int(len(renamed)),
        columns=list(df.columns),
        mapped_columns=mapped,
        missing_required=missing,
        extra_columns=extra,
        inferred_hz=inferred,
        median_dt_s=median_dt,
        gap_count=gap_count or 0,
        max_gap_s=max_gap,
        duration_s=duration,
        notes=notes,
    )


def load_csv(path: Path, stream_type: str, expected_hz: float = 10.0) -> tuple[pd.DataFrame, StreamReport]:
    df = _read_csv(path)
    report = inspect_dataframe(df, path, stream_type, expected_hz)
    df = df.rename(columns=report.mapped_columns)
    return df, report


def clip_duration(df: pd.DataFrame, stream_type: str, duration_s: float) -> pd.DataFrame:
    t = _time_seconds(df, stream_type)
    if t is None or duration_s is None:
        return df
    t0 = float(t.iloc[0])
    return df.loc[t <= t0 + duration_s].copy()


def find_drive_files(raw_dir: Path, drive_id: str) -> dict[str, Path]:
    """Locate V- and S- CSVs whose names contain the drive id."""
    found: dict[str, Path] = {}
    if not raw_dir.exists():
        return found
    needle = drive_id.lower().replace("v-", "").replace("s-", "")
    compact_needle = needle.replace("-", "").replace("_", "")
    for path in raw_dir.rglob("*.csv"):
        name = path.stem.lower()
        compact = name.replace("-", "").replace("_", "")
        if compact_needle not in compact:
            continue
        if name.startswith("v-") or name.startswith("v_"):
            found["vehicle"] = path
        elif name.startswith("s-") or name.startswith("s_"):
            found["smartphone"] = path
    return found
