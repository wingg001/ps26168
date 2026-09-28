"""
Column auto-detection + schema/quality reporting for IO-VNBD V-/S- CSV files.

v2: replaced loose substring matching with tokenized, context-aware rules
after the first real-header inspection surfaced two bugs:
  1. "ACCELEROMETER X (m/s²)" wasn't matched by substring candidates like
     "accx"/"ax" because "accelerometer" + " " + "x" doesn't contain those
     as contiguous substrings once spaces are stripped.
  2. Generic short candidates risked colliding with GPS LATITUDE / LONGITUDE
     (and could in principle collide with a future "lateral"/"longitudinal"
     acceleration column). Fields are now matched on whole tokens with
     required sensor-context tokens, so "latitude"/"longitude" can only ever
     satisfy the lat/lon rules, never accel_x/accel_y.

Each logical field has one or more *tiers* of predicates, evaluated in
priority order across all columns. The first tier with any match wins (this
is how "Velocity" is preferred over "GPS SPEED"/"Wheel Speed ..." for the
vehicle's ground-truth speed field, deterministically, not by accident of
string length). Multiple matches WITHIN the same winning tier are true
ambiguity and are still flagged.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd

Predicate = Callable[[set[str]], bool]


def _tokenize(header: str) -> set[str]:
    """Lowercase word tokens, ignoring units/punctuation.
    e.g. 'ACCELEROMETER X (m/s²)' -> {'accelerometer', 'x'}
    (the bracketed unit is stripped before tokenizing, so 'm'/'s' from the
    unit are never mistaken for sensor-name tokens)."""
    main = re.split(r"[\(\[]", header, maxsplit=1)[0]  # drop unit in parens/brackets
    return set(re.findall(r"[a-z0-9]+", main.lower()))


# field -> ordered list of tiers; each tier is a predicate over the token set
_RULES: dict[str, list[Predicate]] = {
    "time": [
        lambda tok: "time" in tok,
    ],
    "accel_x": [
        lambda tok: ("accelerometer" in tok or "accel" in tok or "acc" in tok)
        and "x" in tok and "gyroscope" not in tok and "gravity" not in tok,
    ],
    "accel_y": [
        lambda tok: ("accelerometer" in tok or "accel" in tok or "acc" in tok)
        and "y" in tok and "gyroscope" not in tok and "gravity" not in tok,
    ],
    "accel_z": [
        lambda tok: ("accelerometer" in tok or "accel" in tok or "acc" in tok)
        and "z" in tok and "gyroscope" not in tok and "gravity" not in tok,
    ],
    # Gyro axis convention: roll = rotation about the longitudinal (x) axis,
    # pitch = about the lateral (y) axis, yaw = about the vertical (z) axis.
    # Standard vehicle-dynamics convention -- flagged explicitly since
    # IO-VNBD's own docs don't spell this mapping out.
    "gyro_x": [
        lambda tok: "gyroscope" in tok and "roll" in tok,
    ],
    "gyro_y": [
        lambda tok: "gyroscope" in tok and "pitch" in tok,
    ],
    "gyro_z": [
        lambda tok: "gyroscope" in tok and "yaw" in tok,
        lambda tok: "yaw" in tok and "rate" in tok,  # vehicle CAN "Yaw Rate (deg/sec)" fallback
    ],
    "gravity_x": [
        lambda tok: "gravity" in tok and "x" in tok,
    ],
    "gravity_y": [
        lambda tok: "gravity" in tok and "y" in tok,
    ],
    "gravity_z": [
        lambda tok: "gravity" in tok and "z" in tok,
    ],
    "lat": [
        lambda tok: "latitude" in tok,
    ],
    "lon": [
        lambda tok: "longitude" in tok,
    ],
    # Velocity (vehicle ground truth) is preferred over generic/GPS speed,
    # and both are preferred over any per-wheel speed channel.
    "speed": [
        lambda tok: "velocity" in tok and "vertical" not in tok,
        lambda tok: "speed" in tok and "wheel" not in tok,
    ],
    "heading": [
        lambda tok: "heading" in tok,
        lambda tok: "gps" in tok and "orientation" in tok,
        lambda tok: "orientation" in tok,
    ],
}
_FIELD_ORDER = list(_RULES.keys())

# purely informational: vehicle sensor channels we don't map into the pipeline
# yet (Phase 2+), but that are worth surfacing in the report.
_ADDITIONAL_CHANNEL_RULES: dict[str, Predicate] = {
    "wheel_speed_front_left": lambda tok: "wheel" in tok and "speed" in tok and "front" in tok and "left" in tok,
    "wheel_speed_front_right": lambda tok: "wheel" in tok and "speed" in tok and "front" in tok and "right" in tok,
    "wheel_speed_rear_left": lambda tok: "wheel" in tok and "speed" in tok and "rear" in tok and "left" in tok,
    "wheel_speed_rear_right": lambda tok: "wheel" in tok and "speed" in tok and "rear" in tok and "right" in tok,
}


def detect_columns(columns: list[str]) -> tuple[dict[str, Optional[str]], dict[str, list[str]]]:
    """
    Returns:
      mapping: logical_field -> best-matched actual column name (or None)
      ambiguous: logical_field -> list of columns tied within the winning tier (len > 1 = real ambiguity)
    """
    tokenized = {c: _tokenize(c) for c in columns}
    mapping: dict[str, Optional[str]] = {}
    ambiguous: dict[str, list[str]] = {}

    for field in _FIELD_ORDER:
        chosen = None
        for tier_predicate in _RULES[field]:
            matches = [c for c, tok in tokenized.items() if tier_predicate(tok)]
            if matches:
                matches_sorted = sorted(matches, key=len)
                chosen = matches_sorted[0]
                if len(matches) > 1:
                    ambiguous[field] = matches
                break  # this tier won; don't fall through to lower-priority tiers
        mapping[field] = chosen
    return mapping, ambiguous


def detect_additional_channels(columns: list[str]) -> dict[str, str]:
    tokenized = {c: _tokenize(c) for c in columns}
    found = {}
    for label, predicate in _ADDITIONAL_CHANNEL_RULES.items():
        matches = [c for c, tok in tokenized.items() if predicate(tok)]
        if matches:
            found[label] = matches[0]
    return found


_UNIT_RE = re.compile(r"[\(\[]([^\)\]]+)[\)\]]")


def extract_unit(header: str) -> Optional[str]:
    """Pulls the unit out of a header like 'TIME SINCE START (ms)' -> 'ms'.
    Also handles a bare trailing '°' with no parens."""
    m = _UNIT_RE.search(header)
    if m:
        return m.group(1).strip()
    if header.strip().endswith("°"):
        return "°"
    return None


def is_milliseconds_unit(unit: Optional[str]) -> bool:
    if unit is None:
        return False
    return unit.strip().lower() in ("ms", "millisecond", "milliseconds")


def convert_time_to_seconds(t: np.ndarray, unit: Optional[str]) -> tuple[np.ndarray, bool]:
    """Returns (t_seconds, was_converted)."""
    if is_milliseconds_unit(unit):
        return t / 1000.0, True
    return t, False


@dataclasses.dataclass
class SchemaReport:
    file_name: str
    n_rows: int
    columns: list[str]
    detected_mapping: dict[str, Optional[str]]
    ambiguous_fields: dict[str, list[str]]
    missing_fields: list[str]
    additional_channels: dict[str, str]
    time_unit_raw: Optional[str]
    time_converted_to_seconds: bool
    sample_rate_hz: Optional[float]
    median_dt_s: Optional[float]
    n_gaps: int
    max_gap_s: Optional[float]
    duration_s: Optional[float]
    n_nan_values: int

    def to_markdown(self) -> str:
        lines = [f"### Schema report — {self.file_name}", ""]
        lines.append(f"- Rows: {self.n_rows}")
        lines.append(f"- Columns ({len(self.columns)}): {', '.join(self.columns)}")
        if self.time_unit_raw is not None:
            conv_note = " -> converted to seconds for all stats below" if self.time_converted_to_seconds else " (already seconds)"
            lines.append(f"- Time column unit detected: '{self.time_unit_raw}'{conv_note}")
        lines.append(f"- Duration: {self.duration_s:.2f} s" if self.duration_s else "- Duration: unknown (no time column detected)")
        lines.append(f"- Sample rate: {self.sample_rate_hz:.3f} Hz (median dt = {self.median_dt_s:.4f} s)"
                      if self.sample_rate_hz else "- Sample rate: unknown")
        lines.append(f"- Timestamp gaps (> 2x median dt): {self.n_gaps}, max gap = {self.max_gap_s}")
        lines.append(f"- NaN values in detected columns: {self.n_nan_values}")
        lines.append("")
        lines.append("| logical field | detected column | status |")
        lines.append("|---|---|---|")
        for field, col in self.detected_mapping.items():
            if col is None:
                status = "MISSING"
            elif field in self.ambiguous_fields:
                status = f"AMBIGUOUS (tied candidates: {self.ambiguous_fields[field]})"
            else:
                status = "ok"
            lines.append(f"| {field} | {col} | {status} |")
        if self.additional_channels:
            lines.append("")
            lines.append("Additional detected channels (not yet wired into Phase 1, informational only):")
            for label, col in self.additional_channels.items():
                lines.append(f"- {label}: {col}")
        return "\n".join(lines)


def build_schema_report(df: pd.DataFrame, file_name: str, time_field_override: Optional[str] = None) -> SchemaReport:
    mapping, ambiguous = detect_columns(list(df.columns))
    additional = detect_additional_channels(list(df.columns))
    time_col = time_field_override or mapping.get("time")

    sample_rate_hz = median_dt = duration_s = None
    n_gaps = 0
    max_gap = None
    time_unit_raw = None
    time_converted = False

    if time_col is not None and time_col in df.columns:
        time_unit_raw = extract_unit(time_col)
        t_raw = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy()
        t, time_converted = convert_time_to_seconds(t_raw, time_unit_raw)
        if len(t) > 2:
            dt = np.diff(np.sort(t))
            dt = dt[dt > 0]
            if len(dt) > 0:
                median_dt = float(np.median(dt))
                sample_rate_hz = 1.0 / median_dt if median_dt > 0 else None
                gap_threshold = 2 * median_dt
                gaps = dt[dt > gap_threshold]
                n_gaps = int(len(gaps))
                max_gap = float(dt.max())
                duration_s = float(t.max() - t.min())

    detected_cols = [c for c in mapping.values() if c is not None]
    n_nan = int(df[detected_cols].isna().sum().sum()) if detected_cols else 0
    missing = [f for f, c in mapping.items() if c is None]

    return SchemaReport(
        file_name=file_name,
        n_rows=len(df),
        columns=list(df.columns),
        detected_mapping=mapping,
        ambiguous_fields=ambiguous,
        missing_fields=missing,
        additional_channels=additional,
        time_unit_raw=time_unit_raw,
        time_converted_to_seconds=time_converted,
        sample_rate_hz=sample_rate_hz,
        median_dt_s=median_dt,
        n_gaps=n_gaps,
        max_gap_s=max_gap,
        duration_s=duration_s,
        n_nan_values=n_nan,
    )
