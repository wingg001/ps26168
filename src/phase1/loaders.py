"""
Robust loading of IO-VNBD V-/S- CSV files.

We never modify the raw CSVs on disk (read-only open, no in-place writes).
Encoding fallback is included because the IO-VNBD paper/tooling notes latin1
degree-symbol characters (e.g. "m/s\u00b2") appearing in some headers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema_utils import convert_time_to_seconds, extract_unit


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw IO-VNBD file not found: {path}")

    last_err = None
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not decode {path} with utf-8/latin1/cp1252") from last_err


def get_time_seconds(df: pd.DataFrame, time_col: str) -> "pd.Series":
    """Returns the time column converted to seconds, auto-detecting ms vs s
    from the column header (e.g. 'TIME SINCE START (ms)' -> divide by 1000).
    This is the single place ms->s conversion happens, so slicing/windowing/
    sample-rate code downstream never has to guess the unit again."""
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found.")
    t_raw = pd.to_numeric(df[time_col], errors="coerce")
    unit = extract_unit(time_col)
    t_seconds, _converted = convert_time_to_seconds(t_raw.to_numpy(), unit)
    return pd.Series(t_seconds, index=df.index, name=f"{time_col}__seconds")


def slice_segment(df: pd.DataFrame, time_col: str, start_s: float, duration_s: float | None) -> pd.DataFrame:
    """Take a short configurable segment of the drive, e.g. the first few
    minutes. start_s/duration_s are always in SECONDS regardless of the raw
    column's native unit (ms is auto-converted via get_time_seconds)."""
    t = get_time_seconds(df, time_col)
    t0 = t.min()
    lo = t0 + start_s
    hi = t0 + start_s + duration_s if duration_s is not None else t.max()
    mask = (t >= lo) & (t <= hi)
    out = df.loc[mask].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(
            f"Segment [{start_s}, {start_s + (duration_s or 0)}] s produced 0 rows. "
            f"File covers t in [{t0}, {t.max()}] seconds (after unit conversion of '{time_col}')."
        )
    return out

