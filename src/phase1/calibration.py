"""
Static accelerometer / gyroscope bias calibration.

Method: take a window of samples where the vehicle is known (or detected) to
be stationary. Gyro bias = mean angular rate over that window (should be ~0
rad/s at rest). Accel bias is computed only on X/Y (lateral/longitudinal):
at rest these should read ~0 once gravity is projected onto Z during frame
alignment, so we treat their static mean as bias. The Z axis is NOT bias-
corrected against 0 here because it also carries the ~1g gravity component
used later for tilt alignment.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .loaders import get_time_seconds


@dataclasses.dataclass
class CalibrationResult:
    window_start_s: float
    window_end_s: float
    n_samples: int
    accel_bias: np.ndarray  # shape (3,) -> [x, y, z] bias, z bias reported but not applied to z
    gyro_bias: np.ndarray   # shape (3,)
    accel_static_std: np.ndarray
    gyro_static_std: np.ndarray
    static_ok: bool         # whether the window actually looks stationary
    gravity_vector: np.ndarray  # mean [ax, ay, az] over the static window (used for tilt alignment)


def _find_lowest_variance_window(t: np.ndarray, accel: np.ndarray, window_s: float, search_end_s: float) -> tuple[float, float]:
    """Slide a window_s-wide window over the first search_end_s seconds and
    return the [start, end] with lowest accel variance (best guess at 'static')."""
    t0 = t[0]
    best = (t0, t0 + window_s)
    best_var = np.inf
    step = max(window_s / 4.0, 0.25)
    start = t0
    while start + window_s <= t0 + search_end_s and start + window_s <= t[-1]:
        mask = (t >= start) & (t < start + window_s)
        if mask.sum() >= 5:
            var = np.var(accel[mask], axis=0).sum()
            if var < best_var:
                best_var = var
                best = (start, start + window_s)
        start += step
    return best


def static_bias_calibration(
    df: pd.DataFrame,
    time_col: str,
    accel_cols: tuple[str, str, str],
    gyro_cols: tuple[str, str, str],
    window_start_s: float | None,
    window_duration_s: float,
    auto_detect: bool = True,
    static_std_threshold: float = 0.5,
) -> CalibrationResult:
    # Always work in seconds; phone TIME SINCE START is stored in milliseconds.
    t = get_time_seconds(df, time_col).to_numpy()
    accel = df[list(accel_cols)].to_numpy(dtype=float)
    gyro = df[list(gyro_cols)].to_numpy(dtype=float)

    t0 = float(t[0])
    if window_start_s is None:
        window_start_s = 0.0
    start = t0 + window_start_s
    end = start + window_duration_s

    mask = (t >= start) & (t < end)
    if mask.sum() < 5 and auto_detect:
        start, end = _find_lowest_variance_window(t, accel, window_duration_s, search_end_s=min(30.0, t[-1] - t0))
        mask = (t >= start) & (t < end)

    if mask.sum() < 5:
        raise ValueError(
            "Could not find a usable static window for calibration "
            f"(found only {int(mask.sum())} samples). Adjust static_window_start_s / "
            "static_window_duration_s in the config, or check the time column units."
        )

    accel_win = accel[mask]
    gyro_win = gyro[mask]

    gyro_bias = gyro_win.mean(axis=0)
    accel_static_std = accel_win.std(axis=0)
    gyro_static_std = gyro_win.std(axis=0)

    # x/y accel should read ~0 at rest -> their mean is bias
    # z accel mean is gravity, kept separately (not zeroed here)
    accel_bias = np.array([accel_win[:, 0].mean(), accel_win[:, 1].mean(), 0.0])
    gravity_vector = accel_win.mean(axis=0)

    static_ok = bool(np.all(accel_static_std < static_std_threshold))

    if auto_detect and not static_ok:
        # one retry: search for a genuinely low-variance window before giving up
        start2, end2 = _find_lowest_variance_window(t, accel, window_duration_s, search_end_s=min(60.0, t[-1] - t0))
        mask2 = (t >= start2) & (t < end2)
        if mask2.sum() >= 5:
            accel_win2 = accel[mask2]
            gyro_win2 = gyro[mask2]
            if np.all(accel_win2.std(axis=0) < static_std_threshold):
                start, end = start2, end2
                mask = mask2
                accel_win, gyro_win = accel_win2, gyro_win2
                gyro_bias = gyro_win.mean(axis=0)
                accel_static_std = accel_win.std(axis=0)
                gyro_static_std = gyro_win.std(axis=0)
                accel_bias = np.array([accel_win[:, 0].mean(), accel_win[:, 1].mean(), 0.0])
                gravity_vector = accel_win.mean(axis=0)
                static_ok = True

    return CalibrationResult(
        window_start_s=float(start),
        window_end_s=float(end),
        n_samples=int(mask.sum()),
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
        accel_static_std=accel_static_std,
        gyro_static_std=gyro_static_std,
        static_ok=static_ok,
        gravity_vector=gravity_vector,
    )
