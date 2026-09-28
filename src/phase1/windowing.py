"""
Overlapping windowing of vehicle-frame-aligned IMU data, with attached
ground truth (speed and/or GPS position at window end) where available.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class WindowedDataset:
    windows: np.ndarray          # (N, T, C) IMU tensor: C = [ax, ay, az, gx, gy, gz]
    window_start_times: np.ndarray  # (N,)
    window_end_times: np.ndarray    # (N,)
    gt_speed: np.ndarray | None      # (N,) speed at window end, if available
    gt_lat: np.ndarray | None
    gt_lon: np.ndarray | None
    sample_rate_hz: float
    window_seconds: float
    overlap_fraction: float


def make_windows(
    t: np.ndarray,
    imu: np.ndarray,           # (M, 6): ax,ay,az,gx,gy,gz already bias-corrected + frame-aligned
    sample_rate_hz: float,
    window_seconds: float,
    overlap_fraction: float,
    gt_speed: np.ndarray | None = None,
    gt_lat: np.ndarray | None = None,
    gt_lon: np.ndarray | None = None,
) -> WindowedDataset:
    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError("overlap_fraction must be in [0, 1).")

    window_len = int(round(window_seconds * sample_rate_hz))
    if window_len < 2:
        raise ValueError(f"window_seconds={window_seconds} at {sample_rate_hz} Hz gives < 2 samples/window.")
    step = max(1, int(round(window_len * (1 - overlap_fraction))))

    n = len(imu)
    starts = list(range(0, n - window_len + 1, step))
    if not starts:
        raise ValueError(
            f"Segment too short for a single window: {n} samples available, "
            f"{window_len} required. Increase segment_duration_s or reduce window_seconds."
        )

    windows = np.stack([imu[s:s + window_len] for s in starts], axis=0)
    start_times = np.array([t[s] for s in starts])
    end_times = np.array([t[s + window_len - 1] for s in starts])

    def _gt_at_end(arr):
        if arr is None:
            return None
        return np.array([arr[s + window_len - 1] for s in starts])

    return WindowedDataset(
        windows=windows,
        window_start_times=start_times,
        window_end_times=end_times,
        gt_speed=_gt_at_end(gt_speed),
        gt_lat=_gt_at_end(gt_lat),
        gt_lon=_gt_at_end(gt_lon),
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        overlap_fraction=overlap_fraction,
    )


def chronological_train_val_split(ds: WindowedDataset, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (train_idx, val_idx) split by time order (no shuffling) so
    normalization stats are fit on 'earlier' data only -- avoids leakage from
    validation windows into training statistics."""
    n = ds.windows.shape[0]
    n_train = max(1, int(round(n * train_fraction)))
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n)
    return train_idx, val_idx
