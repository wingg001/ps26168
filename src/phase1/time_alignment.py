"""Robust time alignment between synchronized phone and vehicle streams."""
from __future__ import annotations

import numpy as np


def _finite_xy(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if len(t) < 3:
        raise ValueError("Not enough finite samples for time alignment.")
    order = np.argsort(t)
    t, y = t[order], y[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], y[keep]


def _robust_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da < 1e-9 or db < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (da * db))


def _score_offset(
    phone_t: np.ndarray,
    phone_speed_kmh: np.ndarray,
    vehicle_t: np.ndarray,
    vehicle_speed_kmh: np.ndarray,
    offset_s: float,
    grid_hz: float = 1.0,
) -> tuple[float, float, int]:
    """Score vehicle_time = phone_time + offset_s using speed correlation."""
    pt, ps = _finite_xy(phone_t, phone_speed_kmh)
    vt, vs = _finite_xy(vehicle_t, vehicle_speed_kmh)

    lo = max(pt.min(), vt.min() - offset_s)
    hi = min(pt.max(), vt.max() - offset_s)
    if hi - lo < 20.0:
        return float("nan"), float("nan"), 0

    step = 1.0 / grid_hz
    grid = np.arange(lo, hi + 0.5 * step, step)
    if len(grid) < 20:
        return float("nan"), float("nan"), 0

    p = np.interp(grid, pt, ps)
    v = np.interp(grid + offset_s, vt, vs)
    corr = _robust_corr(p, v)
    rmse = float(np.sqrt(np.mean((p - v) ** 2))) if np.isfinite(corr) else float("nan")
    return corr, rmse, len(grid)


def estimate_vehicle_time_offset(
    phone_t: np.ndarray,
    phone_speed_kmh: np.ndarray,
    vehicle_t: np.ndarray,
    vehicle_speed_kmh: np.ndarray,
    search_half_width_s: float = 60.0,
    coarse_step_s: float = 0.5,
    fine_step_s: float = 0.05,
) -> dict:
    """Estimate the constant offset linking phone and vehicle elapsed-time bases.

    Convention: ``vehicle_time = phone_time + offset_s``.
    The estimate maximizes normalized correlation between phone GPS speed and
    vehicle velocity on a 1 Hz comparison grid. The search is centered on the
    observed difference between the two time bases, so a phone elapsed-time clock
    can be aligned to the vehicle's seconds-of-day clock without assuming either
    starts at zero. A coarse-to-fine search keeps the computation small.
    """
    phone_t, phone_speed = _finite_xy(phone_t, phone_speed_kmh)
    vehicle_t, vehicle_speed = _finite_xy(vehicle_t, vehicle_speed_kmh)

    if np.std(phone_speed) < 0.5 or np.std(vehicle_speed) < 0.5:
        raise ValueError(
            "Cannot estimate V/S time offset reliably: speed variation is too small "
            "in the available recording. Use a drive/segment with actual motion."
        )

    def search(values: np.ndarray) -> tuple[float, float, float, int]:
        best = (float("nan"), -np.inf, float("nan"), 0)
        for offset in values:
            corr, rmse, n = _score_offset(
                phone_t, phone_speed, vehicle_t, vehicle_speed, float(offset)
            )
            if np.isfinite(corr) and corr > best[1]:
                best = (float(offset), corr, rmse, n)
        return best

    # Center the search on the observed clock-origin difference. This handles
    # phone elapsed time (~0 at recording start) versus vehicle seconds-of-day.
    center_offset = float(np.median(vehicle_t) - np.median(phone_t))
    coarse = np.arange(
        center_offset - search_half_width_s,
        center_offset + search_half_width_s + 0.5 * coarse_step_s,
        coarse_step_s,
    )
    best_offset, best_corr, best_rmse, n = search(coarse)
    if not np.isfinite(best_offset):
        raise ValueError("Could not find any overlapping V/S time alignment candidate.")

    fine_lo = best_offset - coarse_step_s
    fine_hi = best_offset + coarse_step_s
    fine = np.arange(fine_lo, fine_hi + 0.5 * fine_step_s, fine_step_s)
    best_offset, best_corr, best_rmse, n = search(fine)

    quality = "high" if best_corr >= 0.9 else "moderate" if best_corr >= 0.75 else "low"
    return {
        "offset_s": float(best_offset),
        "convention": "vehicle_time = phone_time + offset_s",
        "speed_correlation": float(best_corr),
        "speed_rmse_kmh": float(best_rmse),
        "comparison_grid_hz": 1.0,
        "comparison_samples": int(n),
        "quality": quality,
        "initial_clock_difference_s": center_offset,
        "search_half_width_s": float(search_half_width_s),
        "used_full_recording_for_offset": True,
    }


def interpolate_vehicle_to_phone_time(
    phone_t: np.ndarray,
    vehicle_t: np.ndarray,
    vehicle_values: np.ndarray,
    offset_s: float,
) -> np.ndarray:
    """Interpolate vehicle values at phone times using vehicle_time=phone_time+offset."""
    phone_t = np.asarray(phone_t, dtype=float)
    vehicle_t = np.asarray(vehicle_t, dtype=float)
    vehicle_values = np.asarray(vehicle_values, dtype=float)
    m = np.isfinite(vehicle_t) & np.isfinite(vehicle_values)
    if m.sum() < 2:
        raise ValueError("Not enough finite vehicle samples for interpolation.")
    vt = vehicle_t[m]
    vv = vehicle_values[m]
    order = np.argsort(vt)
    vt, vv = vt[order], vv[order]
    keep = np.concatenate(([True], np.diff(vt) > 0))
    vt, vv = vt[keep], vv[keep]
    query = phone_t + offset_s
    return np.interp(query, vt, vv, left=np.nan, right=np.nan)
