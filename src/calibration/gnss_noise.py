"""Empirical GNSS measurement-noise calibration for Phase 3.

Estimates a robust scalar horizontal GNSS sigma from residual phone-GNSS vs
synchronized vehicle-reference position samples expressed in a common local
ENU frame (same LTP/interpolation used by the filter).

Method
------
For each calibration sample ``k`` compute ENU residuals
    res_E[k] = phone_E[k] - ref_E[k]
    res_N[k] = phone_N[k] - ref_N[k]
using the same time-alignment offset and vehicle-reference interpolation that
the filter consumes (``ref_enu_p``).

Robust per-axis scale is the median absolute deviation about **zero** (i.e. of
``|res_axis|``), converted to a Gaussian sigma via the standard factor
    sigma_axis = 1.4826 * median(|res_axis|)
Taking the absolute residual about zero (rather than about the sample median)
makes the estimate robust to outliers while still reflecting any constant
phone/reference offset ("lever-arm" / datum bias) that the filter's
measurement model will encounter as innovation. A MAD about the median would
discard such an offset and under-estimate R, re-creating excessive NIS. For a
zero-mean Gaussian the two coincide; for biased residuals the about-zero MAD
scales with the observed total discrepancy.

A justified isotropic scalar is the root-mean-square of the two robust axes:
    sigma_iso  = sqrt((sigma_E**2 + sigma_N**2) / 2)
    sigma_used = max(sigma_iso, sigma_floor_m)

The filter then uses ``R_gnss = I2 * sigma_used**2``. A per-axis anisotropic
model is available in the returned stats but is not selected by default.

Calibration samples come from the configurable window ``[window_start_s,
window_start_s + window_duration_s)`` relative to session start, which by
default lies strictly before the GNSS blackout (the evaluation/test window),
so the blackout metrics are computed from samples never used for calibration.
The same-window overlap with the overall (non-blackout) error metrics is an
inherent limitation of single-session datasets and is stated in the report.

Phone GNSS files often broadcast the same fix across many 10 Hz rows (e.g. a
~1/9 Hz raw cadence repeated over ~90 rows). The repeated rows are NOT
independent measurements, so the ``distinct`` mask restricts calibration to
each unique coordinate change (one sample per genuine fix, equal weight).
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


MAD_TO_SIGMA = 1.4826


@dataclass
class GnssNoiseCalibration:
    n_samples: int
    n_distinct_fixes: int
    n_repeated_rows_skipped: int
    window_start_s: float
    window_duration_s: float
    sigma_floor_m: float
    res_e_median_m: float
    res_n_median_m: float
    res_e_mad_m: float
    res_n_mad_m: float
    sigma_e_m: float
    sigma_n_m: float
    sigma_iso_m: float
    rmse_m: float
    p95_m: float
    max_m: float
    sigma_used_m: float
    isotropic: bool = True

    @property
    def r_covariance(self) -> np.ndarray:
        return np.eye(2) * (self.sigma_used_m**2)


def _mad_about_zero(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x)))


def calibrate_gnss_noise(
    phone_e: np.ndarray,
    phone_n: np.ndarray,
    ref_e: np.ndarray,
    ref_n: np.ndarray,
    valid: np.ndarray,
    sigma_floor_m: float = 3.0,
    window_start_s: float = 0.0,
    window_duration_s: float = 30.0,
    t_rel: np.ndarray | None = None,
    distinct: np.ndarray | None = None,
    min_samples: int = 2,
) -> GnssNoiseCalibration:
    """Estimate a robust scalar horizontal GNSS sigma from ENU residuals.

    Parameters
    ----------
    phone_e, phone_n : per-phone-fix ENU from the shared LTP (metres).
    ref_e, ref_n     : vehicle reference ENU interpolated to phone times.
    valid            : boolean mask of samples usable for calibration.
    t_rel            : relative session time per sample (for window selection).
    distinct         : boolean mask marking each DISTINCT GNSS fix (first row of
        a coordinate change). Repeated 10 Hz rows of the same fix are excluded
        so each genuine fix gets equal weight and is not counted N times.
    """
    ok = np.asarray(valid, dtype=bool)
    res_e = np.asarray(phone_e, dtype=float) - np.asarray(ref_e, dtype=float)
    res_n = np.asarray(phone_n, dtype=float) - np.asarray(ref_n, dtype=float)
    # Session-level counts over every row carrying a valid GNSS coordinate.
    gnss_ok_rows = np.isfinite(np.asarray(phone_e, dtype=float)) | np.isfinite(np.asarray(phone_n, dtype=float))
    n_total_rows = int(np.sum(gnss_ok_rows))
    if distinct is not None:
        n_distinct_rows = int(np.sum(np.asarray(distinct, dtype=bool) & gnss_ok_rows))
    else:
        n_distinct_rows = n_total_rows
    if t_rel is not None:
        t_rel = np.asarray(t_rel, dtype=float)
        in_win = (t_rel >= window_start_s) & (t_rel < window_start_s + window_duration_s)
        ok = ok & in_win
    if distinct is not None:
        ok = ok & np.asarray(distinct, dtype=bool)
    ok = ok & np.isfinite(res_e) & np.isfinite(res_n)

    res_e = res_e[ok]
    res_n = res_n[ok]
    n = int(res_e.size)
    if n == 0:
        raise ValueError(
            "GNSS calibration window contains no valid samples; "
            "check calibration window and reference synchronization."
        )
    if n < min_samples:
        raise ValueError(
            f"GNSS calibration window has {n} samples (< min_samples={min_samples})."
        )

    h_err = np.hypot(res_e, res_n)

    mad_e = _mad_about_zero(res_e)
    mad_n = _mad_about_zero(res_n)
    sigma_e = MAD_TO_SIGMA * mad_e
    sigma_n = MAD_TO_SIGMA * mad_n
    sigma_iso = float(np.sqrt(0.5 * (sigma_e**2 + sigma_n**2)))
    rmse = float(np.sqrt(np.mean(res_e**2 + res_n**2)))
    p95 = float(np.percentile(h_err, 95))
    max_m = float(np.max(h_err))
    sigma_used = float(max(sigma_iso, sigma_floor_m))

    n_total = n_total_rows
    if distinct is None:
        n_distinct = n
    else:
        # Session-level distinct fix count (all finite coords, not window subset).
        n_distinct = n_distinct_rows
    n_skip = max(0, n_total - n_distinct)

    return GnssNoiseCalibration(
        n_samples=n,
        n_distinct_fixes=n_distinct,
        n_repeated_rows_skipped=n_skip,
        window_start_s=window_start_s,
        window_duration_s=window_duration_s,
        sigma_floor_m=sigma_floor_m,
        res_e_median_m=float(np.median(res_e)),
        res_n_median_m=float(np.median(res_n)),
        res_e_mad_m=mad_e,
        res_n_mad_m=mad_n,
        sigma_e_m=sigma_e,
        sigma_n_m=sigma_n,
        sigma_iso_m=sigma_iso,
        rmse_m=rmse,
        p95_m=p95,
        max_m=max_m,
        sigma_used_m=sigma_used,
        isotropic=True,
    )


def write_calibration_csv(
    out_path: str | Path,
    cal: GnssNoiseCalibration,
    phone_e: np.ndarray,
    phone_n: np.ndarray,
    ref_e: np.ndarray,
    ref_n: np.ndarray,
    gps_accuracy: np.ndarray,
    t_rel: np.ndarray,
    distinct: np.ndarray | None = None,
):
    """Write per-fix residuals and a summary for the calibration sessions.

    Repeated 10 Hz rows of the same GNSS fix are collapsed: only the first row
    of each DISTINCT fix is exported, so each fix has equal weight.
    """
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idx = np.arange(np.asarray(phone_e).size)
    if distinct is not None:
        idx = idx[np.asarray(distinct, dtype=bool)]

    res_e = (np.asarray(phone_e, dtype=float) - np.asarray(ref_e, dtype=float))[idx]
    res_n = (np.asarray(phone_n, dtype=float) - np.asarray(ref_n, dtype=float))[idx]
    h_err = np.hypot(res_e, res_n)
    df = pd.DataFrame(
        {
            "t_rel_s": np.asarray(t_rel, dtype=float)[idx],
            "phone_e_m": np.asarray(phone_e, dtype=float)[idx],
            "phone_n_m": np.asarray(phone_n, dtype=float)[idx],
            "ref_e_m": np.asarray(ref_e, dtype=float)[idx],
            "ref_n_m": np.asarray(ref_n, dtype=float)[idx],
            "res_e_m": res_e,
            "res_n_m": res_n,
            "horiz_err_m": h_err,
            "gps_accuracy_m": np.asarray(gps_accuracy, dtype=float)[idx],
        }
    )
    df.to_csv(out_path, index=False)

    summary = {
        "n_samples": cal.n_samples,
        "n_distinct_fixes": cal.n_distinct_fixes,
        "n_repeated_rows_skipped": cal.n_repeated_rows_skipped,
        "window_start_s": cal.window_start_s,
        "window_duration_s": cal.window_duration_s,
        "sigma_floor_m": cal.sigma_floor_m,
        "res_e_median_m": cal.res_e_median_m,
        "res_n_median_m": cal.res_n_median_m,
        "res_e_mad_m": cal.res_e_mad_m,
        "res_n_mad_m": cal.res_n_mad_m,
        "sigma_e_m": cal.sigma_e_m,
        "sigma_n_m": cal.sigma_n_m,
        "sigma_iso_m": cal.sigma_iso_m,
        "rmse_m": cal.rmse_m,
        "p95_m": cal.p95_m,
        "max_m": cal.max_m,
        "sigma_used_m": cal.sigma_used_m,
        "R_used": cal.r_covariance.tolist(),
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)


def format_report(cal: GnssNoiseCalibration) -> str:
    return (
        f"Empirical GNSS noise calibration (window [{cal.window_start_s:.0f}, "
        f"{cal.window_start_s + cal.window_duration_s:.0f})s, n={cal.n_samples} "
        f"distinct fixes)\n"
        f"  GNSS rows total={cal.n_distinct_fixes + cal.n_repeated_rows_skipped} | "
        f"distinct fixes={cal.n_distinct_fixes} | repeated rows skipped={cal.n_repeated_rows_skipped}\n"
        f"  residual E median={cal.res_e_median_m:.2f} m | MAD(about 0)={cal.res_e_mad_m:.2f} m\n"
        f"  residual N median={cal.res_n_median_m:.2f} m | MAD(about 0)={cal.res_n_mad_m:.2f} m\n"
        f"  robust sigma E={cal.sigma_e_m:.2f} m | sigma N={cal.sigma_n_m:.2f} m\n"
        f"  sigma_iso=sqrt((sigma_E^2+sigma_N^2)/2)={cal.sigma_iso_m:.2f} m\n"
        f"  horizontal RMSE={cal.rmse_m:.2f} m | p95={cal.p95_m:.2f} m | max={cal.max_m:.2f} m\n"
        f"  sigma_used=max(sigma_iso, floor={cal.sigma_floor_m:.2f})={cal.sigma_used_m:.2f} m\n"
        f"  R_gnss = diag(sigma_used^2, sigma_used^2) = "
        f"diag({cal.sigma_used_m**2:.1f}, {cal.sigma_used_m**2:.1f}) m^2"
    )


@dataclass
class SmartphoneAccuracyCalibration:
    """Smartphone-only horizontal GNSS uncertainty calibration result.

    ``uncertainty_scale_m`` is the 75th percentile of the valid distinct-fix
    GPS ACCURACY values inside the calibration window. It is an **empirical
    uncertainty scale** derived from the phone's own accuracy broadcasts and is
    NOT claimed to be a parametrically estimated standard deviation.
    ``runtime_scale_m`` is that scale floored by ``sigma_floor_m`` and defines
    the isotropic measurement covariance used by the filter
    (``R_gnss = I2 * runtime_scale_m**2``).

    No vehicle/reference data is used anywhere in this calibration.
    """

    n_samples: int
    n_distinct_fixes: int
    n_repeated_rows_skipped: int
    n_invalid_values_skipped: int
    window_start_s: float
    window_duration_s: float
    sigma_floor_m: float
    accuracy_median_m: float
    accuracy_min_m: float
    accuracy_max_m: float
    uncertainty_scale_m: float
    runtime_scale_m: float
    quality_cap_m: float | None = None

    @property
    def r_covariance(self) -> np.ndarray:
        return np.eye(2) * (self.runtime_scale_m**2)


def calibrate_gnss_noise_from_accuracy(
    gnss_accuracy: np.ndarray,
    valid: np.ndarray,
    sigma_floor_m: float = 3.0,
    window_start_s: float = 0.0,
    window_duration_s: float = 30.0,
    t_rel: np.ndarray | None = None,
    distinct: np.ndarray | None = None,
    accuracy_cap_m: float | None = None,
) -> SmartphoneAccuracyCalibration:
    """Estimate a smartphone-only, reference-free GNSS uncertainty scale.

    Only smartphone data is used: the phone's GPS ACCURACY broadcasts, the
    phone's own distinct-fix detection (position-change gating), and the phone
    timestamp. No vehicle/reference position, speed, or heading enters here.

    Method
    ------
    1. Select rows inside the calibration window ``[window_start_s,
       window_start_s + window_duration_s)`` that are also distinct GNSS fixes.
    2. Keep accuracy readings that are finite and strictly positive (validity).
       An optional ``accuracy_cap_m`` may be applied as a *data-quality policy*
       upper bound, excluding readings strictly greater than the cap. It is a
       policy choice, not a physical/validity bound, and is off by default.
    3. ``uncertainty_scale = percentile(A, 75)`` -- an empirical uncertainty
       scale (not claimed to be a parametric sigma).
    4. ``runtime_scale = max(uncertainty_scale, sigma_floor_m)``.
    5. ``R_gnss = I2 * runtime_scale**2``.
    """
    ok = np.asarray(valid, dtype=bool)
    acc = np.asarray(gnss_accuracy, dtype=float)

    finite_pos = np.isfinite(acc) & (acc > 0.0)

    if accuracy_cap_m is not None and float(accuracy_cap_m) > 0.0:
        quality_ok = acc <= float(accuracy_cap_m)
    else:
        quality_ok = np.ones_like(acc, dtype=bool)

    # Session-level counts over every row carrying a valid accuracy reading.
    n_total_rows = int(np.sum(finite_pos))
    if distinct is not None:
        n_distinct_rows = int(np.sum(finite_pos & np.asarray(distinct, dtype=bool)))
    else:
        n_distinct_rows = n_total_rows

    if t_rel is not None:
        t_rel = np.asarray(t_rel, dtype=float)
        in_win = (t_rel >= window_start_s) & (t_rel < window_start_s + window_duration_s)
        ok = ok & in_win
    if distinct is not None:
        ok = ok & np.asarray(distinct, dtype=bool)

    eligible = ok & finite_pos & quality_ok
    n_invalid_skipped = int(np.sum(ok & ~(finite_pos & quality_ok)))

    A = acc[eligible]
    n = int(A.size)

    if n == 0:
        uncertainty_scale = float("nan")
        runtime_scale = float(sigma_floor_m)
    else:
        uncertainty_scale = float(np.percentile(A, 75))
        runtime_scale = float(max(uncertainty_scale, sigma_floor_m))

    return SmartphoneAccuracyCalibration(
        n_samples=n,
        n_distinct_fixes=n_distinct_rows,
        n_repeated_rows_skipped=max(0, n_total_rows - n_distinct_rows),
        n_invalid_values_skipped=n_invalid_skipped,
        window_start_s=window_start_s,
        window_duration_s=window_duration_s,
        sigma_floor_m=float(sigma_floor_m),
        accuracy_median_m=float(np.median(A)) if n else float("nan"),
        accuracy_min_m=float(np.min(A)) if n else float("nan"),
        accuracy_max_m=float(np.max(A)) if n else float("nan"),
        uncertainty_scale_m=uncertainty_scale,
        runtime_scale_m=runtime_scale,
        quality_cap_m=(float(accuracy_cap_m) if accuracy_cap_m is not None else None),
    )


def format_report_accuracy(cal: SmartphoneAccuracyCalibration) -> str:
    cap_txt = f" | quality cap={cal.quality_cap_m:.1f} m" if cal.quality_cap_m is not None else ""
    return (
        f"Smartphone-only GNSS uncertainty (window [{cal.window_start_s:.0f}, "
        f"{cal.window_start_s + cal.window_duration_s:.0f})s)\n"
        f"  distinct-fix accuracy values used={cal.n_samples} | repeated rows of the same fix skipped={cal.n_repeated_rows_skipped}\n"
        f"  in-window rows excluded (invalid accuracy)={cal.n_invalid_values_skipped}{cap_txt}\n"
        f"  GPS ACCURACY: min={cal.accuracy_min_m:.2f} m | median={cal.accuracy_median_m:.2f} m | max={cal.accuracy_max_m:.2f} m\n"
        f"  empirical uncertainty scale (75th percentile)={cal.uncertainty_scale_m:.2f} m\n"
        f"  runtime scale=max(scale, floor={cal.sigma_floor_m:.2f})={cal.runtime_scale_m:.2f} m\n"
        f"  R_gnss = I2 * {cal.runtime_scale_m:.2f}^2 = diag({cal.runtime_scale_m**2:.1f}, {cal.runtime_scale_m**2:.1f}) m^2"
    )


def write_calibration_accuracy_csv(
    out_path: str | Path,
    cal: SmartphoneAccuracyCalibration,
    gnss_accuracy: np.ndarray,
    t_rel: np.ndarray,
    distinct: np.ndarray | None = None,
):
    """Write per-fix GPS ACCURACY values and a summary for the smartphone-only
    calibration. Repeated 10 Hz rows of the same GNSS fix are collapsed: only
    the first row of each DISTINCT fix is exported, so each fix has equal
    weight.
    """
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idx = np.arange(np.asarray(gnss_accuracy).size)
    if distinct is not None:
        idx = idx[np.asarray(distinct, dtype=bool)]

    acc = np.asarray(gnss_accuracy, dtype=float)[idx]
    acc_valid = np.isfinite(acc) & (acc > 0.0)
    if cal.quality_cap_m is not None:
        acc_valid &= acc <= cal.quality_cap_m

    df = pd.DataFrame(
        {
            "t_rel_s": np.asarray(t_rel, dtype=float)[idx],
            "gps_accuracy_m": acc,
            "valid_for_calibration": acc_valid,
        }
    )
    df.to_csv(out_path, index=False)

    summary = {
        "n_samples": cal.n_samples,
        "n_distinct_fixes": cal.n_distinct_fixes,
        "n_repeated_rows_skipped": cal.n_repeated_rows_skipped,
        "n_invalid_values_skipped": cal.n_invalid_values_skipped,
        "window_start_s": cal.window_start_s,
        "window_duration_s": cal.window_duration_s,
        "sigma_floor_m": cal.sigma_floor_m,
        "accuracy_median_m": cal.accuracy_median_m,
        "accuracy_min_m": cal.accuracy_min_m,
        "accuracy_max_m": cal.accuracy_max_m,
        "uncertainty_scale_m_75pct": cal.uncertainty_scale_m,
        "runtime_scale_m": cal.runtime_scale_m,
        "quality_cap_m": cal.quality_cap_m,
        "R_gnss": [
            [cal.runtime_scale_m**2, 0.0],
            [0.0, cal.runtime_scale_m**2],
        ],
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)