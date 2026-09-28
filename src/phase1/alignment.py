"""
Phone-to-vehicle frame alignment.

NOTE: I have not seen the exact wording of PS26168_Execution_Roadmap.md, so
this implements the standard two-stage approach used in inertial-odometry
literature (and consistent with what the IO-VNBD paper describes as
"gravity-vector-based correction" via the S- gravity channels). If your
roadmap specifies a different/more specific method, tell me and I'll adjust
this module only -- nothing else in the pipeline depends on the method used
here beyond "returns a 3x3 rotation matrix".

Stage 1 (tilt / roll-pitch): use the static-window gravity vector g_phone
(measured during calibration.static_bias_calibration) to compute the
rotation that maps g_phone onto the vehicle's +Z (up) axis, magnitude ~9.81.
This corrects for the phone's mounting tilt.

Stage 2 (heading / yaw): after tilt correction, estimate the vehicle's
forward axis using the horizontal acceleration direction during the
largest early acceleration event (car pulling away is the most reliable
"forward" signal available without magnetometer/heading ground truth), and
rotate about Z so that this direction maps to the vehicle's +X (forward).
If GPS heading is available in the segment, that is used instead (more
reliable than the acceleration heuristic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.filters.coordinates import LocalTangentPlane
from src.phase1.loaders import get_time_seconds


def _rotation_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ a_hat = b_hat (shortest-arc rotation)."""
    a_hat = a / (np.linalg.norm(a) + 1e-12)
    b_hat = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a_hat, b_hat)
    c = np.dot(a_hat, b_hat)
    s = np.linalg.norm(v)
    if s < 1e-8:
        # already aligned (or exactly opposite, which we don't expect for gravity)
        return np.eye(3)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R


def tilt_alignment_matrix(gravity_vector: np.ndarray, vehicle_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Rotation mapping the measured static gravity vector onto vehicle +Z."""
    g_mag = np.linalg.norm(gravity_vector)
    target = vehicle_up * g_mag
    return _rotation_from_vectors(gravity_vector, target)


def yaw_alignment_matrix(forward_vector_xy: np.ndarray) -> np.ndarray:
    """Rotation about Z mapping forward_vector_xy (in the tilt-corrected frame)
    onto vehicle +X (forward)."""
    fx, fy = forward_vector_xy
    theta = np.arctan2(fy, fx)  # angle of measured forward from vehicle +X
    c, s = np.cos(-theta), np.sin(-theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def estimate_forward_direction(
    accel_tilt_corrected: np.ndarray,
    speed: np.ndarray | None = None,
    gyro_tilt_corrected: np.ndarray | None = None,
    sustained_window: int = 10,
    min_speed_change_mps: float = 1.0,
    max_yaw_rate_dps: float = 3.0,
    min_dominance_ratio: float = 2.0,
    max_centripetal_accel_mps2: float = 0.5,
    return_details: bool = False,
) -> tuple[np.ndarray, bool] | tuple[np.ndarray, bool, dict]:
    """
    Estimate the horizontal forward direction from a sustained positive acceleration event.

    Physics-based confidence gating:
    1. Positive acceleration check:
       Only positive speed changes (speed[w] - speed[0] >= min_speed_change_mps) are used.
       Deceleration (braking) is explicitly rejected (no abs(delta_v)).
    2. Straightness / low-turn-rate check:
       Segments with significant yaw rate / heading variation are rejected.
    3. Longitudinal-acceleration dominance:
       Reject segments where lateral/centripetal acceleration contaminates the forward estimate.
    4. Confidence fallback:
       If any physical condition fails, returns (np.array([1.0, 0.0]), False) with identity yaw.
    """
    horiz = accel_tilt_corrected[:, :2]
    n = len(horiz)
    w = min(sustained_window, max(2, n // 4))

    details = {
        "confident": False,
        "rejection_reason": None,
        "signed_speed_change_mps": 0.0,
        "accel_norm_mps2": 0.0,
        "max_yaw_rate_dps": 0.0,
        "dominance_ratio": 0.0,
        "max_centripetal_accel_mps2": 0.0,
        "window_start": None,
    }

    fallback_vec = np.array([1.0, 0.0], dtype=float)

    # 1. Speed signal presence & length check
    if speed is None or len(speed) != n or n <= w:
        details["rejection_reason"] = "missing_or_insufficient_speed_data"
        if return_details:
            return fallback_vec, False, details
        return fallback_vec, False

    # Check 1: Signed speed-change check (NO abs(delta_v))
    signed_diff = speed[w:] - speed[:-w]
    best_start = int(np.argmax(signed_diff))
    max_speed_gain = float(signed_diff[best_start])
    details["signed_speed_change_mps"] = max_speed_gain
    details["window_start"] = best_start

    # If even the maximum speed gain is below threshold (or negative / deceleration):
    if max_speed_gain < min_speed_change_mps:
        if max_speed_gain <= 0.0:
            details["rejection_reason"] = f"deceleration_segment (max signed delta_v = {max_speed_gain:.2f} m/s <= 0)"
        else:
            details["rejection_reason"] = f"insufficient_positive_acceleration (signed delta_v = {max_speed_gain:.2f} m/s < {min_speed_change_mps:.2f} m/s)"
        if return_details:
            return fallback_vec, False, details
        return fallback_vec, False

    window_slice = slice(best_start, best_start + w)

    # Check 2: Straightness / low-turn-rate check
    if gyro_tilt_corrected is not None and len(gyro_tilt_corrected) == n:
        omega_z = gyro_tilt_corrected[window_slice, 2]  # rad/s around vertical axis (+Z)
        yaw_rate_dps = np.degrees(np.abs(omega_z))
        max_yaw_rate = float(np.max(yaw_rate_dps))
        details["max_yaw_rate_dps"] = max_yaw_rate

        if max_yaw_rate > max_yaw_rate_dps:
            details["rejection_reason"] = f"excessive_yaw_rate (max {max_yaw_rate:.2f} dps > {max_yaw_rate_dps:.2f} dps)"
            if return_details:
                return fallback_vec, False, details
            return fallback_vec, False

        # Centripetal acceleration in horizontal plane: a_cen = v * |omega_z|
        w_speed = speed[window_slice]
        a_cen = w_speed * np.abs(omega_z)
        max_a_cen = float(np.max(a_cen))
        details["max_centripetal_accel_mps2"] = max_a_cen
    else:
        max_a_cen = 0.0

    # Compute candidate horizontal acceleration vector
    vec = horiz[window_slice].mean(axis=0)
    a_norm = float(np.linalg.norm(vec))
    details["accel_norm_mps2"] = a_norm

    if a_norm < 1e-4:
        details["rejection_reason"] = "insufficient_acceleration_magnitude"
        if return_details:
            return fallback_vec, False, details
        return fallback_vec, False

    # Check 3: Longitudinal-acceleration dominance
    fwd_hat = vec / a_norm
    lat_hat = np.array([-fwd_hat[1], fwd_hat[0]], dtype=float)
    a_lat = horiz[window_slice] @ lat_hat
    rms_lat = float(np.sqrt(np.mean(a_lat ** 2)))
    dom_ratio = float(a_norm / (rms_lat + 1e-6))
    details["dominance_ratio"] = dom_ratio

    if dom_ratio < min_dominance_ratio:
        details["rejection_reason"] = f"insufficient_longitudinal_dominance (ratio {dom_ratio:.2f} < {min_dominance_ratio:.2f})"
        if return_details:
            return fallback_vec, False, details
        return fallback_vec, False

    if max_a_cen > max_centripetal_accel_mps2 or (max_a_cen > 0.0 and a_norm < min_dominance_ratio * max_a_cen):
        details["rejection_reason"] = f"centripetal_acceleration_contamination (a_cen {max_a_cen:.2f} m/s^2, a_fwd {a_norm:.2f} m/s^2)"
        if return_details:
            return fallback_vec, False, details
        return fallback_vec, False

    # All physical checks passed
    details["confident"] = True
    details["yaw_source"] = "GPS_ORIENTATION"
    details["rejection_reason"] = None
    if return_details:
        return vec, True, details
    return vec, True




def calibrate_mounting_yaw_gnss(
    df_phone: pd.DataFrame,
    phone_mapping: dict,
    blackout_start_s: float = 60.0,
    min_speed_mps: float = 2.0,
    min_span_m: float = 20.0,
    min_samples: int = 2,
    max_circular_spread_deg: float = 35.0,
    min_resultant_length: float = 0.85,
    return_details: bool = False,
) -> tuple[np.ndarray, bool] | tuple[np.ndarray, bool, dict]:
    """
    Calibrate phone-to-vehicle mounting yaw from smartphone GNSS course and smartphone orientation.

    Uses strictly causal pre-blackout smartphone data:
    1. Primary vehicle travel direction: GPS ORIENTATION (°)
    2. Smartphone orientation: ORIENTATION (Yaw) (°)
    3. Position-derived course is computed as a cross-check/diagnostic.
    4. Filters out stationary/crawl fixes (speed >= min_speed_mps and span >= min_span_m).
    5. Weighted circular averaging over multiple moving fixes before blackout.
    6. Confidence gating based on sample count, circular spread, and resultant length.
    """
    fallback_vec = np.array([1.0, 0.0], dtype=float)
    details = {
        "confident": False,
        "yaw_source": "identity_fallback",
        "primary_runtime_heading_source": "GPS_ORIENTATION",
        "position_course_used_as_primary": False,
        "rejection_reason": None,
        "mean_delta_deg": 0.0,
        "circ_spread_deg": 0.0,
        "R_resultant": 0.0,
        "n_samples": 0,
        "time_span_s": 0.0,
        "samples": [],
    }

    t_p = get_time_seconds(df_phone, phone_mapping["phone_time"]).to_numpy(dtype=float)
    if len(t_p) == 0:
        details["rejection_reason"] = "empty_phone_time"
        return (fallback_vec, False, details) if return_details else (fallback_vec, False)
    t_rel = t_p - t_p[0]

    col_gps_o = [c for c in df_phone.columns if "gps" in c.lower() and "orient" in c.lower()]
    col_sens_y = [c for c in df_phone.columns if "orient" in c.lower() and "yaw" in c.lower()]
    if not col_gps_o or not col_sens_y:
        details["rejection_reason"] = "missing_orientation_columns"
        return (fallback_vec, False, details) if return_details else (fallback_vec, False)

    gps_o_all = pd.to_numeric(df_phone[col_gps_o[0]], errors="coerce").to_numpy(dtype=float)
    sens_y_all = pd.to_numeric(df_phone[col_sens_y[0]], errors="coerce").to_numpy(dtype=float)

    speed_raw = pd.to_numeric(df_phone[phone_mapping["gps_speed"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    speed_mps_all = speed_raw / 3.6  # IO-VNBD GPS SPEED is in km/h

    lats = pd.to_numeric(df_phone[phone_mapping["gps_latitude"]], errors="coerce").to_numpy(dtype=float)
    lons = pd.to_numeric(df_phone[phone_mapping["gps_longitude"]], errors="coerce").to_numpy(dtype=float)

    distinct_idx = []
    prev_lat, prev_lon = np.nan, np.nan
    for k in range(len(lats)):
        if t_rel[k] >= blackout_start_s:
            break
        if not (np.isfinite(lats[k]) and np.isfinite(lons[k])):
            continue
        if np.isnan(prev_lat) or abs(lats[k] - prev_lat) > 1e-7 or abs(lons[k] - prev_lon) > 1e-7:
            distinct_idx.append(k)
            prev_lat, prev_lon = lats[k], lons[k]

    if len(distinct_idx) < 2:
        details["rejection_reason"] = f"insufficient_distinct_fixes ({len(distinct_idx)} < 2)"
        return (fallback_vec, False, details) if return_details else (fallback_vec, False)

    ltp = LocalTangentPlane(lats[distinct_idx[0]], lons[distinct_idx[0]])

    valid_samples = []
    for i in range(1, len(distinct_idx)):
        ka = distinct_idx[i - 1]
        kb = distinct_idx[i]

        enu_a = ltp.to_enu(lats[ka], lons[ka])
        enu_b = ltp.to_enu(lats[kb], lons[kb])
        de, dn = enu_b[0] - enu_a[0], enu_b[1] - enu_a[1]
        span = float(np.hypot(de, dn))
        spd = float(speed_mps_all[kb])

        # Strict moving fix filter: reject stationary/crawl (< 2.0 m/s or < 20 m span)
        if spd < min_speed_mps or span < min_span_m:
            continue

        g_o = float(gps_o_all[kb])
        s_y = float(sens_y_all[kb])
        if not (np.isfinite(g_o) and np.isfinite(s_y)):
            continue

        delta_deg = float((g_o - s_y + 180.0) % 360.0 - 180.0)
        pos_course = float(np.degrees(np.arctan2(de, dn))) % 360.0
        delta_pc = float((pos_course - s_y + 180.0) % 360.0 - 180.0)

        valid_samples.append({
            "idx": i,
            "kb": kb,
            "t_s": float(t_rel[kb]),
            "span_m": span,
            "speed_mps": spd,
            "gps_o_deg": g_o,
            "sens_y_deg": s_y,
            "delta_deg": delta_deg,
            "pos_course_deg": pos_course,
            "delta_pc_deg": delta_pc,
        })

    n_samples = len(valid_samples)
    details["n_samples"] = n_samples
    details["samples"] = valid_samples

    if n_samples < min_samples:
        details["rejection_reason"] = f"insufficient_moving_samples ({n_samples} < {min_samples})"
        return (fallback_vec, False, details) if return_details else (fallback_vec, False)

    sample_deltas = np.array([s["delta_deg"] for s in valid_samples], dtype=float)
    spans = np.array([s["span_m"] for s in valid_samples], dtype=float)
    w_init = spans / np.sum(spans)

    # Robust Circular Estimator (Huber M-estimator on circular residuals with displacement weighting)
    # Step 1: Initial weighted circular mean
    sin_sum_init = float(np.sum(w_init * np.sin(np.radians(sample_deltas))))
    cos_sum_init = float(np.sum(w_init * np.cos(np.radians(sample_deltas))))
    mu_init = float(np.degrees(np.arctan2(sin_sum_init, cos_sum_init)))

    # Step 2: Wraparound-safe angular residuals relative to initial mean
    residuals_init = np.array([
        (d - mu_init + 180.0) % 360.0 - 180.0 for d in sample_deltas
    ], dtype=float)

    # Step 3: Huber robust downweighting
    mad = float(np.median(np.abs(residuals_init)))
    k_huber = max(15.0, 2.0 * mad)
    huber_weights = np.array([
        min(1.0, k_huber / (abs(r) + 1e-6)) for r in residuals_init
    ], dtype=float)

    w_robust = w_init * huber_weights
    w_robust /= np.sum(w_robust)

    # Step 4: Robust circular mean and resultant length
    sin_sum = float(np.sum(w_robust * np.sin(np.radians(sample_deltas))))
    cos_sum = float(np.sum(w_robust * np.cos(np.radians(sample_deltas))))
    r_res = float(np.hypot(sin_sum, cos_sum))
    mean_delta = float(np.degrees(np.arctan2(sin_sum, cos_sum)))

    # Step 5: Robust circular spread
    residuals_robust = np.array([
        (d - mean_delta + 180.0) % 360.0 - 180.0 for d in sample_deltas
    ], dtype=float)
    circ_spread = float(np.sqrt(np.sum(w_robust * (residuals_robust ** 2))))
    time_span = float(valid_samples[-1]["t_s"] - valid_samples[0]["t_s"])

    details["raw_sample_angles_deg"] = sample_deltas.tolist()
    details["initial_weights"] = w_init.tolist()
    details["robust_weights"] = w_robust.tolist()
    details["weighted_sin_sum"] = sin_sum
    details["weighted_cos_sum"] = cos_sum
    details["mean_delta_deg"] = mean_delta
    details["circ_spread_deg"] = circ_spread
    details["R_resultant"] = r_res
    details["time_span_s"] = time_span

    if circ_spread > max_circular_spread_deg:
        details["rejection_reason"] = f"excessive_circular_spread ({circ_spread:.1f}° > {max_circular_spread_deg:.1f}°)"
        return (fallback_vec, False, details) if return_details else (fallback_vec, False)

    if r_res < min_resultant_length:
        details["rejection_reason"] = f"low_resultant_length ({r_res:.2f} < {min_resultant_length:.2f})"
        return (fallback_vec, False, details) if return_details else (fallback_vec, False)

    details["confident"] = True
    # Android coordinate convention: Azimuth 0° is device +Y (longitudinal forward), 90° is device +X.
    # Therefore forward unit vector in phone frame is [sin(delta), cos(delta)], which yaw_alignment_matrix
    # rotates by delta - 90° into vehicle +X forward.
    fwd_vec = np.array([np.cos(np.radians(mean_delta)), np.sin(np.radians(mean_delta))], dtype=float)
    return (fwd_vec, True, details) if return_details else (fwd_vec, True)



def build_alignment_matrix(
    gravity_vector: np.ndarray,
    accel_segment: np.ndarray | None = None,
    speed_segment: np.ndarray | None = None,
    gyro_segment: np.ndarray | None = None,
    df_phone: pd.DataFrame | None = None,
    phone_mapping: dict | None = None,
    blackout_start_s: float = 60.0,
    sustained_window: int = 10,
    min_speed_change_mps: float = 1.0,
    max_yaw_rate_dps: float = 3.0,
    min_dominance_ratio: float = 2.0,
    max_centripetal_accel_mps2: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Full phone->vehicle rotation = yaw_correction @ tilt_correction.
    
    Calibration hierarchy:
    1. Primary: GNSS course / phone orientation circular calibration before blackout.
    2. Fallback: Acceleration-based forward direction estimation (if motion segment provided).
    3. Safe default: Identity yaw fallback if confidence gates fail.
    """
    R_tilt = tilt_alignment_matrix(gravity_vector)
    yaw_confident = False
    details = {}
    forward = np.array([1.0, 0.0], dtype=float)
    method_used = "identity_fallback"

    # 1. Primary path: Pre-blackout GNSS course calibration
    if df_phone is not None and phone_mapping is not None:
        forward, yaw_confident, details = calibrate_mounting_yaw_gnss(
            df_phone=df_phone,
            phone_mapping=phone_mapping,
            blackout_start_s=blackout_start_s,
            return_details=True,
        )
        if yaw_confident:
            method_used = "GPS_ORIENTATION"

    # 2. Fallback path: Acceleration-based forward direction estimation
    if not yaw_confident and accel_segment is not None and speed_segment is not None:
        accel_tilted = accel_segment @ R_tilt.T
        gyro_tilted = gyro_segment @ R_tilt.T if gyro_segment is not None else None
        forward, yaw_confident, details = estimate_forward_direction(
            accel_tilted,
            speed=speed_segment,
            gyro_tilt_corrected=gyro_tilted,
            sustained_window=sustained_window,
            min_speed_change_mps=min_speed_change_mps,
            max_yaw_rate_dps=max_yaw_rate_dps,
            min_dominance_ratio=min_dominance_ratio,
            max_centripetal_accel_mps2=max_centripetal_accel_mps2,
            return_details=True,
        )
        if yaw_confident:
            method_used = "acceleration"

    if not yaw_confident:
        forward = np.array([1.0, 0.0], dtype=float)

    R_yaw = yaw_alignment_matrix(forward)
    R_total = R_yaw @ R_tilt

    yaw_src = method_used if yaw_confident else "identity_fallback"
    diag = {
        "gravity_vector": gravity_vector.tolist(),
        "gravity_magnitude": float(np.linalg.norm(gravity_vector)),
        "estimated_forward_xy": forward.tolist(),
        "mounting_yaw_deg": float(details.get("mean_delta_deg", 0.0)),
        "yaw_correction_deg": float(np.degrees(np.arctan2(R_yaw[1, 0], R_yaw[0, 0]))),
        "yaw_correction_confident": yaw_confident,
        "calibration_method": method_used,
        "yaw_source": yaw_src,
        "primary_runtime_heading_source": "GPS_ORIENTATION",
        "position_course_used_as_primary": False,
        "yaw_rejection_reason": details.get("rejection_reason"),
        "yaw_details": details,
        "yaw_note": (
            f"Confident mounting yaw calibrated via {yaw_src}."
            if yaw_confident else
            f"LOW CONFIDENCE: mounting yaw estimation rejected ({details.get('rejection_reason')}). "
            "Yaw correction left at 0 deg (identity) fallback."
        ),
    }
    return R_total, diag


def apply_rotation(vectors: np.ndarray, R: np.ndarray) -> np.ndarray:
    """vectors: (N, 3) -> rotated (N, 3)."""
    return vectors @ R.T
