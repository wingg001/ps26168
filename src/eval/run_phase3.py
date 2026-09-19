import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.filters.gnss_deficit import GnssDeficitManager, GnssState

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))



def load_config(config_path=None):
    """Load Phase 3 YAML using a repository-relative default path."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "phase3_navigation.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Phase 3 config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(path_value, base_dir=PROJECT_ROOT):
    """Resolve a path without depending on the user's Windows username."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (Path(base_dir) / p).resolve()


def resolve_manifest_path(path_value, manifest_path):
    """Resolve a dataset path from the Phase 2 manifest without hard-coding a username.

    The IO-VNBD manifest lives under ``<phase2_root>/reports/phase2`` while
    dataset files referenced by the manifest live under ``<phase2_root>/data``.
    Therefore both locations are checked before falling back to the repository root.
    """
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()

    manifest_path = Path(manifest_path).resolve()
    # .../<phase2_root>/reports/phase2/dataset_manifest.json -> parents[2] is phase2_root.
    phase2_root = manifest_path.parent.parent.parent

    candidates = [
        (manifest_path.parent / p).resolve(),
        (phase2_root / p).resolve(),
        (PROJECT_ROOT / p).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return the most useful expected location for the Phase 2 layout.
    return candidates[1]


def estimate_time_alignment(t_p, phone_speed, t_v, vehicle_speed):
    """Estimate the constant V/S timestamp offset for the run.

    Phase 1's ``estimate_vehicle_time_offset`` returns a dict; ``offset_s``
    holds the float offset and ``quality`` the reported alignment quality.
    For genuinely degenerate sessions (e.g. a flat phone GPS speed makes the
    speed-correlation alignment unusable) Phase 1 raises ValueError. In that
    case we fall back to the clock-origin offset guess
    ``median(vehicle_t) - median(phone_t)`` and explicitly label the result
    ``"fallback"``. The fallback is ONLY a timestamp mapping -- it produces no
    navigation/error metrics and claims no alignment quality.
    """
    from src.phase1.time_alignment import estimate_vehicle_time_offset

    try:
        align_res = estimate_vehicle_time_offset(t_p, phone_speed, t_v, vehicle_speed)
        return float(align_res["offset_s"]), str(align_res.get("quality", "low"))
    except ValueError:
        offset_s = float(np.median(t_v) - np.median(t_p))
        return offset_s, "fallback"


def calculate_metrics(errors):
    if len(errors) == 0:
        return 0.0, 0.0, 0.0
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(errors)),
        float(np.sqrt(np.mean(errors**2))),
        float(np.max(errors)),
    )


def parse_phone_gnss_speed(df_p, gps_speed_col):
    """Raw phone GPS-speed column stored values are in m/s.

    Empirically verified: vw12 raw samples read 24.09..26.5 while the reference
    (V-file) shows 24.06..25.4 m/s and distinct-fix displacement speed is ~25.6
    m/s. Interpreting the column as km/h (/3.6) would under-estimate by ~3.6x.
    """
    return pd.to_numeric(df_p[gps_speed_col], errors="coerce").to_numpy(dtype=float)


def estimate_initial_course(phone_e_all, phone_n_all, gnss_distinct, min_fixes=3, min_span_m=50.0):
    """Estimate the initial course (deg, clockwise from north) from the total
    displacement spanned by the first ``min_fixes`` DISTINCT smartphone GNSS
    fixes. Returns None when fewer than ``min_fixes`` distinct fixes exist or
    the travelled distance is below ``min_span_m`` (insufficient motion to
    trust the direction given the noise floor). Smartphone data only.
    """
    dk = np.flatnonzero(np.asarray(gnss_distinct, dtype=bool))
    if len(dk) < min_fixes:
        return None
    k0, k1 = int(dk[0]), int(dk[min_fixes - 1])
    de = float(phone_e_all[k1] - phone_e_all[k0])
    dn = float(phone_n_all[k1] - phone_n_all[k0])
    span = float(np.hypot(de, dn))
    if span < min_span_m:
        return None
    course_deg = float(np.degrees(np.arctan2(de, dn))) % 360.0
    # Yaw convention matches filter euler state (yaw = 90 - heading_cw_from_north).
    yaw_rad = float(np.radians(90.0 - course_deg))
    return {
        "course_deg": course_deg,
        "yaw_rad": yaw_rad,
        "span_m": span,
        "n_fixes": int(min_fixes),
    }


def compute_initialization(
    phone_e_all,
    phone_n_all,
    gnss_distinct,
    phone_speed_mps,
    zupt_enabled,
    calib_accel_bias,
    calib_gyro_bias,
    R_veh,
    min_fixes=3,
    min_span_m=50.0,
    first_fix_only=False,
):
    """Compute the runtime UKF initial state from SMARTPHONE data only.

    - Bias: use the Phase 1 static calibration ONLY when a genuine stationary
      window was found (``zupt_enabled``); otherwise zero bias (never use
      moving-window means as a static bias estimate).
    - Yaw: from distinct phone-GNSS course when sufficient motion exists;
      otherwise the documented yaw = 0 fallback (no ground truth).
    - Velocity: phone GNSS speed (m/s) oriented by the GNSS course; zero with a
      logged reason when no valid speed or no reliable course exists.

    No V-file / reference value is accepted by this function.

    When ``first_fix_only=True``, only the very first distinct GNSS fix is
    considered for course estimation.  This enforces causal initialisation at
    t=0: no future GNSS fix is used to compute the initial heading.
    """
    from src.phase1.alignment import apply_rotation

    accel_bias_veh = np.zeros(3)
    gyro_bias_veh = np.zeros(3)
    if zupt_enabled:
        accel_bias_veh = apply_rotation(calib_accel_bias[None, :], R_veh)[0]
        gyro_bias_veh = apply_rotation(calib_gyro_bias[None, :], R_veh)[0]
        bias_source = "Phase 1 static calibration (valid stationary window)"
    else:
        bias_source = "zero (Phase 1 static calibration invalid/non-stationary; no moving-data bias used)"

    # Causal guard: at t=0 no future GNSS fixes exist.  When first_fix_only
    # is set, restrict the distinct-fix mask to the first fix so that
    # estimate_initial_course cannot reach into the future.
    gnss_mask = np.asarray(gnss_distinct, dtype=bool)
    if first_fix_only:
        first_idx = np.flatnonzero(gnss_mask)
        if len(first_idx) > 0:
            gnss_mask = np.zeros_like(gnss_mask)
            gnss_mask[first_idx[0]] = True

    est = estimate_initial_course(
        phone_e_all, phone_n_all, gnss_mask,
        min_fixes=min_fixes, min_span_m=min_span_m,
    )
    if est is not None:
        init_yaw = est["yaw_rad"]
        heading_source = (
            "phone GNSS course ({:.1f} deg, {:.0f} m, {} distinct fixes)".format(
                est["course_deg"], est["span_m"], est["n_fixes"]
            )
        )
    else:
        init_yaw = 0.0
        heading_source = "fallback yaw=0 (insufficient distinct-fix motion; no ground truth used)"

    init_vel = np.zeros(3)
    valid = np.isfinite(phone_speed_mps) & (phone_speed_mps > 0.0)
    if np.any(valid):
        spd = float(phone_speed_mps[np.flatnonzero(valid)[0]])
        if est is not None:
            cr = np.radians(est["course_deg"])
            init_vel = np.array([spd * np.sin(cr), spd * np.cos(cr), 0.0])
            velocity_source = "phone GNSS speed {:.2f} m/s + GNSS course {:.1f} deg".format(
                spd, est["course_deg"]
            )
        else:
            velocity_source = "zero (speed {:.2f} m/s present but no reliable course)".format(spd)
    else:
        velocity_source = "zero (no valid phone GNSS speed)"

    return {
        "init_yaw": init_yaw,
        "init_vel": init_vel,
        "accel_bias_veh": accel_bias_veh,
        "gyro_bias_veh": gyro_bias_veh,
        "course": est,
        "bias_source": bias_source,
        "heading_source": heading_source,
        "velocity_source": velocity_source,
    }


def run_session(session_id, manifest, config):
    # Project-specific Phase 1/2 dependencies are imported lazily so that
    # configuration/metric helpers and unit tests can run from this Phase 3
    # package alone. The full project repository still supplies these modules
    # for real-data execution.
    from src.filters.coordinates import LocalTangentPlane
    from src.filters.ukf import ScaledUKF
    from src.filters.mechanization import propagate_ins_state
    from src.filters.constraints import (
        is_stationary,
        hx_nhc,
        hx_speed,
        hx_gnss,
        chi_square_gate,
        match_cnn_speed,
    )
    from src.phase1.loaders import load_csv, get_time_seconds
    from src.phase1.calibration import static_bias_calibration
    from src.phase1.alignment import build_alignment_matrix, apply_rotation
    from src.phase1.time_alignment import interpolate_vehicle_to_phone_time
    from src.calibration.gnss_noise import (
        calibrate_gnss_noise,
        format_report,
        write_calibration_csv,
        calibrate_gnss_noise_from_accuracy,
        format_report_accuracy,
        write_calibration_accuracy_csv,
    )
    print(f"\n--- Running UKF Pipeline for Session: {session_id} ---")

    pair = next((p for p in manifest["pairs"] if p["session_id"] == session_id), None)
    if not pair:
        raise ValueError(f"Session {session_id} not found in manifest.")

    # Dataset paths in the manifest may be relative to the manifest directory.
    manifest_path = resolve_repo_path(config["data"]["manifest_path"])
    s_path = resolve_manifest_path(pair["phone_path"], manifest_path)
    v_path = resolve_manifest_path(pair["vehicle_path"], manifest_path)

    if not s_path.exists() or not v_path.exists():
        raise FileNotFoundError(
            f"Data files for {session_id} not found.\nPhone: {s_path}\nVehicle: {v_path}"
        )

    p_info = next(f for f in manifest["per_file"] if f["path"] == pair["phone_path"])
    v_info = next(f for f in manifest["per_file"] if f["path"] == pair["vehicle_path"])
    p_map = p_info["detected_mapping"]
    v_map = v_info["detected_mapping"]

    df_p = load_csv(s_path)
    df_v = load_csv(v_path)

    t_p = get_time_seconds(df_p, p_map["phone_time"]).to_numpy(dtype=float)
    t_v = get_time_seconds(df_v, v_map["vehicle_time"]).to_numpy(dtype=float)

    # 1. TIME SYNCHRONIZATION via Phase 1 logic.
    s_speed = pd.to_numeric(df_p[p_map["gps_speed"]], errors="coerce").fillna(0).to_numpy(dtype=float) / 3.6
    v_speed = pd.to_numeric(df_v[v_map["velocity_kmh"]], errors="coerce").fillna(0).to_numpy(dtype=float) / 3.6
    offset_s, align_quality = estimate_time_alignment(t_p, s_speed, t_v, v_speed)
    time_sync_fallback = align_quality == "fallback"
    if time_sync_fallback:
        print(
            f"Phase 1 Time Alignment: DEGENERATE phone GPS speed; "
            f"timestamp-alignment fallback offset {offset_s:.3f}s. "
            "(No alignment quality is claimed; this is timestamp alignment only.)"
        )
    else:
        print(
            f"Phase 1 Time Alignment: v_time ~ s_time + {offset_s:.3f}s "
            f"(quality: {align_quality})"
        )

    # 2. INITIAL ATTITUDE & ZUPT BASELINE via Phase 1 calibration.
    calib_res = static_bias_calibration(
        df_p,
        p_map["phone_time"],
        (p_map["accel_x"], p_map["accel_y"], p_map["accel_z"]),
        (p_map["gyro_yaw"], p_map["gyro_pitch"], p_map["gyro_roll"]),
        window_start_s=0.0,
        window_duration_s=5.0,
    )

    zupt_enabled = bool(calib_res.static_ok)
    if zupt_enabled:
        # Compare like-for-like quantities: sum of per-axis variances and
        # magnitude of the per-axis static gyro standard deviations.
        base_accel_var = float(np.sum(calib_res.accel_static_std**2))
        base_gyro_mag = float(np.linalg.norm(calib_res.gyro_static_std))
        print(
            f"ZUPT Baseline: accel_var={base_accel_var:.5f}, "
            f"gyro_mag={base_gyro_mag:.5f}"
        )
    else:
        base_accel_var = np.nan
        base_gyro_mag = np.nan
        print("ZUPT: DISABLED (Phase 1 static calibration invalid/unavailable).")

    # Build Phone-to-Vehicle frame alignment matrix from actual calibration.
    imu_raw_slice = df_p[
        [p_map["accel_x"], p_map["accel_y"], p_map["accel_z"]]
    ].iloc[:50].to_numpy(dtype=float)
    R_veh, r_diag = build_alignment_matrix(calib_res.gravity_vector, imu_raw_slice)

    raw_accel = df_p[
        [p_map["accel_x"], p_map["accel_y"], p_map["accel_z"]]
    ].to_numpy(dtype=float)
    raw_gyro = df_p[
        [p_map["gyro_yaw"], p_map["gyro_pitch"], p_map["gyro_roll"]]
    ].to_numpy(dtype=float)

    imu_veh = np.zeros((len(df_p), 6), dtype=float)
    imu_veh[:, 0:3] = apply_rotation(raw_accel, R_veh)
    imu_veh[:, 3:6] = apply_rotation(raw_gyro, R_veh)

    gnss_raw = df_p[[p_map["gps_latitude"], p_map["gps_longitude"]]].to_numpy(dtype=float)
    acc_col = p_map.get("gps_accuracy")
    if acc_col and acc_col in df_p.columns:
        gnss_acc = pd.to_numeric(df_p[acc_col], errors="coerce").to_numpy(dtype=float)
    else:
        gnss_acc = np.full(len(df_p), np.nan, dtype=float)

    # 3. REFERENCE ALIGNMENT: first valid vehicle reference is shared LTP origin.
    valid_v = df_v[[v_map["latitude"], v_map["longitude"]]].dropna()
    if valid_v.empty:
        raise ValueError("No valid reference GPS found.")

    origin_idx = valid_v.index[0]
    lat0 = float(valid_v.loc[origin_idx, v_map["latitude"]])
    lon0 = float(valid_v.loc[origin_idx, v_map["longitude"]])
    ltp = LocalTangentPlane(lat0, lon0)

    v_lats = pd.to_numeric(df_v[v_map["latitude"]], errors="coerce").to_numpy(dtype=float)
    v_lons = pd.to_numeric(df_v[v_map["longitude"]], errors="coerce").to_numpy(dtype=float)

    ref_e = np.full(len(t_v), np.nan, dtype=float)
    ref_n = np.full(len(t_v), np.nan, dtype=float)
    valid_geo = np.isfinite(v_lats) & np.isfinite(v_lons)
    for i in np.flatnonzero(valid_geo):
        enu = ltp.to_enu(v_lats[i], v_lons[i])
        ref_e[i], ref_n[i] = enu[0], enu[1]

    ref_enu_p = np.full((len(t_p), 2), np.nan, dtype=float)
    ref_enu_p[:, 0] = interpolate_vehicle_to_phone_time(t_p, t_v, ref_e, offset_s)
    ref_enu_p[:, 1] = interpolate_vehicle_to_phone_time(t_p, t_v, ref_n, offset_s)

    # 3.1 EMPIRICAL GNSS MEASUREMENT-NOISE CALIBRATION.
    # Computes a robust scalar horizontal GNSS sigma from synchronized
    # phone-GNSS vs vehicle-reference residuals in the common LTP/ENU frame,
    # using a calibration window that lies strictly BEFORE the GNSS blackout
    # (the evaluation/test window is never used for calibration).
    gnss_noise_cfg = config["ukf"].get("gnss_noise", {}) or {}
    gnss_noise_mode = str(gnss_noise_cfg.get("mode", "empirical")) or "empirical"
    phone_e_all = np.full(len(t_p), np.nan, dtype=float)
    phone_n_all = np.full(len(t_p), np.nan, dtype=float)
    for k in range(len(t_p)):
        if np.isfinite(gnss_raw[k, 0]) and np.isfinite(gnss_raw[k, 1]):
            enu_p = ltp.to_enu(gnss_raw[k, 0], gnss_raw[k, 1])
            phone_e_all[k] = enu_p[0]
            phone_n_all[k] = enu_p[1]

    # DISTINCT GNSS FIX DETECTION: a row carries a *new* fix iff its valid
    # coordinates differ from the previous valid GNSS coordinate. Repeated 10 Hz
    # rows of the same fix are NOT treated as independent measurements.
    gnss_distinct = np.zeros(len(t_p), dtype=bool)
    prev_e, prev_n = np.nan, np.nan
    for k in range(len(t_p)):
        if not (np.isfinite(phone_e_all[k]) and np.isfinite(phone_n_all[k])):
            continue
        if np.isnan(prev_e) or abs(phone_e_all[k] - prev_e) > 1e-9 or abs(phone_n_all[k] - prev_n) > 1e-9:
            gnss_distinct[k] = True
            prev_e, prev_n = phone_e_all[k], phone_n_all[k]

    sigma_gnss = None
    if gnss_noise_mode == "smartphone_accuracy":
        cal_start = float(gnss_noise_cfg.get("calibration_window_s", 30.0))
        t_rel_all = t_p - float(t_p[0])
        cal_valid = np.isfinite(phone_e_all) & np.isfinite(phone_n_all)
        cap_cfg = gnss_noise_cfg.get("accuracy_cap_m")
        cal = calibrate_gnss_noise_from_accuracy(
            gnss_acc,
            cal_valid,
            sigma_floor_m=float(gnss_noise_cfg.get("sigma_floor_m", 3.0)),
            window_start_s=0.0,
            window_duration_s=cal_start,
            t_rel=t_rel_all,
            distinct=gnss_distinct,
            accuracy_cap_m=(float(cap_cfg) if cap_cfg is not None else None),
        )
        sigma_gnss = float(cal.runtime_scale_m)
        print("\n" + format_report_accuracy(cal))
        print(
            "SMARTPHONE-ONLY SIGMA: the GNSS uncertainty scale ({:.2f} m) is the 75th "
            "percentile of the phone's own GPS ACCURACY broadcasts over distinct fixes "
            "in window [0, {}s). No vehicle/reference data is used in this "
            "calculation.".format(sigma_gnss, cal_start)
        )
        try:
            write_calibration_accuracy_csv(
                PROJECT_ROOT / "reports" / "phase3" / f"gnss_calibration_{session_id}.csv",
                cal,
                gnss_acc,
                t_rel_all,
                distinct=gnss_distinct,
            )
        except Exception as e:  # pragma: no cover - diagnostics only
            print(f"WARNING: could not write calibration CSV: {e}")
    elif gnss_noise_mode == "empirical":
        cal_start = float(gnss_noise_cfg.get("calibration_window_s", 30.0))
        t_rel_all = t_p - float(t_p[0])
        cal_valid = (
            np.isfinite(phone_e_all)
            & np.isfinite(phone_n_all)
            & np.isfinite(ref_enu_p[:, 0])
            & np.isfinite(ref_enu_p[:, 1])
        )
        cal = calibrate_gnss_noise(
            phone_e_all,
            phone_n_all,
            ref_enu_p[:, 0],
            ref_enu_p[:, 1],
            cal_valid,
            sigma_floor_m=float(gnss_noise_cfg.get("sigma_floor_m", 3.0)),
            window_start_s=0.0,
            window_duration_s=cal_start,
            t_rel=t_rel_all,
            distinct=gnss_distinct,
        )
        sigma_gnss = float(cal.sigma_used_m)
        print("\n" + format_report(cal))
        print(
            "CALIBRATION-ONLY: the empirical GNSS sigma ({:.2f} m) was derived from "
            "this same session's phone-vs-reference residuals (window [0, {}s)). It is "
            "marked calibration-only and is NOT suitable for final held-out evaluation. "
            "For final results, noise must be estimated from disjoint calibration "
            "sessions and applied to held-out test sessions.".format(
                sigma_gnss, cal_start
            )
        )
        try:
            write_calibration_csv(
                PROJECT_ROOT / "reports" / "phase3" / f"gnss_calibration_{session_id}.csv",
                cal,
                phone_e_all,
                phone_n_all,
                ref_enu_p[:, 0],
                ref_enu_p[:, 1],
                gnss_acc,
                t_rel_all,
                distinct=gnss_distinct,
            )
        except Exception as e:  # pragma: no cover - diagnostics only
            print(f"WARNING: could not write calibration CSV: {e}")

    # 4. CNN timestamps: strict nearest-row matching, no interpolation.
    cnn_t, cnn_v = np.array([], dtype=float), np.array([], dtype=float)
    preds_path = resolve_repo_path(config["data"]["phase2_preds_path"])
    if preds_path.exists():
        df_preds = pd.read_csv(preds_path)
        df_sess = df_preds[df_preds["session_id"] == session_id].copy()
        if not df_sess.empty:
            cnn_t = pd.to_numeric(df_sess["time_s"], errors="coerce").to_numpy(dtype=float)
            cnn_t = cnn_t + float(t_p[0])
            cnn_v = pd.to_numeric(df_sess["predicted_speed_kmh"], errors="coerce").to_numpy(dtype=float) / 3.6
            valid_cnn = np.isfinite(cnn_t) & np.isfinite(cnn_v)
            cnn_t, cnn_v = cnn_t[valid_cnn], cnn_v[valid_cnn]
            order = np.argsort(cnn_t)
            cnn_t, cnn_v = cnn_t[order], cnn_v[order]
    else:
        print(f"CNN predictions not found; CNN updates disabled: {preds_path}")

    # 5. INITIALIZE UKF.
    cfg_ukf = config["ukf"]
    ukf = ScaledUKF(
        dim_x=15,
        alpha=cfg_ukf["alpha"],
        beta=cfg_ukf["beta"],
        kappa=cfg_ukf["kappa"],
    )

    # --- UKF noise / covariance (set once, applies to all sessions) ---
    Q = np.zeros(15, dtype=float)
    Q[0:3] = cfg_ukf["Q_std"]["pos"] ** 2
    Q[3:6] = cfg_ukf["Q_std"]["vel"] ** 2
    Q[6:9] = cfg_ukf["Q_std"]["euler"] ** 2
    Q[9:12] = cfg_ukf["Q_std"]["accel_bias"] ** 2
    Q[12:15] = cfg_ukf["Q_std"]["gyro_bias"] ** 2
    ukf.Q = np.diag(Q)

    P0 = np.zeros(15, dtype=float)
    P0[0:3] = 4.0**2              # pos  m
    P0[3:6] = 1.0**2              # vel  m/s
    P0[6:9] = np.radians(5.0) ** 2  # euler rad
    P0[9:12] = 0.05**2            # accel_bias m/s^2
    P0[12:15] = 0.001**2          # gyro_bias rad/s
    ukf.P = np.diag(P0)

    # Pre-compute bias / vehicle-alignment state (needed whether or not
    # delayed initialisation fires).
    bias_est = compute_initialization(
        phone_e_all, phone_n_all, gnss_distinct,
        parse_phone_gnss_speed(df_p, p_map["gps_speed"]),
        zupt_enabled, calib_res.accel_bias, calib_res.gyro_bias,
        R_veh, first_fix_only=True,
    )

    # --- Causal delayed initialisation ---
    # The UKF state is NOT set at t=0.  Instead we search for the first
    # smartphone GNSS fix pair with span >= 50 m (the existing reliability
    # threshold).  When found, we initialise position, velocity and yaw from
    # that single causal baseline.  If no such pair exists before the
    # blackout the filter stays dormant (s1 case).
    phone_speed_mps = parse_phone_gnss_speed(df_p, p_map["gps_speed"])
    ukf_initialized = False
    init_timestamp = None

    # Store the initial state so trajectory length matches filter epochs.
    # Before initialisation the position is NaN (no valid estimate yet).
    results_pos = [np.full(2, np.nan)]
    ins_state = ukf.x.copy()
    raw_ins_pos = [ins_state[0:2].copy()]
    gnss_plot_pos = []
    error_time, error_values, blackout_flags = [], [], []
    errors_all, errors_blackout = [], []
    counters = {
        "gnss_rows_total": int(np.sum(np.isfinite(gnss_raw[:, 0]) & np.isfinite(gnss_raw[:, 1]))),
        "gnss_distinct_fixes": int(np.sum(gnss_distinct)),
        "gnss_repeated_skipped": int(np.sum(np.isfinite(gnss_raw[:, 0]) & np.isfinite(gnss_raw[:, 1]))) - int(np.sum(gnss_distinct)),
        "gnss_events_attempted": 0,
        "gnss_accept": 0,
        "gnss_reject": 0,
        "gnss_numerical_fail": 0,
        "nhc": 0,
        "zupt": 0,
        "cnn": 0,
        "bad_timestamp": 0,
        "adaptive_q_predictions": 0,
    }
    nis_vals = []

    # Phase 6: GNSS deficit state machine with adaptive Q scaling.
    cfg_deficit = config.get("gnss_deficit", {})
    gnss_deficit = GnssDeficitManager(
        nis_window=int(cfg_deficit.get("nis_window", 10)),
        nis_threshold=float(cfg_deficit.get("nis_threshold", 5.99)),
        accuracy_threshold_m=float(cfg_deficit.get("accuracy_threshold_m", 30.0)),
        outage_min_s=float(cfg_deficit.get("outage_min_s", 3.0)),
        q_scales=cfg_deficit.get("q_scales"),
    )
    gnss_state_counts = {s.value: 0 for s in GnssState}
    baseline_Q = ukf.Q.copy()

    b_start = float(t_p[0]) + float(config["blackout_start_s"])
    b_end = b_start + float(config["blackout_duration_s"])

    # 6. FILTER LOOP.
    for k in range(1, len(t_p)):
        dt = float(t_p[k] - t_p[k - 1])
        if dt <= 0.0 or dt > float(config["data"]["time_alignment"].get("max_dt_s", 1.0)):
            counters["bad_timestamp"] += 1
            results_pos.append(ukf.x[0:2].copy() if ukf_initialized else np.full(2, np.nan))
            raw_ins_pos.append(ins_state[0:2].copy())
            continue

        u_k = imu_veh[k]

        # --- Causal delayed initialisation (runs once, before predict) ---
        # Use the most recent consecutive distinct-fix pair whose displacement
        # spans >= 50 m.  This avoids the unstable Fix0 position that can
        # inflate the displacement speed by ~2x (ratio 1.83 vs expected ~1.0).
        in_blackout_k = b_start <= t_p[k] < b_end
        if (not ukf_initialized) and (not in_blackout_k):
            dk_so_far = np.flatnonzero(gnss_distinct[: k + 1])
            n_dk = len(dk_so_far)
            if n_dk >= 3:
                kb = int(dk_so_far[-1])
                ka = int(dk_so_far[-2])
                # Both fixes must be before blackout start.
                if t_p[kb] < b_start and t_p[ka] < b_start:
                    de = float(phone_e_all[kb] - phone_e_all[ka])
                    dn = float(phone_n_all[kb] - phone_n_all[ka])
                    span = float(np.hypot(de, dn))
                    if span >= 50.0:
                        course_deg = float(
                            np.degrees(np.arctan2(de, dn))
                        ) % 360.0
                        yaw_rad = float(np.radians(90.0 - course_deg))
                        causal_speed = float(phone_speed_mps[kb])

                        # Full state init from the causal baseline.
                        lat_g, lon_g = gnss_raw[kb]
                        if np.isfinite(lat_g) and np.isfinite(lon_g):
                            enu_kb = ltp.to_enu(lat_g, lon_g)
                            ukf.x[0] = float(enu_kb[0])
                            ukf.x[1] = float(enu_kb[1])
                        ukf.x[2] = 0.0
                        if causal_speed > 0.0:
                            ukf.x[3] = causal_speed * np.sin(np.radians(course_deg))
                            ukf.x[4] = causal_speed * np.cos(np.radians(course_deg))
                        else:
                            ukf.x[3] = 0.0
                            ukf.x[4] = 0.0
                        ukf.x[5] = 0.0
                        ukf.x[6] = 0.0
                        ukf.x[7] = 0.0
                        ukf.x[8] = yaw_rad
                        ukf.x[9:12] = bias_est["accel_bias_veh"]
                        ukf.x[12:15] = bias_est["gyro_bias_veh"]
                        ukf.P = np.diag(P0)

                        ins_state = ukf.x.copy()
                        ukf_initialized = True
                        init_timestamp = float(t_p[kb] - t_p[0])
                        print(
                            f"  Delayed init at t={init_timestamp:.3f}s "
                            f"(row {kb}): course={course_deg:.1f} deg, "
                            f"speed={causal_speed:.2f} m/s, "
                            f"pos=({ukf.x[0]:.2f}, {ukf.x[1]:.2f})"
                        )

        if not ukf_initialized:
            # No qualifying baseline yet; keep filter dormant.
            ins_state = propagate_ins_state(ins_state, u_k, dt)
            raw_ins_pos.append(ins_state[0:2].copy())
            results_pos.append(np.full(2, np.nan))
            continue

        # --- Normal UKF predict + update (only after initialisation) ---
        ins_state = propagate_ins_state(ins_state, u_k, dt)
        raw_ins_pos.append(ins_state[0:2].copy())

        # Phase 6: apply adaptive Q scaling for the prediction step.
        _qs = gnss_deficit.q_scale
        if _qs != 1.0:
            ukf.Q = baseline_Q * _qs
            counters["adaptive_q_predictions"] += 1
        ukf.predict(propagate_ins_state, dt, u_k)
        if _qs != 1.0:
            ukf.Q = baseline_Q.copy()

        # GNSS update outside simulated blackout, applied ONLY to genuinely new
        # (distinct) fixes. Repeated 10 Hz rows carrying the same coordinates are
        # not independent measurements and never generate a second update.
        gnss_attempted = False
        success = False
        accepted = False
        nis = float("nan")
        if not in_blackout_k and gnss_distinct[k]:
            gnss_attempted = True
            enu_meas = ltp.to_enu(gnss_raw[k, 0], gnss_raw[k, 1])[0:2]
            gnss_plot_pos.append((float(enu_meas[0]), float(enu_meas[1]), float(t_p[k] - t_p[0])))
            counters["gnss_events_attempted"] += 1
            if gnss_noise_mode in ("empirical", "smartphone_accuracy") and sigma_gnss is not None:
                # Scalar per-session scale: reference-derived "empirical"
                # (kept for comparison) or smartphone-only (R3 default).
                R_gnss = np.eye(2) * (sigma_gnss**2)
            else:
                acc = gnss_acc[k]
                if not np.isfinite(acc) or acc <= 0:
                    acc = float(cfg_ukf["R_std"]["gnss_floor"])
                acc = max(float(acc), float(cfg_ukf["R_std"]["gnss_floor"]))
                R_gnss = np.eye(2) * (acc**2)
            gating_fn = lambda nis, dof: chi_square_gate(
                nis, dof, cfg_ukf["gating"]["gnss_nis_confidence"]
            )
            success, nis, accepted = ukf.update(enu_meas, hx_gnss, R_gnss, gating_fn)
            if np.isfinite(nis):
                nis_vals.append(float(nis))
            if success and accepted:
                counters["gnss_accept"] += 1
            elif np.isfinite(nis):
                counters["gnss_reject"] += 1
            else:
                counters["gnss_numerical_fail"] += 1

        # Phase 6: feed the GNSS deficit manager (observation only).
        gnss_deficit.update(
            t_s=float(t_p[k] - t_p[0]),
            gnss_accepted=bool(success and accepted),
            nis=float(nis),
            gps_accuracy_m=float(gnss_acc[k]) if np.isfinite(gnss_acc[k]) else float("nan"),
        )
        gnss_state_counts[gnss_deficit.state.value] += 1

        # ZUPT / NHC / CNN.
        if k >= 10:
            window_a = imu_veh[k - 10 : k, 0:3]
            window_g = imu_veh[k - 10 : k, 3:6]
            stationary = (
                zupt_enabled
                and is_stationary(
                    window_a,
                    window_g,
                    base_accel_var,
                    base_gyro_mag,
                    cfg_ukf["zupt"]["accel_var_multiplier"],
                    cfg_ukf["zupt"]["gyro_mag_multiplier"],
                )
            )

            if stationary:
                R_zupt = np.eye(3) * (cfg_ukf["R_std"]["zupt"] ** 2)
                ukf.update(np.zeros(3), lambda x: x[3:6], R_zupt)
                counters["zupt"] += 1
            else:
                R_nhc = np.diag(
                    [cfg_ukf["R_std"]["nhc"][0] ** 2, cfg_ukf["R_std"]["nhc"][1] ** 2]
                )
                ukf.update(np.zeros(2), hx_nhc, R_nhc)
                counters["nhc"] += 1

                v_cnn = match_cnn_speed(
                    t_p[k], cnn_t, cnn_v, cfg_ukf["cnn"]["time_tolerance_s"]
                )
                if np.isfinite(v_cnn):
                    R_cnn = np.array([[cfg_ukf["R_std"]["cnn_speed"] ** 2]])
                    ukf.update(np.array([v_cnn]), hx_speed, R_cnn)
                    counters["cnn"] += 1

        results_pos.append(ukf.x[0:2].copy())

        if np.isfinite(ref_enu_p[k, 0]) and np.isfinite(ref_enu_p[k, 1]):
            err = float(np.linalg.norm(ukf.x[0:2] - ref_enu_p[k, 0:2]))
            errors_all.append(err)
            error_time.append(float(t_p[k] - t_p[0]))
            error_values.append(err)
            blackout_flags.append(in_blackout_k)
            if in_blackout_k:
                errors_blackout.append(err)

    results_pos = np.asarray(results_pos, dtype=float)
    raw_ins_pos = np.asarray(raw_ins_pos, dtype=float)
    error_time = np.asarray(error_time, dtype=float)
    error_values = np.asarray(error_values, dtype=float)
    blackout_flags = np.asarray(blackout_flags, dtype=bool)

    mae_all, rmse_all, max_all = calculate_metrics(errors_all)
    mae_b, rmse_b, max_b = calculate_metrics(errors_blackout)
    valid_reference_samples = int(len(errors_all))

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Session: {session_id} | Epochs: {len(t_p)}")
    print(f"Time Sync Offset: {offset_s:.3f}s | Quality: {align_quality}")
    print(f"Covariance Regs (PSD fixes): {ukf.regularization_count}")
    print(f"Blackout: {b_start - t_p[0]:.3f}s to {b_end - t_p[0]:.3f}s relative")
    print(
        f"GNSS: rows_total={counters['gnss_rows_total']} | distinct_fixes={counters['gnss_distinct_fixes']} "
        f"| repeated_skipped={counters['gnss_repeated_skipped']} | events_attempted={counters['gnss_events_attempted']}"
    )
    print(
        f"GNSS Accepts/Rejects/Numerical Failures: "
        f"{counters['gnss_accept']} / {counters['gnss_reject']} / {counters['gnss_numerical_fail']}"
    )
    print(
        f"Updates -> ZUPT: {counters['zupt']} | NHC: {counters['nhc']} | "
        f"CNN: {counters['cnn']}"
    )
    print(f"Bad timestamp gaps: {counters['bad_timestamp']}")
    print(f"ZUPT enabled: {zupt_enabled}")
    print(
        f"GNSS Health: NORMAL={gnss_state_counts['normal']} | "
        f"DEGRADED={gnss_state_counts['degraded']} | "
        f"OUTAGE={gnss_state_counts['outage']} | "
        f"RECOVERY={gnss_state_counts['recovery']}"
    )
    print(
        f"Adaptive Q: {counters['adaptive_q_predictions']} predictions with "
        f"Q != baseline (scales: normal=1.0, degraded={gnss_deficit._q_scales['degraded']:.1f}, "
        f"outage={gnss_deficit._q_scales['outage']:.1f}, recovery={gnss_deficit._q_scales['recovery']:.1f})"
    )
    # Initialization details (smartphone-only; informational).
    print("--- Initialization (smartphone data only) ---")
    if ukf_initialized:
        print(
            "  pos=({:.1f}, {:.1f}) m | vel=({:.2f}, {:.2f}) m/s | yaw={:.2f} deg".format(
                float(ukf.x[0]),
                float(ukf.x[1]),
                float(ukf.x[3]),
                float(ukf.x[4]),
                float(np.degrees(ukf.x[8])),            )
        )
        print(f"  init_timestamp: {init_timestamp:.3f}s (delayed from t=0)")
    else:
        print("  NO INITIALISATION (no qualifying GNSS baseline before blackout)")
    print("  bias: {}".format(bias_est["bias_source"]))
    if nis_vals:
        nis_all = np.array(nis_vals)
        print(f"GNSS NIS: median={np.median(nis_all):.1f} | p95={np.percentile(nis_all, 95):.1f} | "
              f"max={np.max(nis_all):.1f} | n={len(nis_all)} (95% gate dof=2: 5.99)")
    print("\n--- Overall Metrics ---")
    print(f"MAE  : {mae_all:.2f} m")
    print(f"RMSE : {rmse_all:.2f} m")
    print("\n--- Blackout Metrics ---")
    print(f"MAE  : {mae_b:.2f} m")
    print(f"RMSE : {rmse_b:.2f} m")
    print(f"MAX  : {max_b:.2f} m")
    print("====================================================")

    out_dir = PROJECT_ROOT / "reports" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "session_id": session_id,
        "epochs": len(t_p),
        "blackout_start_s": float(config["blackout_start_s"]),
        "blackout_end_s": float(config["blackout_start_s"] + config["blackout_duration_s"]),
        "blackout_duration_s": float(config["blackout_duration_s"]),
        "valid_reference_samples": valid_reference_samples,
        "bad_timestamp_count": counters["bad_timestamp"],
        "zupt_enabled": bool(zupt_enabled),
        "gnss_accepts": counters["gnss_accept"],
        "gnss_rejects": counters["gnss_reject"],
        "gnss_numerical_failures": counters["gnss_numerical_fail"],
        "gnss_rows_total": counters["gnss_rows_total"],
        "gnss_distinct_fixes": counters["gnss_distinct_fixes"],
        "gnss_repeated_rows_skipped": counters["gnss_repeated_skipped"],
        "gnss_update_events_attempted": counters["gnss_events_attempted"],
        "gnss_nis_median": float(np.median(nis_vals)) if nis_vals else None,
        "gnss_nis_p95": float(np.percentile(nis_vals, 95)) if nis_vals else None,
        "gnss_nis_max": float(np.max(nis_vals)) if nis_vals else None,
        "gnss_nis_n": len(nis_vals),
        "nhc_updates": counters["nhc"],
        "zupt_updates": counters["zupt"],
        "cnn_updates": counters["cnn"],
        "overall_mae_m": mae_all,
        "overall_rmse_m": rmse_all,
        "overall_max_m": max_all,
        "blackout_mae_m": mae_b,
        "blackout_rmse_m": rmse_b,
        "blackout_max_m": max_b,
        "cov_reg_count": ukf.regularization_count,
        "time_sync_offset_s": offset_s,
        "time_sync_quality": align_quality,
        "time_sync_fallback": bool(time_sync_fallback),
        "sigma_mode": gnss_noise_mode,
        "gnss_runtime_scale_m": float(sigma_gnss) if sigma_gnss is not None else None,
        "gnss_sigma_calibration_only": bool(
            gnss_noise_mode == "empirical" and sigma_gnss is not None
        ),
        "gnss_sigma_m_calibration_only": (
            float(sigma_gnss) if sigma_gnss is not None else None
        ),
        "init_pos_e_m": float(ukf.x[0]) if ukf_initialized else None,
        "init_pos_n_m": float(ukf.x[1]) if ukf_initialized else None,
        "init_vel_e_mps": float(ukf.x[3]) if ukf_initialized else 0.0,
        "init_vel_n_mps": float(ukf.x[4]) if ukf_initialized else 0.0,
        "init_vel_mag_mps": float(np.sqrt(ukf.x[3]**2 + ukf.x[4]**2)) if ukf_initialized else 0.0,
        "init_yaw_deg": float(np.degrees(ukf.x[8])) if ukf_initialized else 0.0,
        "init_heading_source": "delayed causal baseline" if ukf_initialized else "no qualifying baseline before blackout",
        "init_velocity_source": "phone GNSS speed + displacement course" if ukf_initialized else "no qualifying baseline before blackout",
        "init_bias_source": bias_est["bias_source"],
        "init_accel_bias_veh_xyz_mps2": [float(v) for v in bias_est["accel_bias_veh"]],
        "init_gyro_bias_veh_xyz_radps": [float(v) for v in bias_est["gyro_bias_veh"]],
        "gnss_health_normal_epochs": gnss_state_counts["normal"],
        "gnss_health_degraded_epochs": gnss_state_counts["degraded"],
        "gnss_health_outage_epochs": gnss_state_counts["outage"],
        "gnss_health_recovery_epochs": gnss_state_counts["recovery"],
        "gnss_health_final_state": gnss_deficit.state.value,
        "adaptive_q_predictions": counters["adaptive_q_predictions"],
        "adaptive_q_scales": dict(gnss_deficit._q_scales),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / f"{session_id}_metrics.csv", index=False)
    with (out_dir / f"{session_id}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Trajectory plot.
    valid_mask = np.isfinite(ref_enu_p[:, 0]) & np.isfinite(ref_enu_p[:, 1])
    plt.figure(figsize=(10, 8))
    plt.plot(ref_enu_p[valid_mask, 0], ref_enu_p[valid_mask, 1], "k--", label="Ground Truth")
    if raw_ins_pos.shape[0] == len(t_p):
        plt.plot(raw_ins_pos[:, 0], raw_ins_pos[:, 1], "r-", alpha=0.7, label="Raw INS")
    if gnss_plot_pos:
        gp = np.asarray(gnss_plot_pos)
        plt.plot(gp[:, 0], gp[:, 1], "g.", markersize=2, alpha=0.6, label="GNSS")
    plt.plot(results_pos[:, 0], results_pos[:, 1], "b-", label="UKF + NHC/ZUPT")
    plt.title(f"2D ENU Trajectory - {session_id}")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{session_id}_trajectory.png", dpi=150)
    plt.close()

    # Error-vs-time plot with blackout highlighted.
    plt.figure(figsize=(11, 5))
    if error_values.size:
        plt.plot(error_time, error_values, label="Horizontal Position Error")
    plt.axvspan(
        float(config["blackout_start_s"]),
        float(config["blackout_start_s"] + config["blackout_duration_s"]),
        alpha=0.2,
        label="GNSS Blackout",
    )
    plt.xlabel("Time since session start (s)")
    plt.ylabel("Horizontal Position Error (m)")
    plt.title(f"Position Error vs Time - {session_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{session_id}_error_vs_time.png", dpi=150)
    plt.close()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="vw12", help="Session ID to run")
    parser.add_argument("--config", default=None, help="Optional Phase 3 config path")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = resolve_repo_path(config["data"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest missing: {manifest_path}. "
            "Check phase3_navigation.yaml paths."
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    run_session(args.session, manifest, config)
