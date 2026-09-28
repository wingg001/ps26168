"""
End-to-end Phase 1 pipeline for the real synchronized IO-VNBD S1 recording.

The phone stream supplies the IMU input. The synchronized vehicle stream supplies
primary ground-truth speed/position after an explicitly estimated constant time
alignment. Phone GNSS is retained separately for later GNSS-state handling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .alignment import apply_rotation, build_alignment_matrix
from .calibration import static_bias_calibration
from .config import Phase1Config
from .loaders import get_time_seconds, load_csv, slice_segment
from .normalization import apply_normalizer, fit_normalizer
from .schema_utils import build_schema_report
from .time_alignment import estimate_vehicle_time_offset, interpolate_vehicle_to_phone_time
from .windowing import chronological_train_val_split, make_windows

IMU_CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


def _require(mapping: dict, field: str, file_label: str) -> str:
    col = mapping.get(field)
    if col is None:
        raise KeyError(
            f"Required field '{field}' is not mapped for {file_label}. "
            f"Fill it in under s_columns/v_columns in your config YAML."
        )
    return col


def run_phase1(config: Phase1Config) -> dict:
    report: dict = {"drive_id": config.drive_id}

    # ---- 1. load raw (read-only) ----
    s_df = load_csv(config.s_csv)
    v_df = load_csv(config.v_csv)

    scol = config.s_columns.as_dict()
    vcol = config.v_columns.as_dict()
    s_time_col = _require(scol, "time", "s_columns")
    s_accel_cols = tuple(_require(scol, f"accel_{a}", "s_columns") for a in "xyz")
    s_gyro_cols = tuple(_require(scol, f"gyro_{a}", "s_columns") for a in "xyz")
    s_speed_col = _require(scol, "speed", "s_columns")
    s_lat_col = _require(scol, "lat", "s_columns")
    s_lon_col = _require(scol, "lon", "s_columns")

    v_time_col = _require(vcol, "time", "v_columns")
    v_speed_col = _require(vcol, "speed", "v_columns")
    v_lat_col = _require(vcol, "lat", "v_columns")
    v_lon_col = _require(vcol, "lon", "v_columns")
    v_heading_col = vcol.get("heading")

    s_schema = build_schema_report(s_df, Path(config.s_csv).name, time_field_override=s_time_col)
    v_schema = build_schema_report(v_df, Path(config.v_csv).name, time_field_override=v_time_col)
    report["s_schema"] = s_schema.to_markdown()
    report["v_schema"] = v_schema.to_markdown()

    # ---- 2. canonical time arrays (ALL downstream math uses seconds) ----
    s_t_full = get_time_seconds(s_df, s_time_col).to_numpy(dtype=float)
    v_t_full = get_time_seconds(v_df, v_time_col).to_numpy(dtype=float)

    # ---- 3. estimate V/S time-base offset from the full synchronized recording ----
    offset_diag = estimate_vehicle_time_offset(
        s_t_full,
        s_df[s_speed_col].to_numpy(dtype=float),
        v_t_full,
        v_df[v_speed_col].to_numpy(dtype=float),
        search_half_width_s=config.max_vs_time_offset_s,
    )
    # Explicitly compute the aligned overlap in the phone time base.
    # Convention: vehicle_time = phone_time + offset_s, so a vehicle timestamp
    # vt corresponds to phone timestamp vt - offset_s.
    offset_s = float(offset_diag["offset_s"])
    phone_start_abs = float(np.nanmin(s_t_full))
    phone_end_abs = float(np.nanmax(s_t_full))
    vehicle_start_abs = float(np.nanmin(v_t_full))
    vehicle_end_abs = float(np.nanmax(v_t_full))
    aligned_vehicle_start_in_phone = vehicle_start_abs - offset_s
    aligned_vehicle_end_in_phone = vehicle_end_abs - offset_s
    overlap_start_abs = max(phone_start_abs, aligned_vehicle_start_in_phone)
    overlap_end_abs = min(phone_end_abs, aligned_vehicle_end_in_phone)
    if overlap_end_abs <= overlap_start_abs:
        raise RuntimeError(
            "No overlapping S/V time range after alignment. "
            f"Phone=[{phone_start_abs:.3f}, {phone_end_abs:.3f}] s; "
            f"vehicle=[{vehicle_start_abs:.3f}, {vehicle_end_abs:.3f}] s; "
            f"offset={offset_s:.3f} s."
        )

    overlap_start_rel = overlap_start_abs - phone_start_abs
    overlap_end_rel = overlap_end_abs - phone_start_abs
    overlap_duration = overlap_end_abs - overlap_start_abs
    offset_diag.update({
        "phone_time_range_s": [phone_start_abs, phone_end_abs],
        "vehicle_time_range_s": [vehicle_start_abs, vehicle_end_abs],
        "aligned_vehicle_range_in_phone_time_s": [aligned_vehicle_start_in_phone, aligned_vehicle_end_in_phone],
        "aligned_overlap_in_phone_time_s": [overlap_start_abs, overlap_end_abs],
        "aligned_overlap_relative_to_phone_start_s": [overlap_start_rel, overlap_end_rel],
        "aligned_overlap_duration_s": overlap_duration,
    })
    report["vs_time_alignment"] = offset_diag

    # ---- 4. short configurable phone segment ----
    requested_start_s = float(config.segment_start_s)
    requested_duration_s = config.segment_duration_s
    selected_start_s = requested_start_s
    selected_duration_s = requested_duration_s
    segment_adjustment = "none"

    if requested_duration_s is not None:
        requested_end_s = requested_start_s + float(requested_duration_s)
        if requested_start_s < overlap_start_rel or requested_end_s > overlap_end_rel:
            if not config.auto_clip_to_vs_overlap:
                raise RuntimeError(
                    "Configured phone segment is outside the valid aligned V/S overlap. "
                    f"Requested=[{requested_start_s:.3f}, {requested_end_s:.3f}] s relative to phone start; "
                    f"valid overlap=[{overlap_start_rel:.3f}, {overlap_end_rel:.3f}] s. "
                    "Set auto_clip_to_vs_overlap=true or choose a valid segment."
                )
            selected_start_s = max(requested_start_s, overlap_start_rel)
            selected_end_s = min(requested_end_s, overlap_end_rel)
            selected_duration_s = selected_end_s - selected_start_s
            segment_adjustment = "auto-clipped to aligned V/S overlap"
            if selected_duration_s <= 0:
                raise RuntimeError(
                    "The configured segment has no valid overlap with the aligned vehicle data. "
                    f"Requested=[{requested_start_s:.3f}, {requested_end_s:.3f}] s; "
                    f"valid overlap=[{overlap_start_rel:.3f}, {overlap_end_rel:.3f}] s."
                )

    s_seg = slice_segment(s_df, s_time_col, selected_start_s, selected_duration_s)
    s_t = get_time_seconds(s_seg, s_time_col).to_numpy(dtype=float)
    report["segment_n_rows"] = len(s_seg)
    report["segment_requested_start_s"] = requested_start_s
    report["segment_requested_duration_s"] = requested_duration_s
    report["segment_start_s"] = selected_start_s
    report["segment_duration_s"] = float(s_t[-1] - s_t[0])
    report["segment_adjustment"] = segment_adjustment

    # ---- 5. sampling-rate / gap check on the REAL segment ----
    seg_schema = build_schema_report(s_seg, f"{Path(config.s_csv).name} [segment]", time_field_override=s_time_col)
    report["segment_schema"] = seg_schema.to_markdown()
    if seg_schema.sample_rate_hz is None:
        raise RuntimeError("Could not determine sample rate for the segment; check time units/mapping.")
    sample_rate_hz = seg_schema.sample_rate_hz

    # ---- 6. static bias calibration ----
    calib = static_bias_calibration(
        s_seg,
        time_col=s_time_col,
        accel_cols=s_accel_cols,
        gyro_cols=s_gyro_cols,
        window_start_s=config.static_window_start_s,
        window_duration_s=config.static_window_duration_s,
        auto_detect=config.auto_detect_static_window,
    )
    report["calibration"] = {
        "window_start_s": calib.window_start_s,
        "window_end_s": calib.window_end_s,
        "n_samples": calib.n_samples,
        "static_ok": calib.static_ok,
        "accel_bias_xy": calib.accel_bias[:2].tolist(),
        "gyro_bias": calib.gyro_bias.tolist(),
        "accel_static_std": calib.accel_static_std.tolist(),
        "gyro_static_std": calib.gyro_static_std.tolist(),
        "gravity_vector": calib.gravity_vector.tolist(),
    }

    # ---- 7. apply bias correction ----
    accel_raw = s_seg[list(s_accel_cols)].to_numpy(dtype=float)
    gyro_raw = s_seg[list(s_gyro_cols)].to_numpy(dtype=float)
    accel_corr = accel_raw - calib.accel_bias
    gyro_corr = gyro_raw - calib.gyro_bias

    # ---- 8. phone-to-vehicle frame alignment ----
    # Speed used for the yaw heuristic is phone GPS speed, only as a signal for
    # finding a dynamic event; it is NOT the training ground truth.
    phone_speed_kmh = s_seg[s_speed_col].to_numpy(dtype=float)
    # alignment.py uses a m/s threshold for dynamic-event detection. Convert the
    # phone GPS speed (stored in km/h) only for this heading-estimation helper.
    phone_speed_mps = phone_speed_kmh / 3.6
    R, align_diag = build_alignment_matrix(calib.gravity_vector, accel_corr, phone_speed_mps)
    accel_aligned = apply_rotation(accel_corr, R)
    gyro_aligned = apply_rotation(gyro_corr, R)
    accel_aligned[:, 2] -= np.linalg.norm(calib.gravity_vector)
    report["alignment"] = align_diag
    report["alignment"]["rotation_matrix"] = R.tolist()

    imu = np.concatenate([accel_aligned, gyro_aligned], axis=1)

    # ---- 9. interpolate VEHICLE ground truth onto PHONE IMU timestamps ----
    # Convention: vehicle_time = phone_time + estimated offset.
    v_speed_full = v_df[v_speed_col].to_numpy(dtype=float)
    v_lat_full = v_df[v_lat_col].to_numpy(dtype=float)
    v_lon_full = v_df[v_lon_col].to_numpy(dtype=float)
    gt_speed = interpolate_vehicle_to_phone_time(s_t, v_t_full, v_speed_full, offset_diag["offset_s"])
    gt_lat = interpolate_vehicle_to_phone_time(s_t, v_t_full, v_lat_full, offset_diag["offset_s"])
    gt_lon = interpolate_vehicle_to_phone_time(s_t, v_t_full, v_lon_full, offset_diag["offset_s"])

    # Fail loudly rather than silently switching to phone GPS as the primary label.
    finite_speed = np.isfinite(gt_speed)
    finite_gt = finite_speed & np.isfinite(gt_lat) & np.isfinite(gt_lon)
    report["ground_truth_coverage"] = {
        "speed_coverage_pct": float(100.0 * finite_speed.mean()),
        "full_speed_lat_lon_coverage_pct": float(100.0 * finite_gt.mean()),
        "phone_samples": int(len(gt_speed)),
        "vehicle_speed_is_primary": True,
    }
    if finite_gt.mean() < 0.98:
        raise RuntimeError(
            f"Only {100.0 * finite_gt.mean():.2f}% of phone samples received complete vehicle ground truth "
            f"after V/S alignment (speed+lat+lon). Speed-only coverage is {100.0 * finite_speed.mean():.2f}%. "
            f"Aligned overlap={overlap_duration:.3f}s; selected segment={report['segment_duration_s']:.3f}s. "
            "No phone-speed fallback is used."
        )

    # Keep phone GNSS separately for later GNSS-state handling.
    phone_gnss_speed = s_seg[s_speed_col].to_numpy(dtype=float)
    phone_gnss_lat = s_seg[s_lat_col].to_numpy(dtype=float)
    phone_gnss_lon = s_seg[s_lon_col].to_numpy(dtype=float)

    report["ground_truth_source"] = {
        "speed_col": v_speed_col,
        "lat_col": v_lat_col,
        "lon_col": v_lon_col,
        "file": Path(config.v_csv).name,
        "note": "Primary ground truth is the synchronized vehicle-side Velocity (km/hr), "
                "interpolated onto phone IMU timestamps after explicit V/S time-offset estimation. "
                "Phone GPS remains separate and is not used as the primary training label.",
    }
    report["phone_gnss_source"] = {
        "speed_col": s_speed_col,
        "lat_col": s_lat_col,
        "lon_col": s_lon_col,
        "note": "Retained separately for later GNSS availability/state-machine logic.",
    }

    # ---- 10. overlapping windows ----
    ds = make_windows(
        t=s_t,
        imu=imu,
        sample_rate_hz=sample_rate_hz,
        window_seconds=config.window_seconds,
        overlap_fraction=config.overlap_fraction,
        gt_speed=gt_speed,
        gt_lat=gt_lat,
        gt_lon=gt_lon,
    )
    report["windowing"] = {
        "n_windows": ds.windows.shape[0],
        "window_shape": list(ds.windows.shape[1:]),
        "sample_rate_hz": sample_rate_hz,
        "window_seconds": config.window_seconds,
        "overlap_fraction": config.overlap_fraction,
    }

    # Window-end phone GNSS values kept separately for downstream GNSS logic.
    starts = [int(round((t - s_t[0]) * sample_rate_hz)) for t in ds.window_start_times]
    ends = [min(max(0, i), len(phone_gnss_speed) - 1) for i in (np.searchsorted(s_t, ds.window_end_times))]
    phone_gnss_speed_w = phone_gnss_speed[ends]
    phone_gnss_lat_w = phone_gnss_lat[ends]
    phone_gnss_lon_w = phone_gnss_lon[ends]

    # ---- 11. train/val split + train-only normalization ----
    train_idx, val_idx = chronological_train_val_split(ds, config.train_fraction)
    stats = fit_normalizer(ds.windows[train_idx], IMU_CHANNELS)
    windows_norm = apply_normalizer(ds.windows, stats)
    report["normalization"] = stats.to_dict()
    report["split"] = {"n_train": len(train_idx), "n_val": len(val_idx)}

    # ---- 12. save outputs ----
    out_dir = Path(config.output_dir) / config.drive_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "windows.npz",
        windows_raw=ds.windows,
        windows_norm=windows_norm,
        window_start_times=ds.window_start_times,
        window_end_times=ds.window_end_times,
        gt_speed=ds.gt_speed if ds.gt_speed is not None else np.array([]),
        gt_lat=ds.gt_lat if ds.gt_lat is not None else np.array([]),
        gt_lon=ds.gt_lon if ds.gt_lon is not None else np.array([]),
        phone_gnss_speed=phone_gnss_speed_w,
        phone_gnss_lat=phone_gnss_lat_w,
        phone_gnss_lon=phone_gnss_lon_w,
        train_idx=train_idx,
        val_idx=val_idx,
        channel_names=np.array(IMU_CHANNELS),
    )
    with open(out_dir / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)

    report["outputs"] = {
        "windows_npz": str(out_dir / "windows.npz"),
        "normalization_stats_json": str(out_dir / "normalization_stats.json"),
    }

    report_dir = Path(config.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase1_{config.drive_id}_report.md"
    _write_markdown_report(report, report_path)
    report["report_path"] = str(report_path)
    return report


def _write_markdown_report(report: dict, path: Path) -> None:
    lines = [f"# Phase 1 report — {report['drive_id']}", ""]
    lines.append("## V/S time alignment")
    lines.append(f"```json\n{json.dumps(report['vs_time_alignment'], indent=2)}\n```")
    lines.append("")
    lines.append("## Segment")
    lines.append(
        f"- requested start_s: {report.get('segment_requested_start_s')}, "
        f"requested duration_s: {report.get('segment_requested_duration_s')}"
    )
    lines.append(
        f"- selected start_s: {report['segment_start_s']}, "
        f"actual duration_s: {report['segment_duration_s']}, rows: {report['segment_n_rows']}"
    )
    lines.append(f"- adjustment: {report.get('segment_adjustment', 'none')}")
    lines.append("")
    lines.append(report["segment_schema"])
    lines.append("")
    lines.append("## Full-file schema (S-)")
    lines.append(report["s_schema"])
    lines.append("")
    lines.append("## Full-file schema (V-)")
    lines.append(report["v_schema"])
    lines.append("")
    lines.append("## Calibration")
    lines.append(f"```json\n{json.dumps(report['calibration'], indent=2)}\n```")
    lines.append("")
    lines.append("## Frame alignment")
    lines.append(f"```json\n{json.dumps(report['alignment'], indent=2)}\n```")
    lines.append("")
    lines.append("## Ground truth coverage")
    lines.append(f"```json\n{json.dumps(report['ground_truth_coverage'], indent=2)}\n```")
    lines.append("")
    lines.append("## Ground truth source")
    lines.append(f"```json\n{json.dumps(report['ground_truth_source'], indent=2)}\n```")
    lines.append("")
    lines.append("## Phone GNSS (kept separate)")
    lines.append(f"```json\n{json.dumps(report['phone_gnss_source'], indent=2)}\n```")
    lines.append("")
    lines.append("## Windowing")
    lines.append(f"```json\n{json.dumps(report['windowing'], indent=2)}\n```")
    lines.append("")
    lines.append("## Train/val split + normalization (train-set-only stats)")
    lines.append(f"```json\n{json.dumps({**report['split'], 'normalization': report['normalization']}, indent=2)}\n```")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"```json\n{json.dumps(report['outputs'], indent=2)}\n```")
    path.write_text("\n".join(lines), encoding="utf-8")
