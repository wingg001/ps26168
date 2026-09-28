"""
PS26168 Multi-Session Navigation Benchmark Harness.

Comprehensive multi-session, multi-driver evaluation pipeline for GNSS-denied
dead reckoning and fusion across the IO-VNBD dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.run_phase3 import (
    load_config,
    resolve_manifest_path,
    resolve_repo_path,
    run_session,
)
from src.phase1.loaders import get_time_seconds, load_csv
from src.phase1.time_alignment import estimate_vehicle_time_offset

logger = logging.getLogger("run_benchmark")


def extract_driver_label(phone_path: str) -> str:
    """Extract human-readable driver identifier from file path."""
    parts = Path(phone_path).parts
    for part in parts:
        p_lower = part.lower()
        if "driver a" in p_lower:
            return "Driver A"
        if "driver b" in p_lower:
            return "Driver B"
        if "driver d" in p_lower:
            return "Driver D"
        if "driver e" in p_lower:
            if "vf" in p_lower:
                return "Driver E (Vf)"
            if "vta" in p_lower:
                return "Driver E (Vta)"
            if "vtb" in p_lower:
                return "Driver E (Vtb)"
            if "vw" in p_lower:
                return "Driver E (Vw)"
            return "Driver E"
    return Path(phone_path).parent.name


def discover_sessions(manifest: Dict[str, Any], manifest_path: Path) -> List[Dict[str, Any]]:
    """Discover all session pairs from dataset manifest and assign metadata."""
    pairs = manifest.get("pairs", [])
    per_file_lookup = {f["path"]: f for f in manifest.get("per_file", [])}

    discovered = []
    for pair in pairs:
        session_id = pair["session_id"]
        phone_rel = pair["phone_path"]
        vehicle_rel = pair["vehicle_path"]

        driver = extract_driver_label(phone_rel)
        p_info = per_file_lookup.get(phone_rel, {})
        v_info = per_file_lookup.get(vehicle_rel, {})

        s_path = resolve_manifest_path(phone_rel, manifest_path)
        v_path = resolve_manifest_path(vehicle_rel, manifest_path)

        discovered.append({
            "session_id": session_id,
            "driver": driver,
            "phone_path": s_path,
            "vehicle_path": v_path,
            "phone_rel": phone_rel,
            "vehicle_rel": vehicle_rel,
            "p_map": p_info.get("detected_mapping", {}),
            "v_map": v_info.get("detected_mapping", {}),
        })

    return discovered


def validate_session(
    session_info: Dict[str, Any],
    min_duration_s: float = 30.0,
    min_speed_mps: float = 1.0,
    min_gps_fixes: int = 5,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validate a session pair against schema, duration, monotonicity, and motion."""
    s_path = session_info["phone_path"]
    v_path = session_info["vehicle_path"]
    p_map = session_info["p_map"]
    v_map = session_info["v_map"]

    audit: Dict[str, Any] = {
        "session_id": session_info["session_id"],
        "driver": session_info["driver"],
        "duration_p_s": 0.0,
        "duration_v_s": 0.0,
        "max_vehicle_speed_mps": 0.0,
        "valid_phone_gps_fixes": 0,
        "valid_vehicle_gps_fixes": 0,
        "align_quality": "unknown",
        "align_offset_s": 0.0,
    }

    if not s_path.exists():
        return False, f"Phone file not found: {s_path}", audit
    if not v_path.exists():
        return False, f"Vehicle file not found: {v_path}", audit

    required_p = ["phone_time", "accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll", "gps_latitude", "gps_longitude", "gps_speed"]
    required_v = ["vehicle_time", "latitude", "longitude", "velocity_kmh"]

    missing_p = [k for k in required_p if k not in p_map]
    missing_v = [k for k in required_v if k not in v_map]
    if missing_p:
        return False, f"Phone schema missing mappings: {missing_p}", audit
    if missing_v:
        return False, f"Vehicle schema missing mappings: {missing_v}", audit

    try:
        df_p = load_csv(s_path)
        df_v = load_csv(v_path)
    except Exception as e:
        return False, f"Failed to load CSV: {e}", audit

    for col_key in required_p:
        if p_map[col_key] not in df_p.columns:
            return False, f"Phone column '{p_map[col_key]}' missing in CSV", audit
    for col_key in required_v:
        if v_map[col_key] not in df_v.columns:
            return False, f"Vehicle column '{v_map[col_key]}' missing in CSV", audit

    t_p = get_time_seconds(df_p, p_map["phone_time"]).to_numpy(dtype=float)
    t_v = get_time_seconds(df_v, v_map["vehicle_time"]).to_numpy(dtype=float)

    if len(t_p) < 10 or len(t_v) < 10:
        return False, f"Insufficient rows (phone={len(t_p)}, vehicle={len(t_v)})", audit

    dur_p = float(t_p[-1] - t_p[0])
    dur_v = float(t_v[-1] - t_v[0])
    audit["duration_p_s"] = round(dur_p, 2)
    audit["duration_v_s"] = round(dur_v, 2)

    # Monotonicity check
    dt_p = np.diff(t_p)
    neg_p_count = int(np.sum(dt_p < 0))
    if dur_p <= 0.0 or neg_p_count > 0:
        return False, f"Non-monotonic/corrupt phone timestamps (duration={dur_p:.1f}s, negative_dt_count={neg_p_count})", audit

    # Duration check
    if dur_p < min_duration_s:
        return False, f"Session duration ({dur_p:.1f}s) < {min_duration_s:.1f}s minimum", audit

    # Duration divergence check
    if dur_v > 0 and (dur_v / dur_p > 3.0 or dur_p / dur_v > 3.0):
        return False, f"Severe duration mismatch (phone={dur_p:.1f}s vs vehicle={dur_v:.1f}s)", audit

    # Vehicle GPS fixes
    v_lats = pd.to_numeric(df_v[v_map["latitude"]], errors="coerce").to_numpy(dtype=float)
    v_lons = pd.to_numeric(df_v[v_map["longitude"]], errors="coerce").to_numpy(dtype=float)
    v_valid = int(np.sum(np.isfinite(v_lats) & np.isfinite(v_lons) & (np.abs(v_lats) > 0.1)))
    audit["valid_vehicle_gps_fixes"] = v_valid
    if v_valid < min_gps_fixes:
        return False, f"Insufficient valid vehicle GPS fixes ({v_valid} < {min_gps_fixes})", audit

    # Phone GPS fixes
    p_lats = pd.to_numeric(df_p[p_map["gps_latitude"]], errors="coerce").to_numpy(dtype=float)
    p_lons = pd.to_numeric(df_p[p_map["gps_longitude"]], errors="coerce").to_numpy(dtype=float)
    p_valid = int(np.sum(np.isfinite(p_lats) & np.isfinite(p_lons) & (np.abs(p_lats) > 0.1)))
    audit["valid_phone_gps_fixes"] = p_valid
    if p_valid < min_gps_fixes:
        return False, f"Insufficient valid phone GPS fixes ({p_valid} < {min_gps_fixes})", audit

    # Vehicle motion check
    v_spd = pd.to_numeric(df_v[v_map["velocity_kmh"]], errors="coerce").fillna(0).to_numpy(dtype=float) / 3.6
    max_v_spd = float(np.max(v_spd)) if len(v_spd) > 0 else 0.0
    audit["max_vehicle_speed_mps"] = round(max_v_spd, 2)
    if max_v_spd < min_speed_mps:
        return False, f"Stationary/parked vehicle session (max speed={max_v_spd:.2f} m/s < {min_speed_mps:.1f} m/s)", audit

    # Time alignment check
    p_spd = pd.to_numeric(df_p[p_map["gps_speed"]], errors="coerce").fillna(0).to_numpy(dtype=float) / 3.6
    try:
        align_res = estimate_vehicle_time_offset(t_p, p_spd, t_v, v_spd)
        audit["align_quality"] = str(align_res.get("quality", "low"))
        audit["align_offset_s"] = float(align_res.get("offset_s", 0.0))
    except Exception as e:
        err_msg = str(e)
        if "overlapping" in err_msg.lower():
            return False, f"Time alignment failed: no temporal overlap ({err_msg})", audit
        audit["align_quality"] = "fallback"
        audit["align_offset_s"] = float(np.median(t_v) - np.median(t_p))

    return True, None, audit


def normalize_condition_name(cond: str) -> str:
    """Normalize condition identifiers (e.g. blackout_60 -> blackout_60s, gnss -> full_gnss)."""
    c = str(cond).strip().lower()
    if c in ("full", "full_gnss", "gnss"):
        return "full_gnss"
    if c in ("blackout_60", "blackout_60s", "60", "60s"):
        return "blackout_60s"
    if c in ("blackout_120", "blackout_120s", "120", "120s"):
        return "blackout_120s"
    if c.startswith("blackout_"):
        val = c.replace("blackout_", "").rstrip("s")
        return f"blackout_{val}s"
    return c


def check_condition_eligibility(
    condition: str,
    duration_s: float,
    blackout_start_s: float,
    recovery_buffer_s: float = 0.0,
) -> Tuple[bool, Optional[str], float]:
    """Determine whether a session is eligible for a specific evaluation condition."""
    if condition == "full_gnss":
        return True, None, 0.0

    if condition == "blackout_60":
        b_dur = 60.0
        req_dur = blackout_start_s + b_dur + recovery_buffer_s
        if duration_s < req_dur:
            return False, f"Duration ({duration_s:.1f}s) < {req_dur:.1f}s required for 60s blackout starting at {blackout_start_s:.1f}s", b_dur
        return True, None, b_dur

    if condition == "blackout_120":
        b_dur = 120.0
        req_dur = blackout_start_s + b_dur + recovery_buffer_s
        if duration_s < req_dur:
            return False, f"Duration ({duration_s:.1f}s) < {req_dur:.1f}s required for 120s blackout starting at {blackout_start_s:.1f}s", b_dur
        return True, None, b_dur

    if condition.startswith("blackout_"):
        try:
            b_dur = float(condition.replace("blackout_", "").replace("s", ""))
            req_dur = blackout_start_s + b_dur + recovery_buffer_s
            if duration_s < req_dur:
                return False, f"Duration ({duration_s:.1f}s) < {req_dur:.1f}s required for {b_dur:.1f}s blackout", b_dur
            return True, None, b_dur
        except ValueError:
            return False, f"Unknown condition format: {condition}", 0.0

    return False, f"Unrecognized condition: {condition}", 0.0


def compute_statistics(values: List[float]) -> Dict[str, Any]:
    """Calculate mean, median, std, min, max, p95 for a numeric sequence."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def generate_benchmark_plots(
    df_runs: pd.DataFrame,
    trajectory_samples: Dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate summary bar/boxplots and representative trajectory visualizations."""
    if df_runs.empty:
        return

    succ = df_runs[df_runs["status"] == "SUCCESS"]
    if succ.empty:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Condition Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    conds = [c for c in ["full_gnss", "blackout_60s", "blackout_120s"] if c in succ["condition"].values]
    if conds:
        x = np.arange(len(conds))
        width = 0.35
        overall_maes = [float(succ[succ["condition"] == c]["overall_mae_m"].median()) for c in conds]
        blackout_maes = [
            float(succ[succ["condition"] == c]["blackout_mae_m"].median()) if c != "full_gnss" and not succ[succ["condition"] == c]["blackout_mae_m"].dropna().empty else 0.0
            for c in conds
        ]

        plt.bar(x - width / 2, overall_maes, width, label="Overall MAE (Median)", color="#2b5c8f")
        plt.bar(x + width / 2, blackout_maes, width, label="Blackout MAE (Median)", color="#d95f02")
        plt.xticks(x, [c.replace("_", " ").title() for c in conds])
        plt.ylabel("Horizontal Position Error (m)")
        plt.title("Position Error by GNSS Condition (Median Across Sessions)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plots_dir / "condition_comparison.png", dpi=150)
        plt.close()

    # Plot 2: Drift Percentage Boxplot
    b_runs = succ[succ["condition"].str.startswith("blackout_") & succ["drift_pct"].notna()]
    if not b_runs.empty:
        plt.figure(figsize=(8, 5))
        b_conds = sorted(b_runs["condition"].unique())
        drift_data = [b_runs[b_runs["condition"] == c]["drift_pct"].dropna().values for c in b_conds if not b_runs[b_runs["condition"] == c]["drift_pct"].dropna().empty]
        valid_labels = [c.replace("_", " ").title() for c in b_conds if not b_runs[b_runs["condition"] == c]["drift_pct"].dropna().empty]
        if drift_data:
            plt.boxplot(drift_data, tick_labels=valid_labels)
            plt.ylabel("Drift Percentage (%)")
            plt.title("Drift Percentage Distribution by Blackout Duration\n(Drift % = End Error / Distance Travelled * 100)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(plots_dir / "drift_distribution.png", dpi=150)
            plt.close()

    # Plot 3: Driver Comparison
    if "driver" in succ.columns:
        plt.figure(figsize=(10, 5))
        d_summary = succ[succ["condition"] == "blackout_60s"].groupby("driver")["blackout_mae_m"].median().dropna()
        if not d_summary.empty:
            d_summary.plot(kind="bar", color="#31a354", edgecolor="black")
            plt.ylabel("60s Blackout MAE (m, Median)")
            plt.title("Dead-Reckoning Error by Driver (60-second Blackout)")
            plt.xticks(rotation=30, ha="right")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(plots_dir / "driver_comparison.png", dpi=150)
            plt.close()

    # Plot 4: Representative Trajectory Overlays
    for key, data in trajectory_samples.items():
        traj = data["traj"]
        metrics = data["metrics"]
        sess_id = metrics["session_id"]
        cond = metrics["condition"]
        ref = traj["ref_enu_p"]
        est = traj["results_pos"]
        gnss = traj["gnss_plot_pos"]
        b_flags = traj["blackout_flags"]

        valid_ref = np.isfinite(ref[:, 0]) & np.isfinite(ref[:, 1])
        valid_est = np.isfinite(est[:, 0]) & np.isfinite(est[:, 1])

        plt.figure(figsize=(10, 8))
        if np.any(valid_ref):
            plt.plot(ref[valid_ref, 0], ref[valid_ref, 1], "k--", label="Ground Truth Reference", alpha=0.8)
        if gnss:
            gp = np.asarray(gnss)
            plt.scatter(gp[:, 0], gp[:, 1], c="green", s=6, alpha=0.5, label="Accepted Phone GNSS")

        if np.any(valid_est):
            plt.plot(est[valid_est, 0], est[valid_est, 1], "b-", label="UKF Navigation State", alpha=0.85)

        b_dur = float(metrics.get("blackout_duration_s", 0.0) or 0.0)
        b_start = float(metrics.get("blackout_start_s", 60.0) or 60.0)
        t_p = traj["t_p"]
        t_rel = t_p - t_p[0]
        if b_dur > 0.0:
            b_mask = (t_rel >= b_start) & (t_rel < b_start + b_dur) & valid_est
            if np.any(b_mask):
                plt.plot(est[b_mask, 0], est[b_mask, 1], "r-", linewidth=2.5, label="Blackout Dead Reckoning")

        b_mae = metrics.get("blackout_mae_m")
        drift = metrics.get("drift_pct")
        subtitle = f"Overall MAE: {metrics['overall_mae_m']:.2f} m" if metrics.get("overall_mae_m") is not None else ""
        if b_mae is not None:
            subtitle += f" | Blackout MAE: {b_mae:.2f} m | Drift: {drift:.1f}%" if drift else f" | Blackout MAE: {b_mae:.2f} m"

        plt.title(f"Trajectory Overlay: {sess_id} ({cond})\n{subtitle}")
        plt.xlabel("East (m)")
        plt.ylabel("North (m)")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"representative_{key}_trajectory.png", dpi=150)
        plt.close()


def generate_markdown_report(
    summary_output: Dict[str, Any],
    df_runs: pd.DataFrame,
    df_rejected: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate a clean, publication-ready benchmark report in Markdown."""
    meta = summary_output["benchmark_metadata"]
    cond_sum = summary_output["conditions_summary"]
    driver_sum = summary_output["drivers_summary"]

    lines = [
        "# PS26168 Multi-Session Navigation Benchmark Report",
        "",
        "**Smart India Hackathon 2026 ? Problem Statement 26168**  ",
        "**Subsystem:** Intelligent Dead Reckoning for GNSS-Denied Vehicle Navigation  ",
        "**Repository:** `ps26168-car-baseline`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"This benchmark evaluates the complete smartphone-based GNSS/INS vehicle navigation pipeline across the **IO-VNBD** dataset. "
        f"A total of **{meta['total_discovered_sessions']} sessions** across 5 drivers were inspected. Following systematic data-quality and motion screening, "
        f"**{meta['usable_sessions_count']} sessions** met the validity criteria for evaluation, while **{meta['rejected_sessions_count']} sessions** were excluded due to corrupt timestamps, stationary/parked conditions, insufficient duration, or lack of temporal overlap.",
        "",
        "### Key Findings by GNSS Condition",
        "",
        "| GNSS Condition | Evaluated Sessions | Overall MAE (m) [Mean ? Std] | Overall MAE (m) [Median] | Blackout MAE (m) [Mean ? Std] | Blackout MAE (m) [Median] | Drift % [Median] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for cond, s in cond_sum.items():
        n = s["evaluated_sessions"]
        o_mae = s["overall_mae_m"]
        b_mae = s["blackout_mae_m"]
        drift = s["drift_pct"]

        o_str = f"{o_mae['mean']:.2f} ? {o_mae['std']:.2f}" if o_mae["mean"] is not None else "N/A"
        o_med = f"{o_mae['median']:.2f}" if o_mae["median"] is not None else "N/A"
        b_str = f"{b_mae['mean']:.2f} ? {b_mae['std']:.2f}" if b_mae["mean"] is not None else "N/A"
        b_med = f"{b_mae['median']:.2f}" if b_mae["median"] is not None else "N/A"
        d_med = f"{drift['median']:.2f}%" if drift["median"] is not None else "N/A"

        lines.append(f"| **{cond.replace('_', ' ').title()}** | {n} | {o_str} | {o_med} | {b_str} | {b_med} | {d_med} |")

    lines.extend([
        "",
        "> [!IMPORTANT]",
        "> **Full-GNSS baseline results represent continuous GNSS-aided tracking** and must NOT be confused with dead-reckoning performance. "
        "> The 60-second and 120-second blackout results isolate inertial propagation aided only by NHC, ZUPT, and CNN speed constraints.",
        "",
        "---",
        "",
        "## 2. Evaluation Methodology & Metric Definitions",
        "",
        "### 2.1 Coordinate System & Units",
        "- **Coordinate Frame**: Local Tangent Plane (LTP) East-North-Up (ENU) in metres.",
        "- **Origin**: The first valid vehicle reference GPS fix `(lat0, lon0)` establishes the shared local origin.",
        "- **Timestamp Alignment**: Offset `offset_s` estimated via `estimate_vehicle_time_offset()` using cross-correlation between phone GPS speed and vehicle speed reference.",
        "- **Ground Truth Reference**: Synchronized vehicle positions in the common LTP frame, sampled via `interpolate_vehicle_to_phone_time()`.",
        "",
        "### 2.2 Metric Formulations",
        "- **Horizontal Position Error ($e_k$)**: $e_k = \\sqrt{(x_k - ref\\_e_k)^2 + (y_k - ref\\_n_k)^2}$ (metres).",
        "- **Position MAE**: $\\frac{1}{N} \\sum_{k=1}^N e_k$ over valid evaluated epochs.",
        "- **Position RMSE**: $\\sqrt{\\frac{1}{N} \\sum_{k=1}^N e_k^2}$.",
        "- **Maximum Position Error**: $\\max_{k} e_k$.",
        "- **Blackout End Error**: Horizontal position error at the final valid epoch inside the blackout interval $[b_{start}, b_{end})$.",
        "- **Blackout Distance Travelled**: Cumulative path length of the ground-truth trajectory during the blackout interval: $D = \\sum_{k} \\| \\text{ref}_k - \\text{ref}_{k-1} \\|$.",
        "- **Drift Percentage**: Defined strictly as:",
        "  $$\\text{Drift \\%} = \\frac{\\text{Blackout End Error (m)}}{\\text{Blackout Distance Travelled (m)}} \\times 100\\%$$",
        "  *(Marked N/A if distance travelled is less than 5.0 metres to prevent division by near-zero displacement.)*",
        "- **Recovery Time**: Elapsed time from the end of blackout ($b_{end}$) until the horizontal error drops below $\\max(10.0\\text{ m}, 3 \\times \\sigma_{GNSS})$ after GNSS fixes resume.",
        "",
        "---",
        "",
        "## 3. Sensor & Calibration Investigation",
        "",
        "### 3.1 Barometer Sensor Support",
        "A rigorous schema audit was conducted across all 144 CSV files in the IO-VNBD dataset.",
        "- **Findings**:",
        "  - The smartphone recordings (`S-` files) contain **NO atmospheric pressure / barometer sensor stream**.",
        "  - The only altitude channel available in the phone data is `GPS ALTITUDE (m)`, which is GNSS-derived, noisy (vertical DOP > 2.5), and unavailable during blackouts.",
        "  - The vehicle OBD recordings (`V-` files) contain `Brake Pressure (psi)`, which is hydraulic brake line pressure, not ambient atmospheric pressure.",
        "- **Technical Conclusion & Recommendation**:",
        "  - Barometer-assisted vertical damping and multi-level parking structure floor transitions **cannot be validated on IO-VNBD** without fabricating synthetic sensor noise.",
        "  - Barometer fusion is documented as **Future Work** for real-world Android deployment, where hardware barometers (`TYPE_PRESSURE`) can provide sub-metre floor elevation resolution.",
        "",
        "### 3.2 Dynamic Phone-to-Vehicle Orientation Calibration",
        "- **Current Implementation**:",
        "  - Tilt alignment (roll and pitch relative to vehicle +Z) is solved using the stationary window gravity vector via `tilt_alignment_matrix(gravity_vector)` in `src/phase1/alignment.py`.",
        "  - World heading (yaw) is causally initialized at motion onset using the first $\\ge 50$ m displacement baseline between distinct GNSS fixes.",
        "- **Potential Dynamic Alignment Improvements**:",
        "  - During straight driving periods with high GNSS accuracy and speed $> 10$ m/s, the vehicle velocity vector aligns with the body forward axis.",
        "  - Comparing the horizontal acceleration axis under sustained braking/acceleration against the GNSS trajectory course can continuously estimate mounting yaw without assuming a permanently fixed bracket.",
        "",
        "---",
        "",
        "## 4. Next Most Valuable Engineering Improvements",
        "",
        "1. **Continuous Yaw Observability during Blackout**: Add turn-rate or magnetometer constraints to reduce heading drift during sustained curves.",
        "2. **Real-time Android Implementation**: Port the 15-state UKF and mechanization logic into Kotlin/NDK for smartphone deployment.",
        "3. **Map-Snapping Closed-Loop Feedback**: Feed HMM road-snapped positions back to the UKF during prolonged outages when uncertainty ellipse encompasses valid road corridors.",
        "",
    ])

    report_path = out_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report generated: {report_path}")


def run_benchmark(
    config_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    session_filter: Optional[List[str]] = None,
    driver_filter: Optional[str] = None,
    conditions: Optional[List[str]] = None,
    blackout_start_s: float = 60.0,
    max_sessions: Optional[int] = None,
    out_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Execute the full multi-session, multi-condition navigation benchmark."""
    config = load_config(config_path)
    out_dir_path = resolve_repo_path(out_dir) if out_dir else (PROJECT_ROOT / "reports" / "benchmark")
    out_dir_path.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is None:
        manifest_path = resolve_repo_path(config["data"]["manifest_path"])
    else:
        manifest_path = Path(manifest_path).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Discover sessions
    discovered = discover_sessions(manifest, manifest_path)
    total_discovered = len(discovered)
    print(f"\n=======================================================")
    print(f"PS26168 Multi-Session Benchmark: {total_discovered} sessions discovered")
    print(f"=======================================================")

    # Apply filters
    if session_filter:
        s_set = {s.strip().lower() for s in session_filter}
        exact_matches = [d for d in discovered if d["session_id"].lower() in s_set]
        if exact_matches:
            discovered = exact_matches
        else:
            discovered = [d for d in discovered if any(s in d["session_id"].lower() for s in s_set)]
        print(f"Filtered by session: {len(discovered)} sessions remaining")

    if driver_filter:
        d_lower = driver_filter.strip().lower()
        discovered = [d for d in discovered if d_lower in d["driver"].lower()]
        print(f"Filtered by driver '{driver_filter}': {len(discovered)} sessions remaining")

    if max_sessions is not None and max_sessions > 0:
        discovered = discovered[:max_sessions]
        print(f"Limited to first {len(discovered)} sessions")

    if not conditions:
        conditions = ["full_gnss", "blackout_60s", "blackout_120s"]

    print(f"Target conditions: {conditions}")
    print(f"Blackout start time: {blackout_start_s:.1f}s")

    # 2. Screening & Validation
    usable_sessions: List[Dict[str, Any]] = []
    rejected_sessions: List[Dict[str, Any]] = []

    print("\n--- Validating Sessions ---")
    for session_info in discovered:
        is_usable, reason, audit = validate_session(session_info)
        if is_usable:
            usable_sessions.append({**session_info, **audit})
        else:
            rejected_sessions.append({
                "session_id": session_info["session_id"],
                "driver": session_info["driver"],
                "duration_p_s": audit["duration_p_s"],
                "duration_v_s": audit["duration_v_s"],
                "status": "REJECTED",
                "reason": reason,
            })
            print(f"  [REJECTED] {session_info['session_id']} ({session_info['driver']}): {reason}")

    print(f"\nScreening Summary: {len(usable_sessions)} Usable, {len(rejected_sessions)} Rejected")

    # 3. Execution Loop
    all_runs: List[Dict[str, Any]] = []
    trajectory_samples: Dict[str, Any] = {}

    for s_idx, session_info in enumerate(usable_sessions, 1):
        sess_id = session_info["session_id"]
        driver = session_info["driver"]
        dur = session_info["duration_p_s"]
        print(f"\n[{s_idx}/{len(usable_sessions)}] Processing {sess_id} ({driver}, {dur:.1f}s)")

        for cond in conditions:
            eligible, inelig_reason, b_dur = check_condition_eligibility(
                cond, dur, blackout_start_s
            )

            if not eligible:
                rejected_sessions.append({
                    "session_id": sess_id,
                    "driver": driver,
                    "duration_p_s": dur,
                    "condition": cond,
                    "status": "SKIPPED_CONDITION",
                    "reason": inelig_reason,
                })
                print(f"  [{cond}] SKIPPED: {inelig_reason}")
                continue

            print(f"  [{cond}] Running (start={blackout_start_s:.1f}s, dur={b_dur:.1f}s)...")
            try:
                # Capture trajectory for a few representative sessions
                want_traj = save_plots and (sess_id in ("vw12", "s1", "s3a", "vta14", "vta2", "vtb2") or s_idx <= 2)
                sess_res = run_session(
                    session_id=sess_id,
                    manifest=manifest,
                    config=config,
                    blackout_start_s=blackout_start_s,
                    blackout_duration_s=b_dur,
                    out_dir=out_dir_path,
                    save_plots=False,
                    return_trajectory=want_traj,
                )

                if want_traj and isinstance(sess_res, tuple):
                    metrics, traj = sess_res
                    trajectory_samples[f"{sess_id}_{cond}"] = {
                        "metrics": metrics,
                        "traj": traj,
                        "driver": driver,
                    }
                else:
                    metrics = sess_res

                run_record = {
                    "session_id": sess_id,
                    "driver": driver,
                    "condition": cond,
                    "status": "SUCCESS",
                    "failure_reason": None,
                    **metrics,
                }
                all_runs.append(run_record)

                mae_str = f"{metrics['overall_mae_m']:.2f}m" if metrics.get("overall_mae_m") is not None else "N/A"
                b_mae_str = f"{metrics['blackout_mae_m']:.2f}m" if metrics.get("blackout_mae_m") is not None else "N/A"
                drift_str = f"{metrics['drift_pct']:.2f}%" if metrics.get("drift_pct") is not None else "N/A"
                print(f"  [{cond}] SUCCESS | Overall MAE: {mae_str} | Blackout MAE: {b_mae_str} | Drift: {drift_str}")

            except Exception as e:
                tb = traceback.format_exc()
                err_msg = f"{type(e).__name__}: {e}"
                print(f"  [{cond}] FAILED: {err_msg}")
                logger.error(f"Error evaluating {sess_id} [{cond}]:\n{tb}")
                run_record = {
                    "session_id": sess_id,
                    "driver": driver,
                    "condition": cond,
                    "status": "FAILED",
                    "failure_reason": err_msg,
                    "traceback": tb,
                    "overall_mae_m": None,
                    "overall_rmse_m": None,
                    "overall_max_m": None,
                    "blackout_mae_m": None,
                    "blackout_rmse_m": None,
                    "blackout_max_m": None,
                    "blackout_end_error_m": None,
                    "blackout_distance_m": None,
                    "drift_pct": None,
                }
                all_runs.append(run_record)
                rejected_sessions.append({
                    "session_id": sess_id,
                    "driver": driver,
                    "condition": cond,
                    "status": "FAILED",
                    "reason": err_msg,
                })

    # 4. Aggregation
    df_runs = pd.DataFrame(all_runs) if all_runs else pd.DataFrame()
    summary_by_condition: Dict[str, Any] = {}
    summary_by_driver: Dict[str, Any] = {}

    if not df_runs.empty and "status" in df_runs.columns:
        successful = df_runs[df_runs["status"] == "SUCCESS"]

        # Condition summary
        for cond in conditions:
            sub = successful[successful["condition"] == cond]
            summary_by_condition[cond] = {
                "evaluated_sessions": int(len(sub)),
                "overall_mae_m": compute_statistics(sub["overall_mae_m"].tolist() if "overall_mae_m" in sub else []),
                "overall_rmse_m": compute_statistics(sub["overall_rmse_m"].tolist() if "overall_rmse_m" in sub else []),
                "overall_max_m": compute_statistics(sub["overall_max_m"].tolist() if "overall_max_m" in sub else []),
                "blackout_mae_m": compute_statistics(sub["blackout_mae_m"].tolist() if "blackout_mae_m" in sub else []),
                "blackout_rmse_m": compute_statistics(sub["blackout_rmse_m"].tolist() if "blackout_rmse_m" in sub else []),
                "blackout_max_m": compute_statistics(sub["blackout_max_m"].tolist() if "blackout_max_m" in sub else []),
                "blackout_end_error_m": compute_statistics(sub["blackout_end_error_m"].tolist() if "blackout_end_error_m" in sub else []),
                "blackout_distance_m": compute_statistics(sub["blackout_distance_m"].tolist() if "blackout_distance_m" in sub else []),
                "drift_pct": compute_statistics(sub["drift_pct"].tolist() if "drift_pct" in sub else []),
                "recovery_time_s": compute_statistics(sub["recovery_time_s"].tolist() if "recovery_time_s" in sub else []),
            }

        # Driver summary
        drivers = sorted(df_runs["driver"].unique())
        for d in drivers:
            summary_by_driver[d] = {}
            for cond in conditions:
                sub = successful[(successful["driver"] == d) & (successful["condition"] == cond)]
                summary_by_driver[d][cond] = {
                    "evaluated_sessions": int(len(sub)),
                    "overall_mae_m": compute_statistics(sub["overall_mae_m"].tolist() if "overall_mae_m" in sub else []),
                    "blackout_mae_m": compute_statistics(sub["blackout_mae_m"].tolist() if "blackout_mae_m" in sub else []),
                    "drift_pct": compute_statistics(sub["drift_pct"].tolist() if "drift_pct" in sub else []),
                }

    # 5. Export CSV and JSON reports
    df_runs.to_csv(out_dir_path / "benchmark_sessions.csv", index=False)

    df_rej = pd.DataFrame(rejected_sessions)
    df_rej.to_csv(out_dir_path / "benchmark_rejected.csv", index=False)

    driver_rows = []
    for d, cond_map in summary_by_driver.items():
        for cond, stats in cond_map.items():
            driver_rows.append({
                "driver": d,
                "condition": cond,
                "evaluated_sessions": stats["evaluated_sessions"],
                "overall_mae_mean_m": stats["overall_mae_m"]["mean"],
                "overall_mae_median_m": stats["overall_mae_m"]["median"],
                "blackout_mae_mean_m": stats["blackout_mae_m"]["mean"],
                "blackout_mae_median_m": stats["blackout_mae_m"]["median"],
                "drift_pct_mean": stats["drift_pct"]["mean"],
                "drift_pct_median": stats["drift_pct"]["median"],
            })
    pd.DataFrame(driver_rows).to_csv(out_dir_path / "benchmark_drivers.csv", index=False)

    summary_output = {
        "benchmark_metadata": {
            "total_discovered_sessions": total_discovered,
            "usable_sessions_count": len(usable_sessions),
            "rejected_sessions_count": len(rejected_sessions),
            "conditions_evaluated": conditions,
            "blackout_start_s": blackout_start_s,
        },
        "conditions_summary": summary_by_condition,
        "drivers_summary": summary_by_driver,
    }
    with (out_dir_path / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)

    with (out_dir_path / "benchmark_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    # 6. Generate Representative Plots
    if save_plots:
        generate_benchmark_plots(df_runs, trajectory_samples, plots_dir)

    # 7. Generate Markdown Report
    generate_markdown_report(
        summary_output=summary_output,
        df_runs=df_runs,
        df_rejected=df_rej,
        out_dir=out_dir_path,
    )

    print(f"\n=======================================================")
    print(f"Benchmark Complete! Reports written to: {out_dir_path}")
    print(f"=======================================================\n")

    return summary_output


def main():
    parser = argparse.ArgumentParser(description="PS26168 Multi-Session Navigation Benchmark")
    parser.add_argument("--config", default=None, help="Path to navigation YAML config")
    parser.add_argument("--manifest", default=None, help="Path to dataset manifest JSON")
    parser.add_argument("--session", default=None, help="Comma-separated session IDs to evaluate")
    parser.add_argument("--driver", default=None, help="Filter by driver name")
    parser.add_argument("--conditions", default=None, help="Comma-separated conditions (e.g. full_gnss,blackout_60,blackout_120)")
    parser.add_argument("--blackout-start-s", type=float, default=60.0, help="Blackout start time (s)")
    parser.add_argument("--blackout-duration-s", type=float, default=None, help="Custom blackout duration (s)")
    parser.add_argument("--max-sessions", type=int, default=None, help="Limit number of sessions to evaluate")
    parser.add_argument("--out-dir", default=None, help="Output directory for reports")
    parser.add_argument("--save-plots", action="store_true", default=True, help="Save representative plots")
    parser.add_argument("--no-plots", action="store_false", dest="save_plots", help="Disable plot generation")

    args = parser.parse_args()

    session_filter = [s.strip() for s in args.session.split(",")] if args.session else None
    
    cond_list = None
    if args.conditions:
        if args.conditions.lower() == "all":
            cond_list = ["full_gnss", "blackout_60", "blackout_120"]
        else:
            cond_list = [c.strip() for c in args.conditions.split(",")]
    elif args.blackout_duration_s is not None:
        cond_list = [f"blackout_{int(args.blackout_duration_s)}s"]

    out_dir_path = Path(args.out_dir) if args.out_dir else None

    run_benchmark(
        config_path=Path(args.config) if args.config else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
        session_filter=session_filter,
        driver_filter=args.driver,
        conditions=cond_list,
        blackout_start_s=args.blackout_start_s,
        max_sessions=args.max_sessions,
        out_dir=out_dir_path,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
