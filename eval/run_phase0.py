"""Phase 0 entry point: config, logging, load/inspect, skeleton preprocess, plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.visualize import plot_gps_and_imu
from preprocessing.fixtures import write_fixture
from preprocessing.loader import clip_duration, find_drive_files, load_csv
from preprocessing.pipeline import run_skeleton
from utils.config import load_config, resolve_path
from utils.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PS26168 Phase 0 setup runner")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--gnss-denied-start-s", type=float, default=None)
    parser.add_argument("--gnss-denied-duration-s", type=float, default=None)
    parser.add_argument("--evaluation-distance-m", type=float, default=None)
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.duration_s is not None:
        cfg["run"]["duration_s"] = args.duration_s
    if args.gnss_denied_start_s is not None:
        cfg["run"]["gnss_denied_start_s"] = args.gnss_denied_start_s
    if args.gnss_denied_duration_s is not None:
        cfg["run"]["gnss_denied_duration_s"] = args.gnss_denied_duration_s
    if args.evaluation_distance_m is not None:
        cfg["run"]["evaluation_distance_m"] = args.evaluation_distance_m
    return cfg


def main() -> int:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    log_dir = resolve_path(cfg, "paths.log_dir")
    logger = setup_logging(log_dir, cfg["logging"]["filename"], cfg["logging"]["level"])

    logger.info("Phase 0 start | scope=%s | two-wheeler=%s", cfg["project"]["scope"], cfg["project"]["two_wheeler_extension"])
    logger.info(
        "Configured duration_s=%.1f gnss_denied=[%.1f, +%.1f] evaluation_distance_m=%.1f",
        cfg["run"]["duration_s"],
        cfg["run"]["gnss_denied_start_s"],
        cfg["run"]["gnss_denied_duration_s"],
        cfg["run"]["evaluation_distance_m"],
    )
    logger.info("Accuracy/drift numbers are not produced in Phase 0.")

    duration_s = float(cfg["run"]["duration_s"])
    hz = float(cfg["run"]["sample_rate_hz"])
    files: dict[str, Path] = {}
    source = cfg["dataset"]["source"]

    if source == "fixture":
        fixture_dir = resolve_path(cfg, "paths.fixture_dir")
        files = write_fixture(fixture_dir, duration_s=duration_s, hz=hz)
        logger.info("Using schema fixture (not real IO-VNBD recordings): %s", files)
    else:
        raw_dir = resolve_path(cfg, "paths.raw_dir")
        files = find_drive_files(raw_dir, cfg["dataset"]["drive_id"])
        if not files:
            logger.error("No CSVs found under %s for drive_id=%s", raw_dir, cfg["dataset"]["drive_id"])
            logger.error("Place IO-VNBD files there or keep dataset.source: fixture")
            return 1

    reports = []
    vehicle = smartphone = None
    if "vehicle" in files:
        vehicle, report = load_csv(files["vehicle"], "vehicle", cfg["dataset"]["expected_imu_hz"])
        vehicle = clip_duration(vehicle, "vehicle", duration_s)
        reports.append(report.as_dict())
        logger.info("Vehicle stream: %s", report.as_dict())
    if "smartphone" in files:
        smartphone, report = load_csv(files["smartphone"], "smartphone", cfg["dataset"]["expected_imu_hz"])
        smartphone = clip_duration(smartphone, "smartphone", duration_s)
        reports.append(report.as_dict())
        logger.info("Smartphone stream: %s", report.as_dict())

    processed = run_skeleton(vehicle, smartphone)
    for note in processed.notes:
        logger.info(note)

    out_dir = resolve_path(cfg, "paths.output_dir")
    plot_dir = out_dir / "plots"
    saved = plot_gps_and_imu(
        processed.vehicle,
        processed.smartphone,
        plot_dir,
        cfg["run"]["gnss_denied_start_s"],
        cfg["run"]["gnss_denied_duration_s"],
    )
    for path in saved:
        logger.info("Wrote plot %s", path)

    summary_path = out_dir / "phase0_schema_report.json"
    summary = {
        "config": {
            "duration_s": cfg["run"]["duration_s"],
            "gnss_denied_start_s": cfg["run"]["gnss_denied_start_s"],
            "gnss_denied_duration_s": cfg["run"]["gnss_denied_duration_s"],
            "evaluation_distance_m": cfg["run"]["evaluation_distance_m"],
            "source": source,
        },
        "streams": reports,
        "plots": [str(p) for p in saved],
        "accuracy_reported": False,
        "note": "Phase 0 does not compute drift. Numbers must come from later executed evaluation on real data.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    logger.info("Phase 0 completed without navigation/filter execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
