"""Phase 0 plots: available GPS traces and IMU channels. No fabricated drift."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_gps_and_imu(
    vehicle: pd.DataFrame | None,
    smartphone: pd.DataFrame | None,
    output_dir: Path,
    gnss_denied_start_s: float,
    gnss_denied_duration_s: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False
    if vehicle is not None and "GPS Latitude" in vehicle.columns:
        ax.plot(vehicle["GPS Longitude"], vehicle["GPS Latitude"], label="vehicle GPS", linewidth=1.5)
        plotted = True
    if smartphone is not None and "GPS latitude" in smartphone.columns:
        ax.plot(smartphone["GPS longitude"], smartphone["GPS latitude"], label="phone GPS", linewidth=1.0, alpha=0.8)
        plotted = True
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Phase 0 GPS trace (measured fields only)")
    if plotted:
        ax.legend()
        ax.set_aspect("equal", adjustable="datalim")
    path = output_dir / "phase0_gps_trace.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    if smartphone is not None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        t = smartphone["Time since start"] / 1000.0 if "Time since start" in smartphone.columns else range(len(smartphone))
        axes[0].plot(t, smartphone["Accelerometer X"], label="ax")
        axes[0].plot(t, smartphone["Accelerometer Y"], label="ay")
        axes[0].plot(t, smartphone["Accelerometer Z"], label="az")
        axes[0].set_ylabel("accel (m/s^2)")
        axes[0].legend(loc="upper right")
        axes[1].plot(t, smartphone["Gyroscope (Yaw)"], label="yaw")
        axes[1].plot(t, smartphone["Gyroscope (Pitch)"], label="pitch")
        axes[1].plot(t, smartphone["Gyroscope (Roll)"], label="roll")
        axes[1].set_ylabel("gyro (rad/s)")
        axes[1].set_xlabel("time (s)")
        for axis in axes:
            axis.axvspan(
                gnss_denied_start_s,
                gnss_denied_start_s + gnss_denied_duration_s,
                color="orange",
                alpha=0.15,
                label="configured GNSS-denied window",
            )
        axes[0].set_title("Phase 0 smartphone IMU (fixture or real file)")
        path = output_dir / "phase0_phone_imu.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    return saved
