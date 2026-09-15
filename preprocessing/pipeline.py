"""Phase 0 preprocessing skeleton. Calibration/windowing belong to Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PreprocessResult:
    vehicle: pd.DataFrame | None
    smartphone: pd.DataFrame | None
    notes: list[str]


def run_skeleton(
    vehicle: pd.DataFrame | None,
    smartphone: pd.DataFrame | None,
) -> PreprocessResult:
    notes = [
        "Phase 0 skeleton only: no bias calibration, alignment, or windowing yet.",
        "Downstream phases must not treat this output as model-ready tensors.",
    ]
    v_out = vehicle.copy() if vehicle is not None else None
    s_out = smartphone.copy() if smartphone is not None else None
    return PreprocessResult(vehicle=v_out, smartphone=s_out, notes=notes)
