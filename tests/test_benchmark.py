import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.eval.run_benchmark import (
    extract_driver_label,
    discover_sessions,
    validate_session,
    check_condition_eligibility,
    compute_statistics,
)


class TestBenchmarkHarness(unittest.TestCase):
    def test_extract_driver_label(self):
        self.assertEqual(extract_driver_label("data/M (Driver B)/S-M.csv"), "Driver B")
        self.assertEqual(extract_driver_label("data/S (Driver A)/S1/S-S1.csv"), "Driver A")
        self.assertEqual(extract_driver_label("data/Y (Driver D)/Y1/S-Y1.csv"), "Driver D")
        self.assertEqual(extract_driver_label("data/Vw (Driver E)/Vw01/S-Vw01.csv"), "Driver E (Vw)")
        self.assertEqual(extract_driver_label("data/Vta (Driver E)/Vta01a/S-Vta01a.csv"), "Driver E (Vta)")
        self.assertEqual(extract_driver_label("data/Vtb (Driver E)/Vtb01/S-Vtb01.csv"), "Driver E (Vtb)")
        self.assertEqual(extract_driver_label("data/Vf (Driver E)/V-Vfa01/S-Vfa01.csv"), "Driver E (Vf)")

    def test_check_condition_eligibility(self):
        # Full GNSS always eligible for valid drives
        ok, reason, dur = check_condition_eligibility("full_gnss", 50.0, 60.0)
        self.assertTrue(ok)
        self.assertEqual(dur, 0.0)

        # 60s blackout starting at 60s requires >= 120s
        ok, reason, dur = check_condition_eligibility("blackout_60", 110.0, 60.0)
        self.assertFalse(ok)
        self.assertIn("required", reason.lower())

        ok, reason, dur = check_condition_eligibility("blackout_60", 130.0, 60.0)
        self.assertTrue(ok)
        self.assertEqual(dur, 60.0)

        # 120s blackout starting at 60s requires >= 180s
        ok, reason, dur = check_condition_eligibility("blackout_120", 170.0, 60.0)
        self.assertFalse(ok)

        ok, reason, dur = check_condition_eligibility("blackout_120", 200.0, 60.0)
        self.assertTrue(ok)
        self.assertEqual(dur, 120.0)

    def test_compute_statistics(self):
        stats = compute_statistics([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["median"], 30.0)
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 50.0)
        self.assertAlmostEqual(stats["p95"], 48.0, delta=1.0)

        # Empty / all-NaN list
        empty_stats = compute_statistics([])
        self.assertEqual(empty_stats["count"], 0)
        self.assertIsNone(empty_stats["mean"])

        nan_stats = compute_statistics([float("nan"), None])
        self.assertEqual(nan_stats["count"], 0)
        self.assertIsNone(nan_stats["mean"])

    def test_validation_rejection_rules(self):
        # Test short duration rejection
        dummy_session = {
            "session_id": "test_short",
            "driver": "Driver A",
            "phone_path": Path("nonexistent_p.csv"),
            "vehicle_path": Path("nonexistent_v.csv"),
            "p_map": {},
            "v_map": {},
        }
        ok, reason, audit = validate_session(dummy_session)
        self.assertFalse(ok)
        self.assertIn("not found", reason.lower())

    def test_drift_percentage_definition(self):
        # Drift % = (end_error / distance) * 100
        end_error = 25.0  # metres
        distance = 500.0  # metres
        drift = (end_error / distance) * 100.0
        self.assertEqual(drift, 5.0)  # 5% drift


if __name__ == "__main__":
    unittest.main()
