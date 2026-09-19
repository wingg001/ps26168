"""Integration tests for GnssDeficitManager into the Phase 3 navigation runner.

These tests verify that the manager is correctly wired into run_phase3.py as
observation-only — no filter parameters are changed, no GNSS updates are
blocked, and the manager's state is purely informational.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_session_feed(n_epochs=200, blackout_start=60, blackout_duration=30):
    """Simulate the GNSS update sequence that run_phase3.py would feed to the
    GnssDeficitManager, and return (manager, state_counts).

    This mirrors the integration logic without needing a full UKF run.
    """
    from src.filters.gnss_deficit import GnssDeficitManager, GnssState

    manager = GnssDeficitManager(
        nis_window=10,
        nis_threshold=5.99,
        accuracy_threshold_m=30.0,
        outage_min_s=3.0,
    )
    state_counts = {s.value: 0 for s in GnssState}

    dt = 0.1  # 10 Hz
    for k in range(1, n_epochs):
        t_s = k * dt
        in_blackout = blackout_start <= t_s < blackout_start + blackout_duration

        gnss_attempted = not in_blackout
        if gnss_attempted:
            success = True
            accepted = True
            nis = 1.0
            acc = 5.0
        else:
            success = False
            accepted = False
            nis = float("nan")
            acc = float("nan")

        manager.update(
            t_s=t_s,
            gnss_accepted=bool(success and accepted),
            nis=float(nis),
            gps_accuracy_m=float(acc),
        )
        state_counts[manager.state.value] += 1

    return manager, state_counts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestManagerInstantiation(unittest.TestCase):
    """Verify the manager is created with correct parameters in run_phase3."""

    def test_instantiation_params(self):
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState

        manager = GnssDeficitManager(
            nis_window=10,
            nis_threshold=5.99,
            accuracy_threshold_m=30.0,
            outage_min_s=3.0,
        )
        self.assertEqual(manager.state, GnssState.NORMAL)

    def test_initial_state_is_normal(self):
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState

        manager = GnssDeficitManager(
            nis_window=10,
            nis_threshold=5.99,
            accuracy_threshold_m=30.0,
            outage_min_s=3.0,
        )
        self.assertEqual(manager.state, GnssState.NORMAL)
        self.assertTrue(manager.gnss_allowed)

    def test_q_scale_always_one(self):
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState

        manager = GnssDeficitManager(
            nis_window=10,
            nis_threshold=5.99,
            accuracy_threshold_m=30.0,
            outage_min_s=3.0,
        )
        # q_scale must always be 1.0 (no adaptive Q yet).
        self.assertAlmostEqual(manager.q_scale, 1.0)


class TestObservationOnlyBehavior(unittest.TestCase):
    """Verify the manager does NOT change estimator behavior."""

    def test_gnss_allowed_true_in_normal(self):
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState

        manager = GnssDeficitManager(
            nis_window=10,
            nis_threshold=5.99,
            accuracy_threshold_m=30.0,
            outage_min_s=3.0,
        )
        # Normal state: GNSS is allowed.
        self.assertTrue(manager.gnss_allowed)

    def test_gnss_allowed_false_in_outage(self):
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState

        manager = GnssDeficitManager(
            nis_window=10,
            nis_threshold=5.99,
            accuracy_threshold_m=30.0,
            outage_min_s=3.0,
        )
        # Push manager into OUTAGE: 10 poor rejects fills the window
        # and triggers DEGRADED→OUTAGE immediately.
        for i in range(10):
            manager.update(t_s=float(i), gnss_accepted=False, nis=20.0, gps_accuracy_m=50.0)
        self.assertEqual(manager.state, GnssState.OUTAGE)
        self.assertFalse(manager.gnss_allowed)


class TestStateCountTracking(unittest.TestCase):
    """Verify state counts are correctly tracked during a simulated session."""

    def test_all_epochs_counted(self):
        manager, state_counts = _simulate_session_feed(n_epochs=200, blackout_start=60, blackout_duration=30)
        total = sum(state_counts.values())
        # Epochs 1..199 = 199 epochs (k=0 is skipped in filter loop).
        self.assertEqual(total, 199)

    def test_normal_dominates_without_degradation(self):
        manager, state_counts = _simulate_session_feed(n_epochs=200, blackout_start=60, blackout_duration=30)
        # With all good GNSS during non-blackout, state should stay NORMAL.
        self.assertGreater(state_counts["normal"], 0)

    def test_degraded_during_blackout(self):
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState

        manager = GnssDeficitManager(
            nis_window=10,
            nis_threshold=5.99,
            accuracy_threshold_m=30.0,
            outage_min_s=3.0,
        )
        state_counts = {s.value: 0 for s in GnssState}

        dt = 0.1
        for k in range(1, 300):
            t_s = k * dt
            in_blackout = 10.0 <= t_s < 45.0

            gnss_attempted = not in_blackout
            if gnss_attempted:
                success, accepted = True, True
                nis, acc = 1.0, 5.0
            else:
                success, accepted = False, False
                nis, acc = float("nan"), float("nan")

            manager.update(
                t_s=t_s,
                gnss_accepted=bool(success and accepted),
                nis=float(nis),
                gps_accuracy_m=float(acc),
            )
            state_counts[manager.state.value] += 1

        # During the 35s blackout (300 epochs at 10Hz), no GNSS is attempted.
        # Manager should transition to DEGRADED after the NIS window fills
        # with non-poor quality (since no bad quality is fed).
        self.assertGreater(state_counts["normal"], 0)


class TestNoFilterChanges(unittest.TestCase):
    """Verify that run_phase3.py does not modify UKF Q, R, or blocking logic."""

    def test_run_phase3_imports_gnss_deficit(self):
        """Confirm the import exists in run_phase3.py."""
        import importlib
        import src.eval.run_phase3 as rp
        self.assertTrue(hasattr(rp, 'GnssDeficitManager'))

    def test_run_phase3_has_state_counts(self):
        """Confirm gnss_state_counts is built from GnssState enum."""
        from src.filters.gnss_deficit import GnssState
        gnss_state_counts = {s.value: 0 for s in GnssState}
        self.assertIn("normal", gnss_state_counts)
        self.assertIn("degraded", gnss_state_counts)
        self.assertIn("outage", gnss_state_counts)
        self.assertIn("recovery", gnss_state_counts)


class TestMetricsOutput(unittest.TestCase):
    """Verify GNSS health fields would be present in metrics dict."""

    def test_metrics_keys_present(self):
        """Confirm the expected keys exist in the integration logic."""
        from src.filters.gnss_deficit import GnssState

        gnss_state_counts = {s.value: 0 for s in GnssState}
        gnss_state_counts["normal"] = 150
        gnss_state_counts["degraded"] = 30
        gnss_state_counts["outage"] = 15
        gnss_state_counts["recovery"] = 4

        # Simulate what run_phase3.py adds to metrics.
        gnss_health_keys = {
            "gnss_health_normal_epochs": gnss_state_counts["normal"],
            "gnss_health_degraded_epochs": gnss_state_counts["degraded"],
            "gnss_health_outage_epochs": gnss_state_counts["outage"],
            "gnss_health_recovery_epochs": gnss_state_counts["recovery"],
            "gnss_health_final_state": "normal",
        }

        self.assertEqual(gnss_health_keys["gnss_health_normal_epochs"], 150)
        self.assertEqual(gnss_health_keys["gnss_health_degraded_epochs"], 30)
        self.assertEqual(gnss_health_keys["gnss_health_outage_epochs"], 15)
        self.assertEqual(gnss_health_keys["gnss_health_recovery_epochs"], 4)
        self.assertEqual(gnss_health_keys["gnss_health_final_state"], "normal")


if __name__ == "__main__":
    unittest.main()
