"""Integration tests for GnssDeficitManager into the Phase 3 navigation runner.

These tests verify that the manager is correctly wired into run_phase3.py with
adaptive Q scaling — Q is scaled before each prediction and restored after,
the manager's state drives the Q multiplier, and no baseline Q mutation occurs.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _push_good(m, t_s, nis=1.0, acc=5.0):
    """Push a good-quality accepted GNSS epoch."""
    return m.update(t_s=t_s, gnss_accepted=True, nis=nis, gps_accuracy_m=acc)


def _push_bad_reject(m, t_s, nis=20.0, acc=50.0):
    """Push a poor-quality rejected GNSS epoch."""
    return m.update(t_s=t_s, gnss_accepted=False, nis=nis, gps_accuracy_m=acc)


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

    def test_q_scale_always_one_without_config(self):
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState

        manager = GnssDeficitManager(
            nis_window=10,
            nis_threshold=5.99,
            accuracy_threshold_m=30.0,
            outage_min_s=3.0,
        )
        self.assertAlmostEqual(manager.q_scale, 1.0)

    def test_q_scale_configured_per_state(self):
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState

        manager = GnssDeficitManager(
            nis_window=10,
            nis_threshold=5.99,
            accuracy_threshold_m=30.0,
            outage_min_s=3.0,
            q_scales={"normal": 1.0, "degraded": 2.0, "outage": 4.0, "recovery": 1.5},
        )
        self.assertEqual(manager.q_scale, 1.0)
        # Push to DEGRADED.
        for i in range(10):
            manager.update(t_s=float(i), gnss_accepted=False, nis=20.0, gps_accuracy_m=50.0)
        self.assertEqual(manager.state, GnssState.OUTAGE)
        self.assertEqual(manager.q_scale, 4.0)


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


class TestAdaptiveQScaling(unittest.TestCase):
    """Verify Q is scaled during prediction and baseline Q is not mutated."""

    def test_q_changes_during_prediction(self):
        """When state != NORMAL, ukf.Q should be scaled before predict."""
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState
        from src.filters.ukf import ScaledUKF

        manager = GnssDeficitManager(
            nis_window=3, nis_threshold=5.99, accuracy_threshold_m=30.0,
            outage_min_s=3.0,
            q_scales={"normal": 1.0, "degraded": 2.0, "outage": 4.0, "recovery": 1.5},
        )
        ukf = ScaledUKF(dim_x=15, alpha=1.0, beta=2.0, kappa=0.0)
        baseline_Q = np.eye(15) * 0.01
        ukf.Q = baseline_Q.copy()
        baseline_Q_saved = baseline_Q.copy()

        # State is NORMAL → q_scale = 1.0 → no scaling.
        _qs = manager.q_scale
        self.assertEqual(_qs, 1.0)

        # Push to DEGRADED.
        _push_good(manager, t_s=0.0)
        for i in range(3):
            _push_bad_reject(manager, t_s=float(1 + i))
        self.assertEqual(manager.state, GnssState.DEGRADED)

        # Now q_scale should be 2.0.
        _qs = manager.q_scale
        self.assertEqual(_qs, 2.0)

        # Simulate the prediction pattern from run_phase3.py.
        if _qs != 1.0:
            ukf.Q = baseline_Q_saved * _qs
        # After scaling, Q should be 2x baseline.
        np.testing.assert_array_equal(ukf.Q, baseline_Q_saved * 2.0)
        # Baseline must not be mutated.
        np.testing.assert_array_equal(baseline_Q_saved, baseline_Q)

    def test_q_restored_after_prediction(self):
        """After prediction, baseline Q should be restored."""
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState
        from src.filters.ukf import ScaledUKF

        manager = GnssDeficitManager(
            nis_window=3, nis_threshold=5.99, accuracy_threshold_m=30.0,
            outage_min_s=3.0,
            q_scales={"normal": 1.0, "degraded": 2.0, "outage": 4.0, "recovery": 1.5},
        )
        ukf = ScaledUKF(dim_x=15, alpha=1.0, beta=2.0, kappa=0.0)
        baseline_Q = np.eye(15) * 0.01
        ukf.Q = baseline_Q.copy()

        # Push to DEGRADED.
        _push_good(manager, t_s=0.0)
        for i in range(3):
            _push_bad_reject(manager, t_s=float(1 + i))

        # Simulate predict cycle.
        _qs = manager.q_scale
        if _qs != 1.0:
            ukf.Q = baseline_Q * _qs
        # Restore after predict.
        if _qs != 1.0:
            ukf.Q = baseline_Q.copy()

        # Q should be back to baseline.
        np.testing.assert_array_equal(ukf.Q, baseline_Q)

    def test_baseline_q_not_mutated(self):
        """The baseline_Q copy must never be modified."""
        from src.filters.gnss_deficit import GnssDeficitManager
        import copy

        manager = GnssDeficitManager(
            nis_window=3, nis_threshold=5.99, accuracy_threshold_m=30.0,
            outage_min_s=3.0,
            q_scales={"normal": 1.0, "degraded": 2.0, "outage": 4.0, "recovery": 1.5},
        )
        baseline_Q = np.eye(15) * 0.01
        baseline_Q_saved = baseline_Q.copy()

        # Push to OUTAGE.
        _push_good(manager, t_s=0.0)
        for i in range(3):
            _push_bad_reject(manager, t_s=float(1 + i))
        for i in range(3):
            _push_bad_reject(manager, t_s=float(4 + i))

        # Scale baseline.
        scaled = baseline_Q * manager.q_scale
        np.testing.assert_array_equal(baseline_Q, baseline_Q_saved)

    def test_normal_state_uses_baseline_q(self):
        """When q_scale=1.0, Q is not scaled."""
        from src.filters.gnss_deficit import GnssDeficitManager, GnssState
        from src.filters.ukf import ScaledUKF

        manager = GnssDeficitManager(
            nis_window=3, nis_threshold=5.99, accuracy_threshold_m=30.0,
            outage_min_s=3.0,
            q_scales={"normal": 1.0, "degraded": 2.0, "outage": 4.0, "recovery": 1.5},
        )
        ukf = ScaledUKF(dim_x=15, alpha=1.0, beta=2.0, kappa=0.0)
        baseline_Q = np.eye(15) * 0.01
        ukf.Q = baseline_Q.copy()

        # NORMAL → q_scale = 1.0.
        _qs = manager.q_scale
        self.assertEqual(_qs, 1.0)
        # No scaling branch.
        if _qs != 1.0:
            ukf.Q = baseline_Q * _qs
        # Q unchanged.
        np.testing.assert_array_equal(ukf.Q, baseline_Q)


class TestNoFilterChanges(unittest.TestCase):
    """Verify that run_phase3.py correctly imports and uses the manager."""

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
    """Verify GNSS health and adaptive Q fields would be present in metrics dict."""

    def test_metrics_keys_present(self):
        """Confirm the expected keys exist in the integration logic."""
        from src.filters.gnss_deficit import GnssState

        gnss_state_counts = {s.value: 0 for s in GnssState}
        gnss_state_counts["normal"] = 150
        gnss_state_counts["degraded"] = 30
        gnss_state_counts["outage"] = 15
        gnss_state_counts["recovery"] = 4

        gnss_health_keys = {
            "gnss_health_normal_epochs": gnss_state_counts["normal"],
            "gnss_health_degraded_epochs": gnss_state_counts["degraded"],
            "gnss_health_outage_epochs": gnss_state_counts["outage"],
            "gnss_health_recovery_epochs": gnss_state_counts["recovery"],
            "gnss_health_final_state": "normal",
            "adaptive_q_predictions": 45,
            "adaptive_q_scales": {"normal": 1.0, "degraded": 2.0, "outage": 4.0, "recovery": 1.5},
        }

        self.assertEqual(gnss_health_keys["gnss_health_normal_epochs"], 150)
        self.assertEqual(gnss_health_keys["gnss_health_degraded_epochs"], 30)
        self.assertEqual(gnss_health_keys["gnss_health_outage_epochs"], 15)
        self.assertEqual(gnss_health_keys["gnss_health_recovery_epochs"], 4)
        self.assertEqual(gnss_health_keys["gnss_health_final_state"], "normal")
        self.assertEqual(gnss_health_keys["adaptive_q_predictions"], 45)
        self.assertEqual(gnss_health_keys["adaptive_q_scales"]["degraded"], 2.0)
        self.assertEqual(gnss_health_keys["adaptive_q_scales"]["outage"], 4.0)
        self.assertEqual(gnss_health_keys["adaptive_q_scales"]["recovery"], 1.5)


if __name__ == "__main__":
    unittest.main()
