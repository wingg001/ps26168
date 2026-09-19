"""Deterministic unit tests for src.filters.gnss_deficit — no internet required."""

import unittest

from src.filters.gnss_deficit import GnssDeficitManager, GnssState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_manager(**overrides):
    """Return a GnssDeficitManager with sensible defaults."""
    defaults = dict(
        nis_window=5,
        nis_threshold=6.0,
        accuracy_threshold_m=30.0,
        outage_min_s=3.0,
    )
    defaults.update(overrides)
    return GnssDeficitManager(**defaults)


def _push_good(m, t_s, nis=1.0, acc=5.0):
    """Push a good-quality accepted GNSS epoch."""
    return m.update(t_s=t_s, gnss_accepted=True, nis=nis, gps_accuracy_m=acc)


def _push_bad_reject(m, t_s, nis=20.0, acc=50.0):
    """Push a poor-quality rejected GNSS epoch."""
    return m.update(t_s=t_s, gnss_accepted=False, nis=nis, gps_accuracy_m=acc)


def _push_bad_accept(m, t_s, nis=20.0, acc=50.0):
    """Push a poor-quality but accepted GNSS epoch (passes chi-square gate)."""
    return m.update(t_s=t_s, gnss_accepted=True, nis=nis, gps_accuracy_m=acc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitialState(unittest.TestCase):
    """State must be NORMAL on construction."""

    def test_initial_state(self):
        m = _default_manager()
        self.assertEqual(m.state, GnssState.NORMAL)

    def test_initial_q_scale(self):
        m = _default_manager()
        self.assertEqual(m.q_scale, 1.0)

    def test_initial_gnss_allowed(self):
        m = _default_manager()
        self.assertTrue(m.gnss_allowed)


class TestNormalAccepted(unittest.TestCase):
    """Good-quality accepted GNSS keeps state NORMAL."""

    def test_stays_normal(self):
        m = _default_manager()
        for i in range(10):
            s = _push_good(m, t_s=float(i))
            self.assertEqual(s, GnssState.NORMAL)
        self.assertEqual(m.state, GnssState.NORMAL)


class TestIsolatedPoorSample(unittest.TestCase):
    """A single poor sample in NORMAL should not change state
    (window not yet majority-poor)."""

    def test_single_poor(self):
        m = _default_manager()
        # Fill with good samples.
        for i in range(5):
            _push_good(m, t_s=float(i))
        # One bad sample.
        s = _push_bad_reject(m, t_s=5.0)
        self.assertEqual(s, GnssState.NORMAL)


class TestConsecutivePoorNisDegraded(unittest.TestCase):
    """More than half the window with high NIS → DEGRADED."""

    def test_nis_triggers_degraded(self):
        m = _default_manager()
        # Push 5 good samples to fill the window.
        for i in range(5):
            _push_good(m, t_s=float(i))
        # Now push 4 consecutive bad NIS samples (fills window > 50%).
        for i in range(4):
            s = _push_bad_reject(m, t_s=float(5 + i))
        self.assertEqual(s, GnssState.DEGRADED)

    def test_accuracy_triggers_degraded(self):
        m = _default_manager()
        for i in range(5):
            _push_good(m, t_s=float(i))
        # Push 4 with high accuracy (poor).
        for i in range(4):
            s = m.update(
                t_s=float(5 + i), gnss_accepted=True, nis=1.0, gps_accuracy_m=60.0
            )
        self.assertEqual(s, GnssState.DEGRADED)

    def test_single_rejection_stays_normal(self):
        """A single GNSS rejection in NORMAL should NOT trigger DEGRADED
        when the window quality is still majority-good."""
        m = _default_manager()
        _push_good(m, t_s=0.0)
        s = _push_bad_reject(m, t_s=1.0)
        self.assertEqual(s, GnssState.NORMAL)


class TestSustainedOutage(unittest.TestCase):
    """Sustained deficit in DEGRADED → OUTAGE after outage_min_s."""

    def test_rejection_streak_triggers_outage(self):
        m = _default_manager(nis_window=5)
        # Fill window with good, then push into DEGRADED.
        for i in range(5):
            _push_good(m, t_s=float(i))
        for i in range(4):
            _push_bad_reject(m, t_s=float(5 + i))
        self.assertEqual(m.state, GnssState.DEGRADED)
        # Continue rejecting until reject_streak >= nis_window.
        for i in range(5):
            s = _push_bad_reject(m, t_s=float(9 + i))
        self.assertEqual(s, GnssState.OUTAGE)

    def test_timeout_triggers_outage(self):
        m = _default_manager(nis_window=5, outage_min_s=3.0)
        # Fill window with good.
        for i in range(5):
            _push_good(m, t_s=float(i))
        # Push bad samples. DEGRADED entered at t=7 (3/5=60% poor).
        for i in range(6):
            _push_bad_reject(m, t_s=float(5 + i))
        self.assertEqual(m.state, GnssState.OUTAGE)


class TestGnssRecovery(unittest.TestCase):
    """OUTAGE → RECOVERY on first accepted GNSS fix."""

    def test_recovery(self):
        m = _default_manager(nis_window=5, outage_min_s=3.0)
        # Push to OUTAGE.
        for i in range(5):
            _push_good(m, t_s=float(i))
        for i in range(4):
            _push_bad_reject(m, t_s=float(5 + i))
        for i in range(5):
            _push_bad_reject(m, t_s=float(9 + i))
        self.assertEqual(m.state, GnssState.OUTAGE)
        # First accepted fix → RECOVERY.
        s = _push_good(m, t_s=20.0)
        self.assertEqual(s, GnssState.RECOVERY)


class TestRecoveryToNormal(unittest.TestCase):
    """RECOVERY → NORMAL after sustained good GNSS (accept_streak >= nis_window)."""

    def test_sustained_good_returns_normal(self):
        m = _default_manager(nis_window=5, outage_min_s=3.0)
        # Push to OUTAGE.
        for i in range(5):
            _push_good(m, t_s=float(i))
        for i in range(4):
            _push_bad_reject(m, t_s=float(5 + i))
        for i in range(5):
            _push_bad_reject(m, t_s=float(9 + i))
        self.assertEqual(m.state, GnssState.OUTAGE)
        # Enter RECOVERY.
        s = _push_good(m, t_s=20.0)
        self.assertEqual(s, GnssState.RECOVERY)
        # Push nis_window-1 more good samples (total accept_streak = nis_window).
        for i in range(4):
            s = _push_good(m, t_s=float(21 + i))
        self.assertEqual(s, GnssState.NORMAL)

    def test_rejection_during_recovery_resets_streak(self):
        m = _default_manager(nis_window=3, outage_min_s=10.0)
        # Push to OUTAGE.
        _push_good(m, t_s=0.0)
        for i in range(3):
            _push_bad_reject(m, t_s=float(1 + i))
        for i in range(3):
            _push_bad_reject(m, t_s=float(4 + i))
        self.assertEqual(m.state, GnssState.OUTAGE)
        # Enter RECOVERY.
        _push_good(m, t_s=10.0)
        self.assertEqual(m.state, GnssState.RECOVERY)
        # One rejection resets streak.
        _push_bad_reject(m, t_s=11.0)
        self.assertEqual(m.state, GnssState.RECOVERY)
        # Need 3 consecutive good again.
        _push_good(m, t_s=12.0)
        _push_good(m, t_s=13.0)
        s = _push_good(m, t_s=14.0)
        self.assertEqual(s, GnssState.NORMAL)


class TestDegradedToNormal(unittest.TestCase):
    """DEGRADED → NORMAL when quality improves (fewer than half poor)."""

    def test_quality_improves(self):
        m = _default_manager(nis_window=5)
        # Fill window with good.
        for i in range(5):
            _push_good(m, t_s=float(i))
        # Push into DEGRADED with bad samples (>= 60% poor).
        for i in range(5):
            _push_bad_reject(m, t_s=float(5 + i))
        self.assertEqual(m.state, GnssState.DEGRADED)
        # Push good samples until quality recovers (< 50% poor).
        for i in range(3):
            _push_good(m, t_s=float(10 + i))
        self.assertEqual(m.state, GnssState.NORMAL)


class TestQScaleAlwaysOne(unittest.TestCase):
    """q_scale must be 1.0 in all states when no q_scales are provided."""

    def test_q_scale_in_each_state(self):
        m = _default_manager(nis_window=3, outage_min_s=10.0)
        self.assertEqual(m.q_scale, 1.0)
        # NORMAL
        self.assertEqual(m.state, GnssState.NORMAL)
        self.assertEqual(m.q_scale, 1.0)

        # Push to DEGRADED.
        _push_good(m, t_s=0.0)
        for i in range(3):
            _push_bad_reject(m, t_s=float(1 + i))
        self.assertEqual(m.state, GnssState.DEGRADED)
        self.assertEqual(m.q_scale, 1.0)

        # Push to OUTAGE via rejection streak (>= nis_window).
        for i in range(3):
            _push_bad_reject(m, t_s=float(4 + i))
        self.assertEqual(m.state, GnssState.OUTAGE)
        self.assertEqual(m.q_scale, 1.0)

        # Push to RECOVERY.
        _push_good(m, t_s=10.0)
        self.assertEqual(m.state, GnssState.RECOVERY)
        self.assertEqual(m.q_scale, 1.0)


class TestQScaleConfigured(unittest.TestCase):
    """q_scale must return per-state values from q_scales config."""

    def test_q_scale_per_state(self):
        m = _default_manager(
            nis_window=3, outage_min_s=10.0,
            q_scales={"normal": 1.0, "degraded": 2.0, "outage": 4.0, "recovery": 1.5},
        )
        self.assertEqual(m.state, GnssState.NORMAL)
        self.assertEqual(m.q_scale, 1.0)

        # Push to DEGRADED.
        _push_good(m, t_s=0.0)
        for i in range(3):
            _push_bad_reject(m, t_s=float(1 + i))
        self.assertEqual(m.state, GnssState.DEGRADED)
        self.assertEqual(m.q_scale, 2.0)

        # Push to OUTAGE.
        for i in range(3):
            _push_bad_reject(m, t_s=float(4 + i))
        self.assertEqual(m.state, GnssState.OUTAGE)
        self.assertEqual(m.q_scale, 4.0)

        # Push to RECOVERY.
        _push_good(m, t_s=10.0)
        self.assertEqual(m.state, GnssState.RECOVERY)
        self.assertEqual(m.q_scale, 1.5)

    def test_normal_always_one(self):
        """NORMAL q_scale is forced to 1.0 regardless of config."""
        m = _default_manager(
            q_scales={"normal": 5.0, "degraded": 2.0, "outage": 4.0, "recovery": 1.5},
        )
        self.assertEqual(m.q_scale, 1.0)

    def test_default_q_scales_all_one(self):
        """Without q_scales, all states return 1.0."""
        m = _default_manager()
        self.assertEqual(m.q_scale, 1.0)


class TestQScaleValidation(unittest.TestCase):
    """q_scales constructor parameter must be validated."""

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(
                nis_window=5, nis_threshold=6.0, accuracy_threshold_m=30.0,
                q_scales={"normal": 1.0, "degraded": 2.0},  # missing outage, recovery
            )

    def test_nan_value_raises(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(
                nis_window=5, nis_threshold=6.0, accuracy_threshold_m=30.0,
                q_scales={"normal": 1.0, "degraded": float("nan"), "outage": 4.0, "recovery": 1.5},
            )

    def test_negative_value_raises(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(
                nis_window=5, nis_threshold=6.0, accuracy_threshold_m=30.0,
                q_scales={"normal": 1.0, "degraded": -1.0, "outage": 4.0, "recovery": 1.5},
            )

    def test_zero_value_raises(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(
                nis_window=5, nis_threshold=6.0, accuracy_threshold_m=30.0,
                q_scales={"normal": 1.0, "degraded": 0.0, "outage": 4.0, "recovery": 1.5},
            )

    def test_non_numeric_value_raises(self):
        with self.assertRaises(TypeError):
            GnssDeficitManager(
                nis_window=5, nis_threshold=6.0, accuracy_threshold_m=30.0,
                q_scales={"normal": 1.0, "degraded": "bad", "outage": 4.0, "recovery": 1.5},
            )


class TestGnssAllowedBehavior(unittest.TestCase):
    """gnss_allowed must be False only in OUTAGE."""

    def test_normal_true(self):
        m = _default_manager()
        self.assertTrue(m.gnss_allowed)

    def test_degraded_true(self):
        m = _default_manager(nis_window=3)
        _push_good(m, t_s=0.0)
        for i in range(3):
            _push_bad_reject(m, t_s=float(1 + i))
        self.assertEqual(m.state, GnssState.DEGRADED)
        self.assertTrue(m.gnss_allowed)

    def test_outage_false(self):
        m = _default_manager(nis_window=3, outage_min_s=10.0)
        _push_good(m, t_s=0.0)
        for i in range(3):
            _push_bad_reject(m, t_s=float(1 + i))
        for i in range(3):
            _push_bad_reject(m, t_s=float(4 + i))
        self.assertEqual(m.state, GnssState.OUTAGE)
        self.assertFalse(m.gnss_allowed)

    def test_recovery_true(self):
        m = _default_manager(nis_window=3, outage_min_s=10.0)
        _push_good(m, t_s=0.0)
        for i in range(3):
            _push_bad_reject(m, t_s=float(1 + i))
        for i in range(3):
            _push_bad_reject(m, t_s=float(4 + i))
        _push_good(m, t_s=10.0)
        self.assertEqual(m.state, GnssState.RECOVERY)
        self.assertTrue(m.gnss_allowed)


class TestInvalidConstructor(unittest.TestCase):
    """Constructor must reject invalid parameters."""

    def test_nis_window_zero(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(nis_window=0, nis_threshold=6.0,
                               accuracy_threshold_m=30.0)

    def test_nis_window_negative(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(nis_window=-1, nis_threshold=6.0,
                               accuracy_threshold_m=30.0)

    def test_outage_min_negative(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(nis_window=5, nis_threshold=6.0,
                               accuracy_threshold_m=30.0,
                               outage_min_s=-1.0)

    def test_nis_threshold_nan(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(nis_window=5, nis_threshold=float("nan"),
                               accuracy_threshold_m=30.0)

    def test_accuracy_threshold_nan(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(nis_window=5, nis_threshold=6.0,
                               accuracy_threshold_m=float("nan"))

    def test_nis_threshold_type(self):
        with self.assertRaises(TypeError):
            GnssDeficitManager(nis_window=5, nis_threshold="bad",
                               accuracy_threshold_m=30.0)


class TestNonFiniteInput(unittest.TestCase):
    """Non-finite NIS/accuracy must be silently ignored."""

    def test_nan_nis_ignored(self):
        m = _default_manager(nis_window=3)
        s = m.update(t_s=0.0, gnss_accepted=True, nis=float("nan"),
                     gps_accuracy_m=5.0)
        self.assertEqual(s, GnssState.NORMAL)
        self.assertEqual(len(m._nis_history), 0)

    def test_nan_accuracy_ignored(self):
        m = _default_manager(nis_window=3)
        s = m.update(t_s=0.0, gnss_accepted=True, nis=1.0,
                     gps_accuracy_m=float("nan"))
        self.assertEqual(s, GnssState.NORMAL)
        self.assertEqual(len(m._accuracy_history), 0)

    def test_inf_nis_ignored(self):
        m = _default_manager(nis_window=3)
        s = m.update(t_s=0.0, gnss_accepted=True, nis=float("inf"),
                     gps_accuracy_m=5.0)
        # inf is finite per IEEE 754? No, inf == inf is True but inf is not
        # ignored by the `nis == nis` check (inf == inf is True in Python).
        # However, inf > threshold so it IS appended and counts as poor.
        self.assertEqual(len(m._nis_history), 1)

    def test_nan_t_s_raises(self):
        m = _default_manager()
        with self.assertRaises(ValueError):
            m.update(t_s=float("nan"), gnss_accepted=True, nis=1.0,
                     gps_accuracy_m=5.0)


class TestMonotonicTimestamp(unittest.TestCase):
    """Timestamps must be monotonically non-decreasing."""

    def test_decreasing_raises(self):
        m = _default_manager()
        m.update(t_s=10.0, gnss_accepted=True, nis=1.0, gps_accuracy_m=5.0)
        with self.assertRaises(ValueError):
            m.update(t_s=9.0, gnss_accepted=True, nis=1.0, gps_accuracy_m=5.0)

    def test_same_timestamp_allowed(self):
        m = _default_manager()
        s1 = m.update(t_s=5.0, gnss_accepted=True, nis=1.0, gps_accuracy_m=5.0)
        s2 = m.update(t_s=5.0, gnss_accepted=True, nis=1.0, gps_accuracy_m=5.0)
        self.assertEqual(s1, GnssState.NORMAL)
        self.assertEqual(s2, GnssState.NORMAL)


class TestRepeatedSameTimestamp(unittest.TestCase):
    """Multiple updates at the same timestamp should work deterministically."""

    def test_repeated(self):
        m = _default_manager(nis_window=3)
        for _ in range(5):
            s = m.update(t_s=0.0, gnss_accepted=False, nis=20.0,
                         gps_accuracy_m=50.0)
        # State transitions should still occur even with same timestamp.
        self.assertIn(s, (GnssState.DEGRADED, GnssState.OUTAGE))


class TestFullCycle(unittest.TestCase):
    """End-to-end: NORMAL → DEGRADED → OUTAGE → RECOVERY → NORMAL."""

    def test_full_cycle(self):
        m = _default_manager(nis_window=3, outage_min_s=2.0)

        # 1. NORMAL — good samples.
        for i in range(3):
            _push_good(m, t_s=float(i))
        self.assertEqual(m.state, GnssState.NORMAL)

        # 2. DEGRADED — bad samples fill window.
        for i in range(3):
            _push_bad_reject(m, t_s=float(3 + i))
        self.assertEqual(m.state, GnssState.DEGRADED)

        # 3. OUTAGE — sustained deficit past outage_min_s.
        for i in range(3):
            _push_bad_reject(m, t_s=float(7 + i))
        self.assertEqual(m.state, GnssState.OUTAGE)

        # 4. RECOVERY — first accepted fix.
        _push_good(m, t_s=20.0)
        self.assertEqual(m.state, GnssState.RECOVERY)

        # 5. NORMAL — sustained good.
        for i in range(2):
            _push_good(m, t_s=float(21 + i))
        self.assertEqual(m.state, GnssState.NORMAL)


class TestNoFixTimeout(unittest.TestCase):
    """No-fix timeout: NORMAL/DEGRADED -> OUTAGE when no GNSS observation for timeout_s."""

    def test_no_timeout_before_threshold(self):
        m = _default_manager(no_fix_timeout_s=5.0)
        _push_good(m, t_s=0.0)
        # check_timeout at t=4.9 — below timeout.
        s = m.check_timeout(t_s=4.9)
        self.assertEqual(s, GnssState.NORMAL)

    def test_timeout_exactly_at_threshold(self):
        m = _default_manager(no_fix_timeout_s=5.0)
        _push_good(m, t_s=0.0)
        # check_timeout at t=5.0 — exactly at timeout.
        s = m.check_timeout(t_s=5.0)
        self.assertEqual(s, GnssState.OUTAGE)

    def test_timeout_after_threshold(self):
        m = _default_manager(no_fix_timeout_s=5.0)
        _push_good(m, t_s=0.0)
        s = m.check_timeout(t_s=10.0)
        self.assertEqual(s, GnssState.OUTAGE)

    def test_accepted_gnss_resets_activity(self):
        m = _default_manager(no_fix_timeout_s=5.0)
        _push_good(m, t_s=0.0)
        m.check_timeout(t_s=3.0)
        # Another GNSS observation at t=4 resets the timer.
        _push_good(m, t_s=4.0)
        # check_timeout at t=8.0 — only 4s since last activity.
        s = m.check_timeout(t_s=8.0)
        self.assertEqual(s, GnssState.NORMAL)

    def test_rejected_gnss_resets_activity(self):
        m = _default_manager(no_fix_timeout_s=5.0)
        _push_good(m, t_s=0.0)
        m.check_timeout(t_s=3.0)
        # A rejected GNSS observation at t=4 still counts as activity.
        _push_bad_reject(m, t_s=4.0)
        # check_timeout at t=8.0 — only 4s since last activity.
        s = m.check_timeout(t_s=8.0)
        self.assertEqual(s, GnssState.NORMAL)

    def test_no_fix_period_transitions_to_outage(self):
        m = _default_manager(no_fix_timeout_s=5.0)
        _push_good(m, t_s=0.0)
        # Simulate 6 seconds with no GNSS fixes.
        for t in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            s = m.check_timeout(t_s=t)
        self.assertEqual(s, GnssState.OUTAGE)

    def test_timeout_from_degraded(self):
        m = _default_manager(nis_window=3, no_fix_timeout_s=5.0)
        # Push into DEGRADED.
        _push_good(m, t_s=0.0)
        for i in range(3):
            _push_bad_reject(m, t_s=float(1 + i))
        self.assertEqual(m.state, GnssState.DEGRADED)
        # Simulate 6 seconds with no GNSS fixes.
        for t in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            s = m.check_timeout(t_s=t)
        self.assertEqual(s, GnssState.OUTAGE)

    def test_recovery_probe_after_timeout_outage(self):
        m = _default_manager(nis_window=3, no_fix_timeout_s=5.0)
        _push_good(m, t_s=0.0)
        for t in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            m.check_timeout(t_s=t)
        self.assertEqual(m.state, GnssState.OUTAGE)
        # A recovery probe (accepted GNSS) should enter RECOVERY.
        s = _push_good(m, t_s=7.0)
        self.assertEqual(s, GnssState.RECOVERY)

    def test_backward_timestamp_raises(self):
        m = _default_manager(no_fix_timeout_s=5.0)
        _push_good(m, t_s=0.0)
        m.check_timeout(t_s=3.0)
        with self.assertRaises(ValueError):
            m.check_timeout(t_s=2.0)

    def test_nan_timestamp_raises(self):
        m = _default_manager(no_fix_timeout_s=5.0)
        with self.assertRaises(ValueError):
            m.check_timeout(t_s=float("nan"))

    def test_no_fake_gnss_observation(self):
        """check_timeout must not create a GNSS observation."""
        m = _default_manager(no_fix_timeout_s=5.0)
        _push_good(m, t_s=0.0)
        n_nis_before = len(m._nis_history)
        n_acc_before = len(m._accuracy_history)
        m.check_timeout(t_s=6.0)
        # Histories should not grow from check_timeout.
        self.assertEqual(len(m._nis_history), n_nis_before)
        self.assertEqual(len(m._accuracy_history), n_acc_before)

    def test_no_timeout_without_activity(self):
        """If no GNSS observation has ever been received, timeout cannot trigger."""
        m = _default_manager(no_fix_timeout_s=5.0)
        # Never called update — _last_gnss_activity_t is None.
        s = m.check_timeout(t_s=100.0)
        self.assertEqual(s, GnssState.NORMAL)

    def test_timeout_does_not_affect_recovery(self):
        """check_timeout in RECOVERY state should not change state."""
        m = _default_manager(nis_window=3, no_fix_timeout_s=5.0)
        # Push to OUTAGE.
        _push_good(m, t_s=0.0)
        for i in range(3):
            _push_bad_reject(m, t_s=float(1 + i))
        for i in range(3):
            _push_bad_reject(m, t_s=float(4 + i))
        self.assertEqual(m.state, GnssState.OUTAGE)
        # Enter RECOVERY.
        _push_good(m, t_s=10.0)
        self.assertEqual(m.state, GnssState.RECOVERY)
        # Long timeout should not change RECOVERY.
        s = m.check_timeout(t_s=100.0)
        self.assertEqual(s, GnssState.RECOVERY)


class TestNoFixTimeoutValidation(unittest.TestCase):
    """no_fix_timeout_s constructor parameter must be validated."""

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(
                nis_window=5, nis_threshold=6.0, accuracy_threshold_m=30.0,
                no_fix_timeout_s=0.0,
            )

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(
                nis_window=5, nis_threshold=6.0, accuracy_threshold_m=30.0,
                no_fix_timeout_s=-1.0,
            )

    def test_nan_raises(self):
        with self.assertRaises(ValueError):
            GnssDeficitManager(
                nis_window=5, nis_threshold=6.0, accuracy_threshold_m=30.0,
                no_fix_timeout_s=float("nan"),
            )

    def test_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            GnssDeficitManager(
                nis_window=5, nis_threshold=6.0, accuracy_threshold_m=30.0,
                no_fix_timeout_s="bad",
            )


if __name__ == "__main__":
    unittest.main()
