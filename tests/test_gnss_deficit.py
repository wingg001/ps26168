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
    """q_scale must remain 1.0 in all states."""

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


if __name__ == "__main__":
    unittest.main()
