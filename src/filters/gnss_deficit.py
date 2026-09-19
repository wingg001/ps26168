"""GNSS deficit state machine for Phase 6.

Tracks GNSS signal health using a sliding window of NIS values and GPS
accuracy reports.  Exposes a deterministic state machine with four
states: NORMAL, DEGRADED, OUTAGE, RECOVERY.

State transitions
-----------------

::

    NORMAL ──poor quality──▸ DEGRADED ──sustained outage──▸ OUTAGE
       ▲                         │                              │
       └──good quality───────────┘                              │
       ▲                                                        │
       └──sustained good──▸ RECOVERY ◂──first acceptance────────┘

Transition rules (evaluated in order each :meth:`update` call):

1. **RECOVERY → NORMAL**: ``accept_streak >= nis_window``
   Sustained good-quality GNSS after an outage.

2. **OUTAGE → RECOVERY**: ``gnss_accepted is True``
   First accepted GNSS fix after an outage.

3. **DEGRADED → OUTAGE**:
   ``reject_streak >= nis_window`` (sustained rejections) **or**
   elapsed time in DEGRADED exceeds ``outage_min_s`` with no
   acceptance.

4. **DEGRADED → NORMAL**:
   Fewer than half of recent NIS/accuracy samples are poor.

5. **NORMAL → DEGRADED**:
   More than half of recent NIS/accuracy samples are poor, **or**
   GNSS is rejected (single rejection triggers degraded state).

Design notes
------------
* ``q_scale`` returns the per-state process-noise multiplier configured
  via the ``q_scales`` parameter.  NORMAL always returns 1.0.
* ``gnss_allowed`` is ``False`` only in the OUTAGE state.  The existing
  chi-square gate remains the primary per-update acceptance mechanism.
* All history buffers are bounded to ``nis_window`` entries.
* Non-finite NIS/accuracy values are silently ignored (not appended to
  history).  This prevents a single NaN from corrupting the window.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Optional


class GnssState(Enum):
    """GNSS health states."""

    NORMAL = "normal"
    DEGRADED = "degraded"
    OUTAGE = "outage"
    RECOVERY = "recovery"


class GnssDeficitManager:
    """Deterministic GNSS health-state tracker.

    Parameters
    ----------
    nis_window : int
        Sliding-window length for NIS and accuracy history.  Must be > 0.
    nis_threshold : float
        NIS value above which a sample is considered *poor*.
    accuracy_threshold_m : float
        GPS accuracy (metres) above which a sample is considered *poor*.
    outage_min_s : float
        Minimum seconds in DEGRADED before a transition to OUTAGE is
        allowed (even if rejections are sustained).  Must be >= 0.
    q_scales : dict, optional
        Per-state process-noise multipliers.  Keys must be
        ``"normal"``, ``"degraded"``, ``"outage"``, ``"recovery"``.
        All values must be finite and > 0.  The ``"normal"`` value is
        forced to 1.0 regardless of input.
    """

    def __init__(
        self,
        *,
        nis_window: int,
        nis_threshold: float,
        accuracy_threshold_m: float,
        outage_min_s: float = 3.0,
        q_scales: Optional[dict] = None,
    ) -> None:
        if nis_window <= 0:
            raise ValueError(f"nis_window must be > 0, got {nis_window}")
        if not isinstance(nis_threshold, (int, float)):
            raise TypeError(f"nis_threshold must be numeric, got {type(nis_threshold)}")
        if not isinstance(accuracy_threshold_m, (int, float)):
            raise TypeError(
                f"accuracy_threshold_m must be numeric, got {type(accuracy_threshold_m)}"
            )
        if not isinstance(outage_min_s, (int, float)):
            raise TypeError(f"outage_min_s must be numeric, got {type(outage_min_s)}")
        if outage_min_s < 0:
            raise ValueError(f"outage_min_s must be >= 0, got {outage_min_s}")
        if not (nis_threshold == nis_threshold):  # NaN check
            raise ValueError("nis_threshold must not be NaN")
        if not (accuracy_threshold_m == accuracy_threshold_m):  # NaN check
            raise ValueError("accuracy_threshold_m must not be NaN")

        self._nis_window = int(nis_window)
        self._nis_threshold = float(nis_threshold)
        self._accuracy_threshold_m = float(accuracy_threshold_m)
        self._outage_min_s = float(outage_min_s)

        # Sliding-window histories (bounded to nis_window entries).
        self._nis_history: deque[float] = deque(maxlen=self._nis_window)
        self._accuracy_history: deque[float] = deque(maxlen=self._nis_window)

        # Consecutive-event counters (reset on state change or streak break).
        self._reject_streak: int = 0
        self._accept_streak: int = 0

        # Timestamps.
        self._state_enter_t: Optional[float] = None
        self._last_update_t: Optional[float] = None

        # Initial state.
        self._state = GnssState.NORMAL

        # Per-state Q multipliers.  NORMAL is always forced to 1.0.
        _default_q = {s.value: 1.0 for s in GnssState}
        if q_scales is not None:
            for key in GnssState:
                if key.value not in q_scales:
                    raise ValueError(
                        f"q_scales missing key {key.value!r}; "
                        f"got keys {sorted(q_scales.keys())}"
                    )
                v = q_scales[key.value]
                if not isinstance(v, (int, float)):
                    raise TypeError(
                        f"q_scales[{key.value!r}] must be numeric, got {type(v)}"
                    )
                if not (v == v):  # NaN check
                    raise ValueError(f"q_scales[{key.value!r}] must not be NaN")
                if v <= 0:
                    raise ValueError(
                        f"q_scales[{key.value!r}] must be > 0, got {v}"
                    )
                _default_q[key.value] = float(v)
        # NORMAL is always 1.0 regardless of input.
        _default_q[GnssState.NORMAL.value] = 1.0
        self._q_scales: dict[str, float] = _default_q

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> GnssState:
        """Current GNSS health state."""
        return self._state

    @property
    def q_scale(self) -> float:
        """Process-noise multiplier for the current state.

        Returns the per-state value configured via ``q_scales``.
        NORMAL always returns 1.0.
        """
        return self._q_scales[self._state.value]

    @property
    def gnss_allowed(self) -> bool:
        """Whether a GNSS update should be *attempted* this epoch.

        * **NORMAL** → ``True``
        * **DEGRADED** → ``True`` (the existing chi-square gate decides)
        * **OUTAGE** → ``False`` (no GNSS signals available / usable)
        * **RECOVERY** → ``True``

        This does **not** bypass the existing chi-square NIS gate.
        """
        return self._state != GnssState.OUTAGE

    def update(
        self,
        *,
        t_s: float,
        gnss_accepted: bool,
        nis: float,
        gps_accuracy_m: float,
    ) -> GnssState:
        """Process one GNSS epoch and advance the state machine.

        Parameters
        ----------
        t_s : float
            Current time in seconds.  Must be >= the previous call.
        gnss_accepted : bool
            Whether the GNSS fix was accepted by the chi-square gate.
        nis : float
            Normalised Innovation Squared from the GNSS update.
            Non-finite values are silently ignored.
        gps_accuracy_m : float
            Phone-reported GPS accuracy in metres.  Non-finite values
            are silently ignored.

        Returns
        -------
        GnssState
            The (possibly updated) state after this epoch.
        """
        if not isinstance(t_s, (int, float)):
            raise TypeError(f"t_s must be numeric, got {type(t_s)}")
        if not (t_s == t_s):  # NaN check
            raise ValueError("t_s must not be NaN")
        if self._last_update_t is not None and t_s < self._last_update_t:
            raise ValueError(
                f"t_s must be monotonically non-decreasing "
                f"(got {t_s} < previous {self._last_update_t})"
            )

        # Append finite values to sliding-window histories.
        if isinstance(nis, (int, float)) and nis == nis:  # finite check
            self._nis_history.append(float(nis))
        if isinstance(gps_accuracy_m, (int, float)) and gps_accuracy_m == gps_accuracy_m:
            self._accuracy_history.append(float(gps_accuracy_m))

        # --- State transitions (evaluated in priority order) ---

        if self._state == GnssState.RECOVERY:
            if gnss_accepted:
                self._accept_streak += 1
            else:
                self._accept_streak = 0

            if self._accept_streak >= self._nis_window:
                self._transition(GnssState.NORMAL, t_s)

        elif self._state == GnssState.OUTAGE:
            if gnss_accepted:
                self._accept_streak = 1
                self._reject_streak = 0
                self._transition(GnssState.RECOVERY, t_s)

        elif self._state == GnssState.DEGRADED:
            if gnss_accepted:
                self._reject_streak = 0
            else:
                self._reject_streak += 1

            elapsed = t_s - self._state_enter_t if self._state_enter_t is not None else 0.0

            if self._reject_streak >= self._nis_window:
                self._transition(GnssState.OUTAGE, t_s)
            elif elapsed >= self._outage_min_s and not gnss_accepted:
                self._transition(GnssState.OUTAGE, t_s)
            elif self._quality_is_good():
                self._transition(GnssState.NORMAL, t_s)

        elif self._state == GnssState.NORMAL:
            if self._quality_is_poor():
                self._transition(GnssState.DEGRADED, t_s)

        self._last_update_t = t_s
        return self._state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transition(self, new_state: GnssState, t_s: float) -> None:
        """Execute a state transition and reset relevant counters."""
        self._state = new_state
        self._state_enter_t = t_s
        if new_state == GnssState.NORMAL:
            self._reject_streak = 0
            self._accept_streak = 0
        elif new_state == GnssState.RECOVERY:
            self._reject_streak = 0
        elif new_state == GnssState.DEGRADED:
            self._accept_streak = 0
            self._reject_streak = 0

    def _quality_is_poor(self) -> bool:
        """True if more than half of recent samples exceed thresholds."""
        n_nis = len(self._nis_history)
        n_acc = len(self._accuracy_history)

        nis_poor = 0
        for v in self._nis_history:
            if v > self._nis_threshold:
                nis_poor += 1

        acc_poor = 0
        for v in self._accuracy_history:
            if v > self._accuracy_threshold_m:
                acc_poor += 1

        nis_poor_ratio = nis_poor / n_nis if n_nis > 0 else 0.0
        acc_poor_ratio = acc_poor / n_acc if n_acc > 0 else 0.0

        return nis_poor_ratio > 0.5 or acc_poor_ratio > 0.5

    def _quality_is_good(self) -> bool:
        """True if fewer than half of recent samples exceed thresholds."""
        n_nis = len(self._nis_history)
        n_acc = len(self._accuracy_history)

        nis_poor = 0
        for v in self._nis_history:
            if v > self._nis_threshold:
                nis_poor += 1

        acc_poor = 0
        for v in self._accuracy_history:
            if v > self._accuracy_threshold_m:
                acc_poor += 1

        nis_poor_ratio = nis_poor / n_nis if n_nis > 0 else 0.0
        acc_poor_ratio = acc_poor / n_acc if n_acc > 0 else 0.0

        return nis_poor_ratio < 0.5 and acc_poor_ratio < 0.5
