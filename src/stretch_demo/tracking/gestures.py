"""Gyro gestures: a flick (hands_v2's gripper toggle) and a double shrug (head's
gripper toggle).

Same thresholds and logic as mujoco-sim's joint_config.discrete (cognitive
complexity 18) and head_shoulder.ShrugGestures (29), rewritten as small state
machines so every method stays under the demo's limit of 10. Time comes in as an
argument, so both are testable without a clock.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlickSettings:
    """A flick fires when the rate about its axis stays past `fire` in one direction
    for `hold_s`, while at least `dominance` x the rate about either other axis. It
    can't fire again until the rate drops below `rearm` AND `refractory_s` passed,
    so the return swing isn't read as a flick the other way."""

    fire: float = np.deg2rad(250.0)  # rad/s
    hold_s: float = 0.03
    dominance: float = 1.5
    rearm: float = np.deg2rad(60.0)
    refractory_s: float = 0.35


class FlickDetector:
    def __init__(self, axis: int, sign: float = 1.0, settings: FlickSettings = FlickSettings()):
        self.axis = axis
        self.sign = sign
        self.settings = settings
        self.armed = True
        self.last_fire = float("-inf")
        self.since: float | None = None  # when the rate went (and stayed) past fire
        self.since_sign = 0.0

    def update(self, gyro, now: float) -> float:
        """+1/-1 on the tick a flick fires, else 0."""
        rate = self.sign * gyro[self.axis]
        if abs(rate) < self.settings.rearm:
            self.armed = True
        sign = self._candidate_sign(rate, gyro)
        if sign == 0.0 or sign != self.since_sign:
            self.since = now if sign else None
            self.since_sign = sign
            return 0.0
        if not self._ready(now):
            return 0.0
        self.armed, self.last_fire, self.since, self.since_sign = False, now, None, 0.0
        return sign

    def _candidate_sign(self, rate: float, gyro) -> float:
        s = self.settings
        others = max(abs(g) for axis, g in enumerate(gyro) if axis != self.axis)
        if abs(rate) > s.fire and abs(rate) > s.dominance * others:
            return float(np.sign(rate))
        return 0.0

    def _ready(self, now: float) -> bool:
        s = self.settings
        held = now - self.since >= s.hold_s
        return self.armed and held and now - self.last_fire > s.refractory_s


class DoubleShrug:
    """Shoulder elevation RATE -> "first" (a quick shrug that may start a double)
    and "double". Only upward swings count; the return swing is downward.

    Rate, not angle: speed is what separates a gesture from slow movement, and it
    doesn't depend on where the shoulder rests relative to calibration (an angle
    threshold failed on real shrugs, 2026-10-21). `miss` explains the last quick-ish
    swing that didn't count, for tuning a participant's flick_dps."""

    def __init__(self):
        self.armed = True
        self.fast_since: float | None = None
        self.first_at: float | None = None
        self.burst = False   # in a quick-ish upward swing (past half of flick_dps)
        self.peak_dps = 0.0
        self.mixed = False   # fast enough but not dominant over the other axes
        self.fired = False   # this swing produced a quick shrug
        self.miss: str | None = None
        self.misses = 0

    def update(self, rate: float, other_rate: float, g, now: float) -> str | None:
        """rate: elevation rate (+ = up), rad/s; other_rate: largest |rate| about
        the other two axes; g: GestureSettings."""
        dps, other_dps = np.rad2deg(rate), np.rad2deg(other_rate)
        self._expire_first(g, now)
        self._track_burst(dps, g)
        event = self._quick_shrug(dps, other_dps, g, now)
        self._end_burst(dps, g)
        return event

    def _expire_first(self, g, now: float) -> None:
        if self.first_at is not None and now - self.first_at > g.double_window_s:
            self.first_at = None
            self._miss(f"only one quick shrug: the second must start within {g.double_window_s:.1f} s")

    def _track_burst(self, dps: float, g) -> None:
        if dps > 0.5 * g.flick_dps and not self.burst:
            self.burst, self.peak_dps, self.mixed, self.fired = True, dps, False, False
        if self.burst:
            self.peak_dps = max(self.peak_dps, dps)

    def _quick_shrug(self, dps: float, other_dps: float, g, now: float) -> str | None:
        if abs(dps) < g.rearm_dps:
            self.armed = True
        if not self._fast(dps, other_dps, g):
            self.fast_since = None
            return None
        if self.fast_since is None:
            self.fast_since = now
        if not self.armed or now - self.fast_since < g.flick_hold_s:
            return None
        self.armed, self.fast_since, self.fired = False, None, True
        return self._pair(now)

    def _fast(self, dps: float, other_dps: float, g) -> bool:
        if dps <= g.flick_dps:
            return False
        if dps < g.flick_dominance * other_dps:
            self.mixed = True
            return False
        return True

    def _pair(self, now: float) -> str:
        if self.first_at is None:
            self.first_at = now
            return "first"
        self.first_at = None
        return "double"

    def _end_burst(self, dps: float, g) -> None:
        if not self.burst or dps >= g.rearm_dps:
            return
        self.burst = False
        if self.fired:
            return
        if self.mixed:
            self._miss(f"mixed with other motion: a shrug must be {g.flick_dominance:.1f}x faster "
                       f"than the other axes (peaked {self.peak_dps:.0f} deg/s)")
        else:
            self._miss(f"not fast enough: peaked {self.peak_dps:.0f} deg/s (needs {g.flick_dps:.0f} deg/s)")

    def _miss(self, why: str) -> None:
        self.miss = why
        self.misses += 1
