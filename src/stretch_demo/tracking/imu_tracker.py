"""Per-IMU calibration and angles, ported from mujoco-sim's retarget/imu.py
(settling, latest_quaternion) and retarget/joint_config.py (ImuTracker).

Only what the demo controllers use is kept: orientation from the calibrated
neutral, and the bias-removed gyro peak the flick/shrug detectors read. The
hand-velocity integration ('p' inputs) is gone.

Calibration never blocks: controllers feed it a batch per tick, so the viewer
stays live and shows "hold still" while it runs, and it can be rerun (new
visitor) from the terminal at any time.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# Calibration locks only once each IMU's orientation has stayed within SETTLE_TOL
# of its own average for SETTLE_WINDOW_S. The x-IMU3 heading is magnetometer-
# referenced and can still be swinging right after streaming starts even with the
# limb perfectly still; locking then bakes that swing in (see mujoco-sim imu.py).
SETTLE_WINDOW_S = 2.0
SETTLE_TOL = np.deg2rad(2.0)
# ZYX Euler is singular at pitch = +-90 deg; clamp short of it.
PITCH_GUARD = np.deg2rad(85.0)


def latest_quaternion(batch) -> Rotation | None:
    """Newest orientation in a batch. ximu3 is scalar-FIRST (w, x, y, z); scipy's
    default is scalar-last, which silently gives a valid but wrong rotation."""
    if batch is None or not batch.quaternion:
        return None
    newest = max(batch.quaternion, key=lambda q: q.timestamp)
    return Rotation.from_quat(newest.quaternion, scalar_first=True)


def settle_spread(samples: list, now: float) -> float | None:
    """Largest angle of any (t, Rotation) sample from the window's mean, or None
    while the window doesn't span (nearly) SETTLE_WINDOW_S yet."""
    if not samples or now - samples[0][0] < SETTLE_WINDOW_S * 0.9:
        return None
    rots = Rotation.concatenate([rot for _, rot in samples])
    return float(np.max((rots.mean().inv() * rots).magnitude()))


@dataclass
class ImuFrame:
    angles: np.ndarray  # (roll, pitch, yaw) from neutral, rad
    gyro: np.ndarray    # bias-removed angular rate about x/y/z, peak of the newest samples, rad/s


class ImuTracker:
    def __init__(self, label: str):
        self.label = label
        self.ref: Rotation | None = None
        self.gyro_bias = np.zeros(3)
        self.gyro = np.zeros(3)
        self._last_ts: int | None = None
        self._calib: list[tuple[float, Rotation]] = []
        self._calib_gyro: list[tuple[float, np.ndarray]] = []

    @property
    def calibrated(self) -> bool:
        return self.ref is not None

    def reset_calibration(self) -> None:
        self.ref = None
        self._calib = []
        self._calib_gyro = []

    def add_calibration_sample(self, batch, now: float) -> None:
        rot = latest_quaternion(batch)
        if rot is None:
            return
        self._calib.append((now, rot))
        self._calib_gyro += [(now, np.deg2rad(s.gyroscope)) for s in self._new_inertial(batch)]
        self._calib = [c for c in self._calib if now - c[0] <= SETTLE_WINDOW_S]
        self._calib_gyro = [g for g in self._calib_gyro if now - g[0] <= SETTLE_WINDOW_S]

    def spread(self, now: float) -> float | None:
        return settle_spread(self._calib, now)

    def finish_calibration(self, now: float) -> bool:
        """True once the window is settled; sets the neutral and the gyro bias."""
        spread = self.spread(now)
        if spread is None or spread > SETTLE_TOL or not self._calib_gyro:
            return False
        self.ref = Rotation.concatenate([rot for _, rot in self._calib]).mean()
        self.gyro_bias = np.mean([g for _, g in self._calib_gyro], axis=0)
        return True

    def update(self, batch) -> ImuFrame | None:
        rot = latest_quaternion(batch)
        if rot is None or self.ref is None:
            return None
        self._update_gyro(self._new_inertial(batch))
        yaw, pitch, roll = (self.ref.inv() * rot).as_euler("ZYX")
        pitch = float(np.clip(pitch, -PITCH_GUARD, PITCH_GUARD))
        return ImuFrame(np.array([roll, pitch, yaw]), self.gyro.copy())

    def _update_gyro(self, new: list) -> None:
        """Peak |rate| per axis of the new samples. With none new (the sim ticks
        faster than the IMU), keep the last value: reading 0 would look like the
        motion stopped."""
        if not new:
            return
        rates = np.array([np.deg2rad(s.gyroscope) for s in new]) - self.gyro_bias
        peak_rows = np.argmax(np.abs(rates), axis=0)
        self.gyro = rates[peak_rows, np.arange(3)]

    def _new_inertial(self, batch) -> list:
        """Inertial samples newer than any seen before, oldest first. The same
        batch (and, from ImuBus, the same samples) is read several times."""
        if batch is None:
            return []
        last = self._last_ts
        fresh = sorted((s for s in batch.inertial if last is None or s.timestamp > last),
                       key=lambda s: s.timestamp)
        if fresh:
            self._last_ts = fresh[-1].timestamp
        return fresh


class NeutralCalibration:
    """Calibrates a set of trackers together: everyone holds still, all lock at
    once. step() per tick until it returns True."""

    def __init__(self, trackers: dict[int, ImuTracker]):
        self.trackers = trackers

    def restart(self) -> None:
        for tracker in self.trackers.values():
            tracker.reset_calibration()

    def step(self, batches: dict, now: float) -> bool:
        for index, tracker in self.trackers.items():
            tracker.add_calibration_sample(batches.get(index), now)
        return all(tracker.finish_calibration(now) for tracker in self.trackers.values())

    def progress(self, now: float) -> str:
        parts = [f"{t.label}: {_spread_text(t.spread(now))}" for t in self.trackers.values()]
        return "HOLD STILL -- calibrating neutral (" + ", ".join(parts) + ")"


def _spread_text(spread: float | None) -> str:
    if spread is None:
        return "collecting"
    return f"moved {np.rad2deg(spread):.1f} deg"
