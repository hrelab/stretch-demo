"""Per-IMU calibration and angles, ported from mujoco-sim's retarget/imu.py
(settling, latest_quaternion) and retarget/joint_config.py (ImuTracker).

Only what the demo controllers use is kept: orientation from the calibrated
neutral, and the bias-removed gyro peak the flick/shrug detectors read. The
hand-velocity integration ('p' inputs) is gone.

Calibration never blocks: controllers feed it a batch per tick, so the viewer
stays live and shows "hold still" while it runs, and it can be rerun (new
visitor) from the terminal at any time.

Heading comes from the gyros, not the x-IMU3's own orientation. Its fusion
pulls heading toward the magnetometer, and with the IMUs lying still on a desk
it was seen (2026-11-04) swinging heading 28 deg over ~4 s, seconds AFTER
calibration had locked, while the gyros read ~0.3 deg/s. On a tilted IMU that
heading swing also shows up as roll/pitch, and rate controllers (hands_v2) turn
any such offset into a joint that keeps moving. So after calibration the tracker
integrates the bias-removed gyro for orientation and, every update, snaps only
its TILT to the device's (gravity-referenced, trustworthy); heading is left to
the gyro. What's left is slow gyro-bias drift, which the controllers' leaky yaw
baselines already absorb.
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
UP = np.array([0.0, 0.0, 1.0])  # earth frame
MAX_GYRO_DT = 0.05  # s: a longer gap between samples (dropout) isn't integrated across


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


def integrate_gyro(rot: Rotation, samples: list, prev_ts: int | None, bias: np.ndarray) -> Rotation:
    """Advance `rot` (sensor -> earth) by body-frame gyro samples (deg/s, device
    timestamps in us); prev_ts is the sample before the first one, if any."""
    for sample in samples:
        if prev_ts is not None:
            dt = min((sample.timestamp - prev_ts) * 1e-6, MAX_GYRO_DT)
            if dt > 0:
                rot = rot * Rotation.from_rotvec((np.deg2rad(sample.gyroscope) - bias) * dt)
        prev_ts = sample.timestamp
    return rot


def with_tilt_of(rot: Rotation, device: Rotation) -> Rotation:
    """`rot` turned by the smallest rotation (about a horizontal axis, so heading
    is kept) that gives it the same down direction as `device`."""
    have, want = rot.inv().apply(UP), device.inv().apply(UP)
    axis = np.cross(have, want)
    sin = float(np.linalg.norm(axis))
    if sin < 1e-9:
        return rot
    angle = np.arctan2(sin, float(np.dot(have, want)))
    return rot * Rotation.from_rotvec(axis / sin * angle).inv()


@dataclass
class ImuFrame:
    angles: np.ndarray  # (roll, pitch, yaw) from neutral, rad
    gyro: np.ndarray    # bias-removed angular rate about x/y/z, peak of the newest samples, rad/s


class ImuTracker:
    def __init__(self, label: str):
        self.label = label
        self.ref: Rotation | None = None
        self.orientation: Rotation | None = None  # gyro heading + device tilt, sensor -> earth
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
        self.orientation = self.ref
        self.gyro_bias = np.mean([g for _, g in self._calib_gyro], axis=0)
        return True

    def update(self, batch) -> ImuFrame | None:
        rot = latest_quaternion(batch)
        if rot is None or self.ref is None:
            return None
        prev_ts = self._last_ts
        new = self._new_inertial(batch)
        self._update_gyro(new)
        self.orientation = with_tilt_of(integrate_gyro(self.orientation, new, prev_ts, self.gyro_bias), rot)
        yaw, pitch, roll = (self.ref.inv() * self.orientation).as_euler("ZYX")
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
