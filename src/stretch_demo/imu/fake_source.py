"""Synthetic IMUs -> ImuBus, for rehearsing without hardware and for tests.

Publishes at the real batch rate (100 Hz, 4 inertial samples per batch = 400 Hz)
with gravity in the accelerometer, like mujoco-sim's scripts/fake_imu.py. Every
IMU rests at neutral until set_pose() tilts it; the gyro is derived from the pose
change, so the flick/shrug detectors see a tilt the way they'd see a real one.
"""
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from stretch_demo.imu.bus import ImuBus
from stretch_demo.imu.types import ImuBatch, InertialSample, QuaternionSample

HZ = 100.0
SAMPLES_PER_BATCH = 4
UP = np.array([0.0, 0.0, 1.0])  # g, earth frame


class FakeSource:
    def __init__(self, bus: ImuBus, count: int = 2, hz: float = HZ):
        self.bus = bus
        self.count = count
        self.period = 1.0 / hz
        self._lock = threading.Lock()
        self._pose = [Rotation.identity() for _ in range(count)]
        self._prev = [Rotation.identity() for _ in range(count)]
        self._timestamp_us = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def describe(self) -> str:
        return f"{self.count} fake IMUs at neutral"

    def set_pose(self, index: int, roll_deg: float, pitch_deg: float, yaw_deg: float) -> None:
        """Pose relative to the start, as intrinsic ZYX (yaw, pitch, roll) -- the
        same convention the trackers read angles back in."""
        rot = Rotation.from_euler("ZYX", [yaw_deg, pitch_deg, roll_deg], degrees=True)
        with self._lock:
            self._pose[index] = rot

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="fake-imu", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            self.publish_once()
            next_tick += self.period
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    def publish_once(self) -> None:
        """One batch per IMU (also what tests call directly, without the thread)."""
        dt_us = int(self.period * 1e6 / SAMPLES_PER_BATCH)
        start_us = self._timestamp_us
        self._timestamp_us += dt_us * SAMPLES_PER_BATCH
        for index in range(self.count):
            self.bus.publish(self._batch(index, start_us, dt_us))

    def _batch(self, index: int, start_us: int, dt_us: int) -> ImuBatch:
        with self._lock:
            rot, prev = self._pose[index], self._prev[index]
            self._prev[index] = rot
        # Body-frame angular rate that takes prev to rot over one batch period.
        gyro_dps = tuple(np.rad2deg((prev.inv() * rot).as_rotvec() / self.period))
        accel = tuple(rot.inv().apply(UP))  # what a still accelerometer reads
        inertial = tuple(
            InertialSample(start_us + (k + 1) * dt_us, accel, gyro_dps)
            for k in range(SAMPLES_PER_BATCH)
        )
        w_last = tuple(rot.as_quat(scalar_first=True))
        quaternion = (QuaternionSample(start_us + SAMPLES_PER_BATCH * dt_us, w_last),)
        return ImuBatch(index, inertial, quaternion)
