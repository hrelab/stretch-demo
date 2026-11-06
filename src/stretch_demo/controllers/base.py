"""What every demo controller shares.

None of them switch modes: each IMU motion always drives the same thing, so a
visitor never has to remember which mode they're in. The base is driven by
turning past a threshold (as imu_hands_vel did), the gripper by a gesture.

The harness calls setup() once (and again for the next visitor), then
get_ctrl() every tick:
- while calibrating (the start, `c`, `n`), it holds the robot still and the
  overlay says "hold still"; calibration never blocks, so the viewer stays live;
- with no data from an IMU, it holds and says which one;
- otherwise the subclass's control() updates setpoints/gripper and returns the
  base drive level, and compose() turns that into the actuator ctrl vector.

gripper_enabled=False (self-feeding: the spoon is held throughout) keeps the
gripper closed and makes every gripper gesture a no-op.
"""
import time

import numpy as np

from stretch_demo.controllers.limits import JointWatch, limits_text
from stretch_demo.sim.base_drive import OmniDrive
from stretch_demo.tracking.imu_tracker import ImuTracker, NeutralCalibration
from stretch_demo.tracking.shaping import Setpoints

JOINTS = ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll")


class Gripper:
    """Binary gripper. The Stretch 4 closes at the LOW end of its ctrlrange."""

    def __init__(self, lo: float, hi: float, enabled: bool, closed: bool):
        self.lo, self.hi = lo, hi
        self.enabled = enabled
        self.closed = closed or not enabled

    @property
    def target(self) -> float:
        return self.lo if self.closed else self.hi

    def set(self, closed: bool) -> None:
        if self.enabled and closed != self.closed:
            self.closed = closed
            print(f"Gripper {'closed' if closed else 'open'}")

    def toggle(self) -> None:
        self.set(not self.closed)

    def describe(self) -> str:
        if not self.enabled:
            return "gripper holds the spoon"
        return "gripper CLOSED" if self.closed else "gripper open"


class ImuController:
    name = "controller"
    imu_labels: tuple[str, ...] = ("imu 0", "imu 1")

    def __init__(self, bus, gripper_enabled: bool = True):
        self.bus = bus
        self.gripper_enabled = gripper_enabled
        self.trackers = {i: ImuTracker(label) for i, label in enumerate(self.imu_labels)}
        self.calibration = NeutralCalibration(self.trackers)
        self.calibrating = True
        self._message = ""
        self.limits = ""  # which joints won't move further, for the overlay
        self.clock = time.monotonic  # replaceable in tests

    # --- harness interface ---

    def setup(self, model, data, act) -> None:
        self.act = act
        self.dt = model.opt.timestep
        self.lo = model.actuator_ctrlrange[:, 0].copy()
        self.hi = model.actuator_ctrlrange[:, 1].copy()
        limits = {j: (self.lo[act[j]], self.hi[act[j]]) for j in JOINTS}
        self.setpoints = Setpoints({j: float(data.ctrl[act[j]]) for j in JOINTS}, limits)
        g = act["gripper"]
        start_closed = abs(data.ctrl[g] - self.lo[g]) < abs(data.ctrl[g] - self.hi[g])
        self.gripper = Gripper(self.lo[g], self.hi[g], self.gripper_enabled, start_closed)
        self.base = OmniDrive(model, data, act)
        self.watch = JointWatch(act)
        self._template = data.ctrl.copy()  # head and anything else nobody drives
        self.on_setup()
        self.recalibrate()

    def recalibrate(self) -> None:
        self.calibration.restart()
        self.calibrating = True
        self.limits = ""
        print("Calibrating: the visitor holds a comfortable neutral pose, still, for ~2 s.")

    def get_ctrl(self, model, data):
        now = self.clock()
        batches = {i: self.bus.latest(i) for i in self.trackers}
        if self.calibrating:
            return self._calibrate(batches, now)
        frames = self._frames(batches)
        if frames is None:
            return self.compose(0.0)
        self._message = ""
        ctrl = self.compose(self.control(frames, self.dt, now))
        self.limits = limits_text(self.watch.check(self.setpoints, data, now))
        return ctrl

    def status(self) -> str:
        return self._message or self.describe()

    # --- for subclasses ---

    def on_setup(self) -> None:
        """Reset per-visitor state (filters, gesture detectors)."""

    def control(self, frames: dict, dt: float, now: float) -> float:
        """Update self.setpoints / self.gripper; return the base strafe level in
        [-1, 1] (+ = to the robot's left)."""
        raise NotImplementedError

    def describe(self) -> str:
        return ""

    def compose(self, base_level: float) -> np.ndarray:
        ctrl = self._template.copy()
        for joint in JOINTS:
            ctrl[self.act[joint]] = self.setpoints[joint]
        ctrl[self.act["gripper"]] = self.gripper.target
        ctrl = np.clip(ctrl, self.lo, self.hi)
        # Base AFTER the clip: the wheels report ctrlrange [0, 0]. Strafing is what
        # moves the forward-reaching arm sideways along a task.
        self.base.drive(ctrl, 0.0, strafe=base_level, dt=self.dt)
        return ctrl

    # --- internals ---

    def _calibrate(self, batches: dict, now: float) -> np.ndarray:
        if self.calibration.step(batches, now):
            self.calibrating = False
            self.watch.reset()
            self.on_setup()
            self._message = ""
            print(f"Calibrated. {self.name} is live.")
        else:
            self._message = self.calibration.progress(now)
        return self.compose(0.0)

    def _frames(self, batches: dict) -> dict | None:
        frames = {i: tracker.update(batches[i]) for i, tracker in self.trackers.items()}
        silent = [self.trackers[i].label for i, frame in frames.items() if frame is None]
        if silent:
            self._message = f"NO DATA from {', '.join(silent)} -- holding still"
            return None
        return frames


def drives_text(drives: dict[str, float]) -> str:
    """'lift +40% | arm -12%' for the drives that are moving."""
    return " | ".join(f"{name} {'+' if v > 0 else '-'}{abs(v):.0%}" for name, v in drives.items() if v)
