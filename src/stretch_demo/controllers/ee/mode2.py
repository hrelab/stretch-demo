"""ee_mode2: mujoco-sim's end-effector Mode 2 controller (end_effector/
imu_ee_mode2_controller.py, ee_controller commit 85f57de by jawy1248), on the
demo's IMU pipeline. IMUs on the backs of both hands (imu 0 = left, imu 1 = right).
Every motion moves the GRASP CENTER (inverse kinematics picks the joints and
base); nothing is a single joint.

  left pitch                 -> up / down: the wrist first pitches 10 deg that
                                way, then the grasp center travels vertically
  left roll                  -> turn the gripper about the vertical (Ry)
  right pitch                -> forward / back
  right roll                 -> sideways
  both yaws                  -> unused
  space or presenter clicker -> gripper open / close (viewer window focused;
                                terminal `g` does the same)

Neutral left pitch stops vertical travel and levels the wrist again. All
translation is rate control; the base strafes and, with the arm near either end
of its travel, assists forward/back (spring zones, see ik.base_speed_bounds).

Changes from the source, for the demo:
- Input: the demo's ImuBus/ImuTracker (gyro-referenced heading, settled
  calibration, NO DATA hold) replaces Redis and the pre-wrist base class.
  Mode 2 never used BASE mode, the lockout or the pre-wrist orientation solver,
  so they aren't ported.
- Gripper keys: from the MuJoCo viewer window (key_callback) instead of a
  global pynput hook, so no X11/accessibility permissions are needed. The
  viewer only reports presses, so a held key's auto-repeat is absorbed by
  restarting the cooldown on every press.
- Command directions: the camera's HEADING with world up kept up (see
  level_camera_axes). The source used the camera's own axes, which is the same
  thing with a level wrist (feeding, pickup) but not with the door task's wrist
  pre-rolled 90 deg, where the camera's "right" points up.
- Start pose: the task's, not mujoco-sim's scene-aware spawn (startup_pose.py):
  each task's pose and viewer camera are tuned for its scoring.
- Telemetry/feedback-window hooks are dropped; status goes to the overlay.
- Pitch stage timeout: vertical travel starts once the wrist has pitched, or
  STAGE_TIMEOUT_S after asking, whichever comes first. In the source it waited
  for the pitch only, and at the lift's lower limit the pitch can't happen (the
  IK keeps the grasp center still, which would need the lift lower still), so
  "up" never started after reaching the floor: stuck in PITCH UP (found
  2026-11-06 driving the pickup task; the same holds going down from the top).
- Complexity: _task_twist (13) is split into PitchStage + level_camera_axes.
"""
import time
from threading import Lock

import numpy as np

from stretch_demo.controllers.base import ImuController
from stretch_demo.controllers.ee.ik import JOINT_NAMES, PostwristIK
from stretch_demo.controllers.ee.settings import PostwristSettings
from stretch_demo.sim.base_drive import OMNI_FULL_SPEED
from stretch_demo.tracking.shaping import RollUnwrap

LEFT, RIGHT = 0, 1

PITCH_BEFORE_LIFT_DEG = 10.0
PITCH_STAGE_SPEED_DEG_S = 30.0
PITCH_ARRIVAL_TOLERANCE_DEG = 2.0  # servo settling under gravity
STAGE_TIMEOUT_S = 0.8  # the pitch normally takes ~0.4 s; never wait longer than this
GRIPPER_COOLDOWN_S = 0.15
# GLFW keycodes: space, and what a presenter clicker sends (arrows, page up/down).
GRIPPER_KEYS = {32: "space", 262: "right", 263: "left", 266: "page up", 267: "page down"}
# Optical camera axes from MuJoCo's (right, up, backward): right, down, forward.
OPTICAL_AXES = np.diag([1.0, -1.0, -1.0])
DOWN = np.array([0.0, 0.0, -1.0])


class KeyToggle:
    """Debounced, thread-safe key presses -> toggle requests.

    A press only counts GRIPPER_COOLDOWN_S after the previous gripper-key press.
    Every press restarts the window, so a held key's auto-repeat never toggles
    again, and a clicker's double-send counts once.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._lock = Lock()
        self._last = float("-inf")
        self._pending = 0

    def press(self, keycode: int) -> None:
        if keycode not in GRIPPER_KEYS:
            return
        now = self.clock()
        with self._lock:
            if now - self._last >= GRIPPER_COOLDOWN_S:
                self._pending += 1
            self._last = now

    def take(self) -> int:
        with self._lock:
            count, self._pending = self._pending, 0
        return count


def level_camera_axes(data, camera_id: int) -> np.ndarray:
    """Columns (right, down, forward): the camera's heading, kept level.

    Up/down stays world up/down whatever the wrist's roll or the pitch stage's
    tilt; forward is the camera's view direction projected on the floor.
    """
    optical = data.cam_xmat[camera_id].reshape(3, 3) @ OPTICAL_AXES
    forward = optical[:, 2].copy()
    forward[2] = 0.0
    norm = np.linalg.norm(forward)
    if norm < 1e-6:  # looking straight up/down: no heading to keep
        return optical
    forward /= norm
    return np.column_stack([np.cross(DOWN, forward), DOWN, forward])


def mode2_rates(cfg: PostwristSettings, left, right):
    """(linear, angular) in optical camera terms (X right, Y down, Z forward)
    from each hand's (roll, pitch) in radians.

    IMU1 (left) uses the translation input settings; IMU2 (right) the rotation
    input settings, even for translation. Speed limits follow the output axis.
    + rotation about camera Y turns the view toward camera-right.
    """
    motion = cfg.motion
    left = motion.rate_input(left)
    right = cfg.rotation_input(right)
    linear = np.array([right[0], -left[1], right[1]]) * np.asarray(motion.max_speed_m_s)
    turn = -left[0] * np.deg2rad(cfg.max_rotation_deg_s[1]) * motion.orientation_scale[1]
    return linear, np.array([0.0, turn, 0.0])


def phase_label(direction: float, vertical: bool, error: float) -> str:
    if vertical:
        return "LIFT UP" if direction > 0 else "LIFT DOWN"
    if direction:
        return "PITCH UP" if direction > 0 else "PITCH DOWN"
    return "LEVELING" if abs(error) > 1e-6 else "LEVEL"


class PitchStage:
    """Pitch before vertical travel.

    With up/down requested, the wrist pitches toward 10 deg that way (relative to
    its start, at up to 30 deg/s); vertical travel waits until the command is
    there and the measured wrist is within 2 deg, or STAGE_TIMEOUT_S at most.
    At neutral it levels again. The IK's wrist pitch bounds are narrowed to
    start +-10 deg for the session.
    """

    def __init__(self, model, act, ik: PostwristIK, level: float):
        idx = act["wrist_pitch"]
        self.joint = model.actuator_trnid[idx, 0]
        self.qadr = model.jnt_qposadr[self.joint]
        self.gear = model.actuator_gear[idx, 0]
        self.level = level
        self.span = np.deg2rad(PITCH_BEFORE_LIFT_DEG) * self.gear
        column = JOINT_NAMES.index("wrist_pitch")
        limits = ik.limits[column]
        limits[0] = max(limits[0], level - abs(self.span))
        limits[1] = min(limits[1], level + abs(self.span))
        self.bounds = limits.copy()
        self.phase = "LEVEL"
        self.vertical = False
        self._direction = 0.0
        self._waited = 0.0

    def step(self, direction: float, command: float, data, dt: float) -> float:
        """direction: +1 up, -1 down, 0 neither; command: the wrist pitch setpoint.
        Returns the pitch rate (rad/s) to request about the pitch joint's axis."""
        goal = np.clip(self.level - direction * self.span, *self.bounds)  # wrist pitch - = up
        error = (goal - command) / self.gear
        limit = np.deg2rad(PITCH_STAGE_SPEED_DEG_S)
        rate = float(np.clip(error / dt, -limit, limit))
        settled = abs(goal / self.gear - data.qpos[self.qadr]) <= np.deg2rad(PITCH_ARRIVAL_TOLERANCE_DEG)
        self._waited = self._waited + dt if direction == self._direction else 0.0
        self._direction = direction
        arrived = abs(error) < 1e-6 and settled
        self.vertical = bool(direction and (arrived or self._waited >= STAGE_TIMEOUT_S))
        self.phase = phase_label(direction, self.vertical, error)
        return rate


class EeMode2Controller(ImuController):
    name = "ee_mode2"
    imu_labels = ("left hand", "right hand")

    def __init__(self, bus, gripper_enabled: bool = True, settings: PostwristSettings | None = None):
        super().__init__(bus, gripper_enabled)
        self.cfg = settings or PostwristSettings()
        self.keys = KeyToggle()

    def setup(self, model, data, act) -> None:
        self.data = data
        super().setup(model, data, act)
        self.ik = PostwristIK(model, act, self.cfg)
        self.stage = PitchStage(model, act, self.ik, self.setpoints["wrist_pitch"])
        self.camera_id = model.camera(self.cfg.camera_name).id  # fails loudly if missing

    def on_setup(self) -> None:
        self.unwrap = {LEFT: RollUnwrap(), RIGHT: RollUnwrap()}

    # --- harness hooks ---

    def on_key(self, keycode: int) -> None:
        self.keys.press(keycode)

    def compose(self, base_level: float, forward: float = 0.0) -> np.ndarray:
        # Every tick ends here (live, calibrating or holding for missing IMU
        # data), so gripper requests always go through.
        for _ in range(self.keys.take()):
            self.gripper.toggle()
        return super().compose(base_level, forward)

    def on_hold(self) -> None:
        # A lost stream stops the base at once, residual slew included (as in the source).
        self.base.state[:] = 0.0
        self.stage.phase = "HOLD"

    # --- control ---

    def control(self, frames: dict, dt: float, now: float) -> tuple[float, float]:
        left = self._roll_pitch(LEFT, frames)
        right = self._roll_pitch(RIGHT, frames)
        linear, angular = mode2_rates(self.cfg, left, right)
        return self._follow(self._task_twist(linear, angular, dt), dt)

    def _roll_pitch(self, imu: int, frames: dict) -> np.ndarray:
        roll, pitch, _ = frames[imu].angles
        return np.array([self.unwrap[imu](roll), pitch])

    def _task_twist(self, linear, angular, dt: float) -> np.ndarray:
        """World (linear, angular) twist of the grasp center. Right-hand
        translation is camera-relative (levelled); left-hand up/down is world
        vertical once the pitch stage is done."""
        basis = level_camera_axes(self.data, self.camera_id)
        up = -linear[1]
        pitch_rate = self.stage.step(np.sign(up), self.setpoints["wrist_pitch"], self.data, dt)
        world_linear = basis[:, 0] * linear[0] + basis[:, 2] * linear[2]
        if self.stage.vertical:
            world_linear[2] += up
        world_angular = basis @ angular + self.data.xaxis[self.stage.joint] * pitch_rate
        return np.r_[world_linear, world_angular]

    def _follow(self, task, dt: float) -> tuple[float, float]:
        """IK rates -> joint setpoints; returns the base (strafe, forward) levels."""
        if not np.any(task) and not np.any(self.base.state[:2]):
            self.ik.achieved = np.zeros(6)  # zero rates solve a zero twist exactly
            return 0.0, 0.0
        setpoint = self._ctrl_vector()
        self.ik.update(self.data, setpoint)
        rates = self.ik.solve(task, setpoint, self.base.state, dt)
        for name, rate in zip(JOINT_NAMES, rates[2:]):
            self.setpoints.step(name, rate * dt)
        forward, strafe = rates[:2] / OMNI_FULL_SPEED[:2]
        return float(strafe), float(forward)

    def _ctrl_vector(self) -> np.ndarray:
        """The IK works on a full ctrl vector in actuator units."""
        vector = self._template.copy()
        for joint in JOINT_NAMES:
            vector[self.act[joint]] = self.setpoints[joint]
        return vector

    def describe(self) -> str:
        hint = " (space/clicker toggles)" if self.gripper.enabled else ""
        return f"{self.stage.phase} | {self.gripper.describe()}{hint}"
