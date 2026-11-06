"""hands_vel: mujoco-sim's imu_hands_vel, the hand-mounted controller that worked
best hands-on. IMUs on the backs of both hands (imu 0 = left, imu 1 = right).

  left hand  roll/pitch/yaw  -> wrist roll/pitch/yaw, 1:1 position from the
                                task's start pose (5 deg deadband)
  left hand  yaw past 40 deg -> base strafes that way (full speed at 70 deg)
  right hand pitch           -> lift VELOCITY (tilt to move, neutral to stop)
  right hand forearm twist   -> arm extend VELOCITY
  right hand yaw past +-25   -> gripper open / close (latched, hysteresis),
                                only while the right hand is otherwise near level

Constants and signs are imu_hands / imu_hands_vel's (see their docstrings in
mujoco-sim for how each was arrived at), with these demo changes:
- With the gripper off (self-feeding), right-hand yaw does nothing.
- The wrist pitch sign is flipped: mujoco-sim tuned it on a Stretch 3, whose
  wrist pitch is physically reversed relative to the Stretch 4's.
- No lift/arm freeze, and the gripper only latches with the right hand near
  level. Tilting the hand (lift) and twisting it (arm) at the same time reads as
  yaw: 25 deg of tilt with 45 deg of twist reads ~18 deg of yaw, 30 + 50 ~24 deg.
  mujoco-sim froze lift/arm past 15 deg of yaw, so driving both at once kept
  pausing them (found hands-on, 2026-11-04), and could even toggle the gripper.
  With tilt and twist both under GRIPPER_LEVEL, the fake yaw stays under ~7 deg,
  far from the 25 deg gripper threshold.
"""
import numpy as np

from stretch_demo.controllers.base import ImuController, drives_text
from stretch_demo.tracking.shaping import Ema, YawDriftBaseline, deadband, shape

LEFT, RIGHT = 0, 1

WRIST_SIGNS = (+1.0, +1.0, +1.0)  # (yaw, pitch, roll); pitch -1 on the Stretch 3
LIFT_SIGN = -1.0
EXTEND_SIGN = +1.0
WRIST_DEADBAND = np.deg2rad(5.0)
EMA_ALPHA = 0.15

RATE_DEADBAND = np.deg2rad(8.0)
LIFT_FULL_TILT = np.deg2rad(40.0)
EXTEND_FULL_TILT = np.deg2rad(35.0)
LIFT_MAX_RATE = 0.15   # m/s
ARM_MAX_RATE = 0.20    # m/s
EXPO = 0.6

BASE_YAW_START = np.deg2rad(40.0)
BASE_YAW_FULL = np.deg2rad(70.0)

GRIPPER_CLOSE_YAW = np.deg2rad(-25.0)
GRIPPER_OPEN_YAW = np.deg2rad(25.0)
GRIPPER_LEVEL = np.deg2rad(20.0)  # gripper yaw only counts with |pitch|, |roll| under this


def latch_gripper(closed: bool, yaw: float) -> bool:
    """Close past GRIPPER_CLOSE_YAW, open past GRIPPER_OPEN_YAW, else hold."""
    if np.sign(GRIPPER_CLOSE_YAW) * yaw > abs(GRIPPER_CLOSE_YAW):
        return True
    if np.sign(GRIPPER_OPEN_YAW) * yaw > abs(GRIPPER_OPEN_YAW):
        return False
    return closed


class HandsVelController(ImuController):
    name = "hands_vel"
    imu_labels = ("left hand", "right hand")

    def setup(self, model, data, act) -> None:
        super().setup(model, data, act)
        # The 1:1 wrist mapping is relative to the task's start pose, so a neutral
        # hand keeps e.g. the door task's pre-rolled wrist instead of undoing it.
        self.wrist_start = {name: self.setpoints[name] for name in ("wrist_yaw", "wrist_pitch", "wrist_roll")}

    def on_setup(self) -> None:
        self.ema = Ema(EMA_ALPHA)
        self.base_yaw = YawDriftBaseline()
        self.grip_yaw = YawDriftBaseline()
        self.drives: dict[str, float] = {}

    def control(self, frames: dict, dt: float, now: float) -> float:
        roll, pitch, yaw = frames[LEFT].angles
        self._wrist(roll, pitch, yaw)
        base = shape(WRIST_SIGNS[0] * self.base_yaw.correct(yaw, dt), BASE_YAW_START, BASE_YAW_FULL)

        g_roll, g_pitch, g_yaw = frames[RIGHT].angles
        grip_yaw = self.grip_yaw.correct(g_yaw, dt)
        lift, extend = self._arm(g_pitch, g_roll, dt)
        if self.gripper.enabled and max(abs(g_pitch), abs(g_roll)) < GRIPPER_LEVEL:
            self.gripper.set(latch_gripper(self.gripper.closed, grip_yaw))
        self.drives = {"lift": lift, "arm": extend, "base": base}
        return base

    def _wrist(self, roll: float, pitch: float, yaw: float) -> None:
        for name, angle, sign in (("wrist_yaw", yaw, WRIST_SIGNS[0]),
                                  ("wrist_pitch", pitch, WRIST_SIGNS[1]),
                                  ("wrist_roll", roll, WRIST_SIGNS[2])):
            target = self.wrist_start[name] + sign * deadband(angle, WRIST_DEADBAND)
            self.setpoints.set(name, self.ema(name, target))

    def _arm(self, pitch: float, roll: float, dt: float) -> tuple[float, float]:
        lift = shape(LIFT_SIGN * pitch, RATE_DEADBAND, LIFT_FULL_TILT, EXPO)
        extend = shape(EXTEND_SIGN * roll, RATE_DEADBAND, EXTEND_FULL_TILT, EXPO)
        # Smooth the RATE, not the position: takes the edge off tremor without
        # delaying a stop.
        self.setpoints.step("lift", self.ema("lift_rate", lift * LIFT_MAX_RATE) * dt)
        self.setpoints.step("arm", self.ema("arm_rate", extend * ARM_MAX_RATE) * dt)
        return lift, extend

    def describe(self) -> str:
        parts = [drives_text(self.drives)] if any(self.drives.values()) else ["holding"]
        parts.append(self.gripper.describe())
        return " | ".join(parts)
