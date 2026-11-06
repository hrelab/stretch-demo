"""hands_v2: mujoco-sim's bio_m2_v2 without modes. IMUs on the backs of both
hands; every axis is tilt-to-move (rate control) with the matched axes and tuned
speeds of bio_m2_v2.

  hand 1 pitch / roll        -> wrist pitch / roll velocity
  hand 1 yaw                 -> wrist yaw velocity ...
  hand 1 yaw past 40 deg     -> ... and instead the base strafes that way (full at 70)
                                (yaw only counts with hand 1's pitch and roll under
                                YAW_LEVEL: tilting and rolling together reads as
                                yaw, ~18 deg for 25 + 45, and at a rate that keeps
                                spinning the wrist; seen hands-on 2026-11-04)
  hand 2 pitch (tip up/down) -> lift velocity
  hand 2 forearm twist       -> arm extend velocity
  hand 2 quick yaw flick     -> gripper open / close

bio_m2_v2 needed an ARM/BASE mode because hand 1's tilts drove the base in BASE
mode. Here the base comes from hand 1 yaw past a threshold (as in hands_vel),
which frees the hand 2 yaw flick that used to switch modes for the gripper. The
flick keeps bio_m2_v2's strict settings (an accidental toggle is worse than a
missed one), and lift/arm pause while hand 2 is flicking so the flick's wobble
doesn't move them.
"""
import numpy as np

from stretch_demo.controllers.base import ImuController, drives_text
from stretch_demo.tracking.gestures import FlickDetector, FlickSettings
from stretch_demo.tracking.shaping import RollUnwrap, YawDriftBaseline, shape

HAND1, HAND2 = 0, 1
YAW = 2

# Wrist pitch is +1 here, -1 in bio_m2_v2: it was tuned on a Stretch 3, whose
# wrist pitch is physically reversed relative to the Stretch 4's.
WRIST_YAW_SIGN, WRIST_PITCH_SIGN, WRIST_ROLL_SIGN = (+1.0, +1.0, +1.0)
LIFT_SIGN = -1.0
EXTEND_SIGN = +1.0

MAX_RATE = {
    "lift": 0.15, "arm": 0.20,  # m/s
    "wrist_pitch": np.deg2rad(60.0), "wrist_roll": np.deg2rad(60.0), "wrist_yaw": np.deg2rad(60.0),
}
RATE_DEADBAND = np.deg2rad(8.0)
FULL_TILT = np.deg2rad(35.0)
EXPO = 0.6

YAW_LEVEL = np.deg2rad(20.0)
BASE_YAW_START = np.deg2rad(40.0)
BASE_YAW_FULL = np.deg2rad(70.0)

GRIP_FLICK = FlickSettings(fire=np.deg2rad(300.0), hold_s=0.04, dominance=2.0)
FLICK_PAUSE_S = 0.4  # lift/arm stay paused this long after a flick


class HandsV2Controller(ImuController):
    name = "hands_v2"
    imu_labels = ("hand 1 (wrist)", "hand 2 (lift/arm)")

    def on_setup(self) -> None:
        self.flick = FlickDetector(YAW, +1.0, GRIP_FLICK)
        self.roll1, self.roll2 = RollUnwrap(), RollUnwrap()
        self.yaw1 = YawDriftBaseline()
        self.paused_until = float("-inf")
        self.drives: dict[str, float] = {}

    def control(self, frames: dict, dt: float, now: float) -> float:
        roll1, pitch1, yaw1 = frames[HAND1].angles
        roll2, pitch2, _ = frames[HAND2].angles
        roll1, roll2 = self.roll1(roll1), self.roll2(roll2)
        yaw1 = self.yaw1.correct(yaw1, dt)
        if max(abs(pitch1), abs(roll1)) >= YAW_LEVEL:
            yaw1 = 0.0
        paused = self._gripper_flick(frames[HAND2].gyro, now)

        base = shape(WRIST_YAW_SIGN * yaw1, BASE_YAW_START, BASE_YAW_FULL)
        drive = {
            "wrist_pitch": shape(WRIST_PITCH_SIGN * pitch1, RATE_DEADBAND, FULL_TILT, EXPO),
            "wrist_roll": shape(WRIST_ROLL_SIGN * roll1, RATE_DEADBAND, FULL_TILT, EXPO),
            "wrist_yaw": 0.0 if base else shape(WRIST_YAW_SIGN * yaw1, RATE_DEADBAND, FULL_TILT, EXPO),
            "lift": 0.0 if paused else shape(LIFT_SIGN * pitch2, RATE_DEADBAND, FULL_TILT, EXPO),
            "arm": 0.0 if paused else shape(EXTEND_SIGN * roll2, RATE_DEADBAND, FULL_TILT, EXPO),
        }
        for name, level in drive.items():
            self.setpoints.step(name, level * MAX_RATE[name] * dt)
        self.drives = {**drive, "base": base}
        return base

    def _gripper_flick(self, gyro, now: float) -> bool:
        """Toggle the gripper on a hand-2 yaw flick; True while lift/arm should
        pause (mid-flick or just after one)."""
        if not self.gripper.enabled:
            return False
        if self.flick.update(gyro, now):
            self.gripper.toggle()
            self.paused_until = now + FLICK_PAUSE_S
        return now < self.paused_until or abs(gyro[YAW]) > GRIP_FLICK.rearm

    def describe(self) -> str:
        moving = drives_text(self.drives) or "holding"
        return f"{moving} | {self.gripper.describe()}"
