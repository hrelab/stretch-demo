"""head: mujoco-sim's bio_m2_head without modes, for users with little hand
movement (high cervical SCI). imu 0 on the head, imu 1 on top of one shoulder
(--shoulder, default right).

  head pitch (nod)           -> lift velocity: nod down = lower, back = raise
  head roll (ear to shoulder)-> base strafes that way: tilt left = robot moves left
  head yaw (turn)            -> wrist yaw velocity
  shoulder forward / back    -> arm extend / retract velocity
  double shrug               -> gripper open / close
  (wrist pitch and roll stay where the task put them)

Why this layout (2026-11-04, replacing head yaw past 60 deg -> base and head roll
-> wrist pitch): strafing by turning the head 60 deg took the visitor's eyes off
the screen exactly while the robot moved, and ear-to-shoulder -> wrist pitch
didn't feel natural. A head tilt strafes with the eyes still on the screen, in
the direction of the tilt (every task camera looks along the arm, so the robot's
left is the screen's left). No task needs wrist pitch: each is done with the
wrist level. The shrug stays a binary input (the gyro double shrug that worked
on hardware): from a relaxed seated position it only works one way, so it can't
drive anything up AND down.

Every range, speed and sign is in MobilitySettings (--mobility FILE; mujoco-sim's
--calibrate-rom profiles load as-is). head_lift_sign was flipped after the first
hardware try (nod down raised the lift); base_roll_sign is untested on hardware.
"""
from stretch_demo.controllers.base import ImuController, drives_text
from stretch_demo.tracking.gestures import DoubleShrug
from stretch_demo.tracking.mobility import AXES, MobilitySettings
from stretch_demo.tracking.shaping import RollUnwrap, YawDriftBaseline, shape

HEAD, SHOULDER = 0, 1


class HeadController(ImuController):
    name = "head"
    imu_labels = ("head", "shoulder")

    def __init__(self, bus, gripper_enabled: bool = True, settings: MobilitySettings | None = None):
        super().__init__(bus, gripper_enabled)
        self.settings = settings or MobilitySettings()
        self.settings.validate()

    def on_setup(self) -> None:
        self.shrugs = DoubleShrug()
        self._misses_seen = 0
        self.unwrap = {HEAD: RollUnwrap(), SHOULDER: RollUnwrap()}
        self.drift = {HEAD: YawDriftBaseline(), SHOULDER: YawDriftBaseline()}
        self.drives: dict[str, float] = {}

    def control(self, frames: dict, dt: float, now: float) -> float:
        s = self.settings
        angles = self._angles(frames, dt)
        self._gestures(frames[SHOULDER].gyro, now)
        levels = {name: self._level(name, angle) for name, angle in angles.items()}
        drive = {
            "lift": -levels["head_pitch"] * s.head_lift_sign,
            "wrist_yaw": levels["head_yaw"],
            "arm": levels["shoulder_protraction"] * s.arm_sign,
        }
        for joint, level in drive.items():
            self.setpoints.step(joint, level * s.max_rate[joint] * dt)
        base = levels["head_roll"] * s.base_roll_sign  # + = tilt left = robot's left
        self.drives = {**drive, "base": base}
        return base

    def _angles(self, frames: dict, dt: float) -> dict[str, float]:
        """Each motion's signed angle (+ = its pos_name direction), from both IMUs
        after roll unwrapping and yaw drift removal."""
        imus = {}
        for index, frame in frames.items():
            roll, pitch, yaw = frame.angles
            imus[index] = (self.unwrap[index](roll), pitch, self.drift[index].correct(yaw, dt))
        return {name: motion.sign * imus[SHOULDER if name.startswith("shoulder") else HEAD][AXES.index(motion.axis)]
                for name, motion in self.settings.motions().items()}

    def _level(self, name: str, angle: float) -> float:
        s = self.settings
        rom = s.motions()[name].rom(angle)
        return shape(angle, s.deadband_frac * rom, s.full_frac * rom, s.expo)

    def _gestures(self, gyro, now: float) -> None:
        """A double shrug toggles the gripper; a near miss prints why."""
        elevation = self.settings.shoulder_motion("shoulder_elevation")
        axis = AXES.index(elevation.axis)
        rate = elevation.sign * gyro[axis]
        others = max(abs(g) for i, g in enumerate(gyro) if i != axis)
        if self.shrugs.update(rate, others, self.settings.gestures, now) == "double":
            self.gripper.toggle()
        if self.shrugs.misses != self._misses_seen:
            self._misses_seen = self.shrugs.misses
            print(f"Shrug not counted -- {self.shrugs.miss}")

    def describe(self) -> str:
        moving = drives_text(self.drives) or "holding"
        hint = "" if not self.gripper.enabled else " (double shrug toggles)"
        return f"{moving} | {self.gripper.describe()}{hint}"
