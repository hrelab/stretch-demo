"""head: mujoco-sim's bio_m2_head without modes, for users with little hand
movement (high cervical SCI). imu 0 on the head, imu 1 on top of one shoulder
(--shoulder, default right).

  head pitch (nod)           -> lift velocity: nod down = lower, back = raise
  head yaw (turn)            -> wrist yaw velocity ...
  head yaw past ~60 deg      -> ... and instead the base strafes that way
  head roll (ear to shoulder)-> wrist pitch velocity
  shoulder forward / back    -> arm extend / retract velocity
  double shrug               -> gripper open / close
  (wrist roll stays put)

Why this layout (chosen 2026-11-02): the shrug only works one way from a
relaxed seated position, so it can't drive the lift up AND down; as a binary
input (the gyro double shrug that worked on hardware) it's reliable. That
leaves the head to drive the lift, the wrist and -- past a big turn -- the base.

Every range, speed and sign is in MobilitySettings (--mobility FILE; mujoco-sim's
--calibrate-rom profiles load as-is). Untested on hardware: the head-pitch->lift
sign (head_lift_sign), head-yaw->base (base_yaw_sign).
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
        base = self._base_level(angles["head_yaw"])
        drive = {
            "lift": -levels["head_pitch"] * s.head_lift_sign,  # flexion (nod down) lowers
            "wrist_pitch": levels["head_roll"] * s.wrist_pitch_sign,
            "wrist_yaw": 0.0 if base else levels["head_yaw"],
            "arm": levels["shoulder_protraction"] * s.arm_sign,
        }
        for joint, level in drive.items():
            self.setpoints.step(joint, level * s.max_rate[joint] * dt)
        self.drives = {**drive, "base": base}
        return base * s.base_yaw_sign

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

    def _base_level(self, yaw: float) -> float:
        s = self.settings
        rom = s.head_yaw.rom(yaw)
        return shape(yaw, s.base_yaw_start_frac * rom, s.base_yaw_full_frac * rom)

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
