"""Per-user head/shoulder settings for the `head` controller.

The same MobilitySettings as mujoco-sim's retarget/head_shoulder.py, so its
profiles (profiles/p0*.json, made by `sim --calibrate-rom --save-profile`) load
here unchanged with --mobility. Fields that only meant something with modes
(the PAUSE-mode long hold, lift onset, lockout) are kept so those files still
load, but the demo ignores them.

Defaults are for the Stretch 4: the signs mujoco-sim's bio_m2_head needed on
Stretch 4 hardware (2026-10-21/28; its ROBOT_DEFAULTS["stretch4"]) are built in.
Trust them over sim reasoning.

Added for the demo's no-mode mapping (all editable in a profile JSON):
  head_lift_sign       head pitch drives the LIFT. -1 = nod down lowers, set after
                       the first hardware try (2026-11-04: +1 raised it). Separate
                       from lift_sign, which was measured for the shoulder driving it.
  base_roll_sign       head roll (ear to shoulder) strafes the base; +1 = tilting
                       left moves the robot left. Untested on hardware: flip it if
                       the robot goes the other way.
"""
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

AXES = ("roll", "pitch", "yaw")  # index order of the angle tuples
WRIST_SIGNS = (+1.0, -1.0, +1.0)  # (yaw, pitch, roll) for hand/head IMUs, from mujoco-sim imu_hands


@dataclass
class Motion:
    """One body motion: which IMU axis measures it and the user's range each way
    from neutral. `sign` makes + the pos_name direction."""

    axis: str
    sign: float
    neg_deg: float
    pos_deg: float
    neg_name: str = "-"
    pos_name: str = "+"

    def rom(self, value: float) -> float:
        """ROM in radians on the side `value` (already signed) falls on."""
        return float(np.deg2rad(self.pos_deg if value >= 0 else self.neg_deg))


@dataclass
class GestureSettings:
    """Double shrug: two upward swings over flick_dps (held flick_hold_s, at least
    flick_dominance x the other axes' rate) whose starts are within double_window_s.
    rearm_dps re-arms after a swing. The remaining fields are kept for profile
    compatibility with mujoco-sim and unused here."""

    flick_dps: float = 60.0
    flick_hold_s: float = 0.03
    flick_dominance: float = 1.5
    rearm_dps: float = 25.0
    double_window_s: float = 1.0
    on_frac: float = 0.3
    off_frac: float = 0.15
    hold_s: float = 1.5
    hold_direction: str = "down"  # Stretch 4 (hardware); unused here
    lift_onset_s: float = 0.0
    min_lockout_s: float = 0.3


def _head_pitch() -> Motion:
    return Motion("pitch", WRIST_SIGNS[1], 60.0, 50.0, "extension", "flexion")


def _head_roll() -> Motion:
    return Motion("roll", WRIST_SIGNS[2], 45.0, 45.0, "lateral flexion (left)", "lateral flexion (right)")


def _head_yaw() -> Motion:
    return Motion("yaw", WRIST_SIGNS[0], 80.0, 80.0, "rotation (right)", "rotation (left)")


def _shoulder_elevation() -> Motion:
    # Right acromion; signs from the first real-IMU session (2026-10-21).
    return Motion("roll", +1.0, 10.0, 30.0, "depression", "elevation")


def _shoulder_protraction() -> Motion:
    return Motion("yaw", -1.0, 20.0, 25.0, "retraction", "protraction")


def _max_rate() -> dict:
    return {"lift": 0.15, "arm": 0.20, "wrist_pitch": float(np.deg2rad(60.0)),
            "wrist_roll": float(np.deg2rad(60.0)), "wrist_yaw": float(np.deg2rad(60.0))}


MOTIONS = ("head_pitch", "head_roll", "head_yaw", "shoulder_elevation", "shoulder_protraction")


@dataclass
class MobilitySettings:
    head_pitch: Motion = field(default_factory=_head_pitch)
    head_roll: Motion = field(default_factory=_head_roll)
    head_yaw: Motion = field(default_factory=_head_yaw)
    shoulder_elevation: Motion = field(default_factory=_shoulder_elevation)
    shoulder_protraction: Motion = field(default_factory=_shoulder_protraction)
    shoulder_side: str = "right"

    deadband_frac: float = 0.15  # of each direction's ROM: nothing moves inside it
    full_frac: float = 0.6       # of each direction's ROM: full speed from here on
    expo: float = 0.6
    max_rate: dict = field(default_factory=_max_rate)
    lift_sign: float = -1.0          # mujoco-sim: shoulder -> lift (unused here)
    arm_sign: float = -1.0           # shoulder protraction -> arm (hardware)
    wrist_pitch_sign: float = -1.0   # mujoco-sim: head roll -> wrist pitch (unused here)
    base_forward_sign: float = -1.0  # mujoco-sim BASE mode (unused here)
    base_turn_sign: float = +1.0     # mujoco-sim BASE mode (unused here)
    base_turn_fraction: float = 0.5  # unused here
    gripper_closed_is_hi: bool = False
    gestures: GestureSettings = field(default_factory=GestureSettings)

    head_lift_sign: float = -1.0
    base_roll_sign: float = +1.0

    def shoulder_motion(self, name: str) -> Motion:
        """Shoulder motions mirrored for the left side (roll and yaw flip)."""
        motion = getattr(self, name)
        if self.shoulder_side == "left" and motion.axis in ("roll", "yaw"):
            return Motion(**{**asdict(motion), "sign": -motion.sign})
        return motion

    def motions(self) -> dict[str, Motion]:
        return {name: self.shoulder_motion(name) if name.startswith("shoulder") else getattr(self, name)
                for name in MOTIONS}

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, changes: dict) -> None:
        """Apply a partial nested dict; all-or-nothing, ValueError says why."""
        candidate = MobilitySettings.from_dict(_merge(self.to_dict(), changes))
        candidate.validate()
        for f in fields(self):
            setattr(self, f.name, getattr(candidate, f.name))

    def validate(self) -> None:
        _validate_motions(self)
        _validate_fractions(self)
        _validate_gestures(self.gestures)

    @classmethod
    def from_dict(cls, data: dict) -> "MobilitySettings":
        data = dict(data)
        for name in MOTIONS:
            if name in data:
                data[name] = Motion(**data[name])
        if "gestures" in data:
            data["gestures"] = GestureSettings(**data["gestures"])
        return cls(**data)

    @classmethod
    def load(cls, path) -> "MobilitySettings":
        """Defaults, overridden by whatever subset the JSON file sets."""
        settings = cls()
        settings.update(json.loads(Path(path).read_text()))
        return settings


def _validate_motions(s: MobilitySettings) -> None:
    for name in MOTIONS:
        motion = getattr(s, name)
        if motion.axis not in AXES:
            raise ValueError(f"{name}.axis must be one of {AXES}, not {motion.axis!r}")
        if not (motion.neg_deg > 0 and motion.pos_deg > 0):
            raise ValueError(f"{name}: ROM must be > 0 deg both ways")
    if s.shoulder_side not in ("left", "right"):
        raise ValueError(f"shoulder_side must be 'left' or 'right', not {s.shoulder_side!r}")


def _validate_fractions(s: MobilitySettings) -> None:
    if not 0 <= s.deadband_frac < s.full_frac <= 1:
        raise ValueError("need 0 <= deadband_frac < full_frac <= 1")


def _validate_gestures(g: GestureSettings) -> None:
    if not 0 < g.rearm_dps < g.flick_dps:
        raise ValueError("need 0 < gestures.rearm_dps < gestures.flick_dps")
    if g.flick_dominance < 1 or g.flick_hold_s < 0 or g.double_window_s <= 0:
        raise ValueError("gestures: need flick_dominance >= 1 and positive times")


def _merge(base: dict, changes: dict) -> dict:
    out = dict(base)
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f"unknown setting {key!r}")
        nested = isinstance(value, dict) and isinstance(base[key], dict)
        out[key] = _merge(base[key], value) if nested else value
    return out
