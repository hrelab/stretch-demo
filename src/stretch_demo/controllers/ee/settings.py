"""End-effector controller settings, from mujoco-sim's end_effector package
(imu_ee_prewrist_controller.EeSettings / RateInput, imu_ee_postwrist.
PostwristSettings) at ee_controller commit 85f57de.

Profiles are the same flat JSON files (`--ee-profile`, e.g.
profiles/imu_ee_mode2_faster.json); all angles are degrees. Changes from the
source: EeSettings.__post_init__ (cognitive complexity 20) is split into one
validator per rule, and fields only the pre-wrist controller read are kept so
mujoco-sim profiles still load.
"""
import json
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path

import numpy as np

from stretch_demo.sim.base_drive import OMNI_FULL_SPEED

EXPO = 0.6  # bio_m2_v2's: finer control just past the deadband

TRIPLES = ("full_tilt_deg", "deadband_deg", "translation_signs", "max_speed_m_s", "orientation_scale")
SCALARS = ("wrist_speed_deg_s", "reach_base_speed_m_s", "reach_start_fraction")


class RateInput:
    """Shape a sensor's (roll, pitch[, yaw]) into signed drive levels in [-1, 1]."""

    def __init__(self, full_tilt_deg, deadband_deg, signs):
        self.deadband = np.deg2rad(deadband_deg)
        self.inverse_span = 1 / (np.deg2rad(full_tilt_deg) - self.deadband)
        self.signs = np.asarray(signs)

    def __call__(self, angles):
        count = len(angles)
        signed = np.asarray(angles) * self.signs[:count]
        u = np.clip((np.abs(signed) - self.deadband[:count]) * self.inverse_span[:count], 0, 1)
        return np.sign(signed) * ((1 - EXPO) * u + EXPO * u**3)


def _triple(name: str, value) -> tuple[float, float, float]:
    values = np.asarray(value, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain three finite numbers")
    return tuple(float(v) for v in values)


def _check_tilts(deadband, full_tilt) -> None:
    deadband, full_tilt = np.asarray(deadband), np.asarray(full_tilt)
    if np.any(deadband < 0) or np.any(full_tilt <= deadband) or np.any(full_tilt >= 90):
        raise ValueError("require 0 <= deadband_deg < full_tilt_deg < 90 per axis")


def _check_signs(signs) -> None:
    if not np.all(np.isin(signs, (-1, 1))):
        raise ValueError("translation_signs must each be +1 or -1")


def _check_speeds(s: "EeSettings") -> None:
    if np.any(np.asarray(s.max_speed_m_s) <= 0):
        raise ValueError("max_speed_m_s must be positive")
    if s.max_speed_m_s[0] > OMNI_FULL_SPEED[1]:
        raise ValueError(f"strafe speed cannot exceed {OMNI_FULL_SPEED[1]} m/s")
    if not np.isfinite(s.wrist_speed_deg_s) or s.wrist_speed_deg_s <= 0:
        raise ValueError("wrist_speed_deg_s must be finite and positive")


def _check_reach(s: "EeSettings") -> None:
    if not np.isfinite(s.reach_base_speed_m_s) or not 0 <= s.reach_base_speed_m_s <= OMNI_FULL_SPEED[0]:
        raise ValueError(f"reach_base_speed_m_s must be between 0 and {OMNI_FULL_SPEED[0]}")
    if not np.isfinite(s.reach_start_fraction) or not 0.5 <= s.reach_start_fraction < 1:
        raise ValueError("reach_start_fraction must be between 0.5 (inclusive) and 1 (exclusive)")


@dataclass(frozen=True)
class EeSettings:
    """Input arrays are (roll, pitch, yaw); speeds are command (X, Y, Z)."""

    full_tilt_deg: tuple[float, float, float] = (35.0, 35.0, 35.0)
    deadband_deg: tuple[float, float, float] = (8.0, 8.0, 8.0)
    translation_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_speed_m_s: tuple[float, float, float] = (0.088, 0.20, 0.15)
    orientation_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    wrist_speed_deg_s: float = 60.0
    reach_base_speed_m_s: float = 0.03
    reach_start_fraction: float = 0.9

    @cached_property
    def rate_input(self) -> RateInput:
        return RateInput(self.full_tilt_deg, self.deadband_deg, self.translation_signs)

    def __post_init__(self):
        for name in TRIPLES:
            object.__setattr__(self, name, _triple(name, getattr(self, name)))
        for name in SCALARS:
            object.__setattr__(self, name, float(getattr(self, name)))
        _check_tilts(self.deadband_deg, self.full_tilt_deg)
        _check_signs(self.translation_signs)
        _check_speeds(self)
        _check_reach(self)


POSTWRIST_SPEED = (0.05, 0.05, 0.05)  # m/s: post-wrist's default, not pre-wrist's


@dataclass(frozen=True)
class PostwristSettings:
    motion: EeSettings = field(default_factory=lambda: EeSettings(max_speed_m_s=POSTWRIST_SPEED))
    rotation_full_tilt_deg: tuple = (35.0, 35.0, 35.0)
    rotation_deadband_deg: tuple = (8.0, 8.0, 8.0)
    rotation_signs: tuple = (1.0, 1.0, 1.0)
    max_rotation_deg_s: tuple = (45.0, 45.0, 45.0)
    camera_name: str = "gripper_camera_left_rgb"

    @cached_property
    def rotation_input(self) -> RateInput:
        return RateInput(self.rotation_full_tilt_deg, self.rotation_deadband_deg, self.rotation_signs)

    def __post_init__(self):
        for name in ("rotation_full_tilt_deg", "rotation_deadband_deg", "rotation_signs", "max_rotation_deg_s"):
            object.__setattr__(self, name, _triple(name, getattr(self, name)))
        _check_tilts(self.rotation_deadband_deg, self.rotation_full_tilt_deg)
        _check_signs(self.rotation_signs)
        if np.any(np.asarray(self.max_rotation_deg_s) <= 0):
            raise ValueError("max_rotation_deg_s must contain three finite positive rates")
        if not isinstance(self.camera_name, str) or not self.camera_name:
            raise ValueError("camera_name must be a nonempty MuJoCo camera name")

    @classmethod
    def load(cls, path) -> "PostwristSettings":
        """A flat JSON profile: EeSettings fields and PostwristSettings fields
        side by side, all optional."""
        values = json.loads(Path(path).read_text()) if path else {}
        if not isinstance(values, dict):
            raise ValueError("EE profile must be a JSON object")
        motion_keys = {f.name for f in fields(EeSettings)}
        own_keys = {f.name for f in fields(cls)} - {"motion"}
        unknown = values.keys() - motion_keys - own_keys
        if unknown:
            raise ValueError(f"unknown post-wrist profile fields: {sorted(unknown)}")
        motion = {"max_speed_m_s": POSTWRIST_SPEED} | {k: v for k, v in values.items() if k in motion_keys}
        return cls(motion=EeSettings(**motion), **{k: v for k, v in values.items() if k in own_keys})
