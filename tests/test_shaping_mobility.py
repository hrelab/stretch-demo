from pathlib import Path

import numpy as np
import pytest

from stretch_demo.tracking.mobility import MobilitySettings
from stretch_demo.tracking.shaping import Ema, RollUnwrap, Setpoints, YawDriftBaseline, shape

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


def test_shape_deadband_ramp_and_sign():
    assert shape(0.05, 0.1, 0.5) == 0.0
    assert shape(0.3, 0.1, 0.5) == pytest.approx(0.5)
    assert shape(-0.9, 0.1, 0.5) == -1.0
    assert 0 < shape(0.3, 0.1, 0.5, expo=0.6) < 0.5  # expo: gentler mid-range


def test_roll_unwrap_is_continuous_across_180():
    unwrap = RollUnwrap()
    out = [unwrap(np.deg2rad(a)) for a in (170, 179, -179, -170)]
    assert np.rad2deg(out[-1]) == pytest.approx(190.0)


def test_yaw_drift_baseline_soaks_up_slow_drift_only():
    base = YawDriftBaseline(tau_s=10.0)
    quick = base.correct(0.5, 0.01)
    assert quick == pytest.approx(0.5, abs=0.01)
    for _ in range(10000):
        held = base.correct(0.5, 0.01)
    assert abs(held) < 0.01


def test_setpoints_clamp_to_limits_and_ema_smooths():
    sp = Setpoints({"lift": 0.5}, {"lift": (0.0, 1.0)})
    sp.step("lift", 2.0)
    assert sp["lift"] == 1.0
    ema = Ema(0.5)
    assert ema("x", 1.0) == 1.0 and ema("x", 0.0) == 0.5


# Head-controller profiles only: profiles/imu_ee_* belong to ee_mode2.
@pytest.mark.parametrize("profile", sorted(PROFILES.glob("p[0-9]*.json")), ids=lambda p: p.name)
def test_mujoco_sim_profiles_load_unchanged(profile):
    settings = MobilitySettings.load(profile)
    assert settings.head_yaw.pos_deg > 0
    assert settings.base_roll_sign == 1.0  # demo field, absent from the file: default


def test_defaults_carry_the_stretch4_hardware_signs():
    s = MobilitySettings()
    assert s.wrist_pitch_sign == -1.0 and s.arm_sign == -1.0 and s.gestures.hold_direction == "down"
    assert s.head_lift_sign == -1.0  # head->lift: from its own hardware try, not the shoulder's


def test_update_is_all_or_nothing():
    settings = MobilitySettings()
    with pytest.raises(ValueError):
        settings.update({"deadband_frac": 0.9, "head_pitch": {"pos_deg": 20.0}})
    assert settings.head_pitch.pos_deg == 50.0
    with pytest.raises(ValueError, match="unknown setting"):
        settings.update({"nope": 1})


def test_left_shoulder_mirrors_roll_and_yaw():
    settings = MobilitySettings()
    settings.update({"shoulder_side": "left"})
    assert settings.shoulder_motion("shoulder_elevation").sign == -MobilitySettings().shoulder_elevation.sign
