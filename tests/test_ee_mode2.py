"""ee_mode2 (mujoco-sim's end-effector Mode 2): settings, inputs, directions in
the real physics, and every task done start to finish with hand tilts alone."""
import json

import mujoco
import mujoco.viewer
import numpy as np
import pytest

from conftest import frame, harness_for
from stretch_demo.cli import CONTROLLERS, TASKS, parse_args
from stretch_demo.controllers.ee.mode2 import (GRIPPER_COOLDOWN_S, STAGE_TIMEOUT_S, EeMode2Controller,
                                               KeyToggle, level_camera_axes, phase_label)
from stretch_demo.controllers.ee.settings import EeSettings, PostwristSettings
from stretch_demo.imu.bus import ImuBus

SPACE, PAGE_DOWN, LETTER_A = 32, 267, 65
NEUTRAL = {0: frame(), 1: frame()}


class Visitor:
    """A Mode 2 controller on a task's harness, stepped through the physics
    with synthetic hand tilts (no IMUs), scoring the task as the demo does."""

    def __init__(self, task: str, settings=None):
        self.h = harness_for(task)
        self.m, self.d, self.task = self.h.model, self.h.data, self.h.task
        self.task.bind(self.m, self.d)
        self.c = EeMode2Controller(ImuBus(2), self.task.gripper_enabled, settings)
        self.c.setup(self.m, self.d, self.h.act)
        self.c.calibrating = False
        self.c.on_setup()
        self.t = 0.0

    def hold(self, frames, seconds: float) -> None:
        for i in range(round(seconds / self.m.opt.timestep)):
            self.d.ctrl[:] = self.c.compose(*self.c.control(frames, self.c.dt, self.t))
            mujoco.mj_step(self.m, self.d)
            self.t += self.m.opt.timestep
            if i % 10 == 0:
                self.task.step(self.m, self.d, self.t)

    def steer(self, point, target, axes="xyz", tol=0.01, max_s=25.0) -> None:
        """Tilt toward `target` like a visitor watching the robot: decisively
        (at least 18 deg) until within `tol` on the chosen axes, then let go.
        The task cameras look along +x, so camera-right is -y."""
        start = self.t
        while self.t - start < max_s:
            error = np.where([axis in axes for axis in "xyz"], np.asarray(target) - point(), 0.0)
            if np.all(np.abs(error) < tol):
                break
            tilt = [_tilt(e, tol) for e in error]
            self.hold({0: frame(pitch=tilt[2]), 1: frame(pitch=tilt[0], roll=-tilt[1])}, 0.05)
        self.hold(NEUTRAL, 0.4)

    def press(self, keycode: int) -> None:
        self.c.keys._last = float("-inf")  # well apart from any earlier press
        self.c.on_key(keycode)
        self.hold(NEUTRAL, 1.5)

    def grasp(self) -> np.ndarray:
        return self.d.body("grasp_center_link").xpos.copy()


def _tilt(error: float, tol: float) -> float:
    if abs(error) < tol:
        return 0.0
    return float(np.sign(error) * min(35.0, 18.0 + 17.0 * abs(error) / 0.05))


def heading_deg(d) -> float:
    x = d.body("grasp_center_link").xmat.reshape(3, 3)[:, 0]
    return float(np.rad2deg(np.arctan2(x[1], x[0])))


# --- settings ---

def test_default_and_faster_profiles_load():
    assert PostwristSettings().motion.max_speed_m_s == (0.05, 0.05, 0.05)
    faster = PostwristSettings.load("profiles/imu_ee_mode2_faster.json")
    assert faster.motion.max_speed_m_s == (0.0625, 0.0625, 0.0625)
    assert faster.max_rotation_deg_s[1] == 56.25


@pytest.mark.parametrize("bad", [
    {"deadband_deg": [40, 8, 8]},          # deadband past full tilt
    {"translation_signs": [1, 0, 1]},
    {"max_speed_m_s": [0.2, 0.05, 0.05]},  # faster strafe than the base can
    {"reach_start_fraction": 1.0},
    {"camera_name": ""},
    {"no_such_field": 1},
])
def test_bad_profiles_are_refused(tmp_path, bad):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        PostwristSettings.load(path)


def test_settings_validation_matches_the_source():
    with pytest.raises(ValueError, match="three finite"):
        EeSettings(full_tilt_deg=(35.0, float("nan"), 35.0))
    assert EeSettings().rate_input(np.deg2rad([8.0, 35.0])).tolist() == pytest.approx([0.0, 1.0])


def test_cli_builds_ee_mode2_with_a_profile():
    args = parse_args(["--env", "pickup", "--controller", "ee_mode2", "--imu", "fake",
                       "--ee-profile", "profiles/imu_ee_mode2_faster.json"])
    controller = CONTROLLERS["ee_mode2"](ImuBus(2), TASKS["pickup"](), args)
    assert isinstance(controller, EeMode2Controller)
    assert controller.cfg.motion.max_speed_m_s[0] == 0.0625


# --- gripper keys ---

class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_key_toggle_counts_presses_not_auto_repeat():
    clock = Clock()
    keys = KeyToggle(clock)
    keys.press(SPACE)
    assert keys.take() == 1
    for _ in range(20):  # a held key: the OS repeats it every ~33 ms
        clock.t += 0.033
        keys.press(SPACE)
    assert keys.take() == 0
    clock.t += GRIPPER_COOLDOWN_S + 0.01
    keys.press(PAGE_DOWN)  # the clicker
    keys.press(LETTER_A)   # not a gripper key
    assert keys.take() == 1


def test_space_toggles_the_gripper_even_with_no_imu_data():
    v = Visitor("pickup")
    v.c.on_key(SPACE)
    v.c.on_hold()  # an IMU went silent
    v.c.compose(0.0)
    assert v.c.gripper.closed
    assert not v.c.base.state.any()


def test_feeding_keeps_the_spoon_whatever_is_pressed():
    v = Visitor("feeding")
    v.press(SPACE)
    assert v.c.gripper.closed


# --- directions, in the physics ---

@pytest.mark.parametrize("hands, moved", [
    ({0: frame(), 1: frame(pitch=35)}, ("x", +0.06)),    # right pitch: forward
    ({0: frame(), 1: frame(roll=35)}, ("y", -0.06)),     # right roll: to the right
    ({0: frame(pitch=35), 1: frame()}, ("z", +0.04)),    # left pitch: up (after the wrist pitches)
    ({0: frame(pitch=-35), 1: frame()}, ("z", -0.04)),   # ... and down
])
def test_hand_tilts_move_the_grasp_center(hands, moved):
    v = Visitor("pickup")
    start = v.grasp()
    v.hold(hands, 2.0)
    delta = v.grasp() - start
    axis = "xyz".index(moved[0])
    assert np.sign(delta[axis]) == np.sign(moved[1]) and abs(delta[axis]) > abs(moved[1])
    assert np.abs(np.delete(delta, axis)).max() < 0.015  # and only that way


def test_left_roll_turns_the_gripper_about_its_grasp_center():
    v = Visitor("pickup")
    start, heading = v.grasp(), heading_deg(v.d)
    v.hold({0: frame(roll=35), 1: frame()}, 1.5)
    assert heading_deg(v.d) - heading > 10.0               # turned left (seen from above)
    assert np.linalg.norm(v.grasp() - start) < 0.02        # without swinging the grasp center


def test_up_pitches_the_wrist_first_and_neutral_levels_it():
    v = Visitor("pickup")
    v.hold({0: frame(pitch=35), 1: frame()}, 0.1)
    assert v.c.stage.phase == "PITCH UP" and not v.c.stage.vertical
    v.hold({0: frame(pitch=35), 1: frame()}, 0.6)
    assert v.c.stage.phase == "LIFT UP"
    assert np.rad2deg(v.c.setpoints["wrist_pitch"]) == pytest.approx(-10.0, abs=0.1)
    v.hold(NEUTRAL, 1.5)
    assert v.c.stage.phase == "LEVEL" and v.c.setpoints["wrist_pitch"] == pytest.approx(0.0, abs=1e-3)


def test_up_still_works_from_the_bottom_of_the_lift():
    """In the source, the pitch stage never finished at the lift's lower limit
    (the IK would need the lift lower still), so up never started."""
    v = Visitor("pickup")
    v.hold({0: frame(pitch=-35), 1: frame()}, 9.0)
    assert v.c.setpoints["lift"] == pytest.approx(0.0, abs=1e-3)
    bottom = v.grasp()[2]
    v.hold({0: frame(pitch=35), 1: frame()}, STAGE_TIMEOUT_S + 2.0)
    assert v.grasp()[2] > bottom + 0.04


def test_door_sideways_stays_level_despite_the_rolled_wrist():
    """The door task pre-rolls the wrist 90 deg, which turns the gripper camera's
    own "right" to point up; commands use its levelled heading instead."""
    v = Visitor("door")
    axes = level_camera_axes(v.d, v.c.camera_id)
    assert axes[:, 1] == pytest.approx([0.0, 0.0, -1.0])
    start = v.grasp()
    v.hold({0: frame(), 1: frame(roll=35)}, 2.0)
    delta = v.grasp() - start
    assert delta[1] < -0.06 and abs(delta[2]) < 0.01


def test_phase_labels():
    assert phase_label(+1, True, 0.0) == "LIFT UP"
    assert phase_label(-1, False, 0.1) == "PITCH DOWN"
    assert phase_label(0, False, 0.1) == "LEVELING"
    assert phase_label(0, False, 0.0) == "LEVEL"


# --- every task, start to finish, with hand tilts and the gripper key ---

def test_mode2_does_the_floor_pickup():
    v = Visitor("pickup")
    bottle = v.d.body("bottle").xpos.copy()
    target = np.array([bottle[0] - 0.022, bottle[1], 0.19])  # centred on it, fingers around it
    v.steer(v.grasp, np.r_[v.grasp()[0], target[1], v.grasp()[2]], axes="y")
    v.steer(v.grasp, np.r_[v.grasp()[:2], target[2]], axes="z")  # down first...
    v.steer(v.grasp, np.r_[target[0], v.grasp()[1:]], axes="x")  # ...then reach around it
    v.press(SPACE)
    v.steer(v.grasp, np.r_[v.grasp()[:2], 0.76], axes="z")       # clear the side table
    assert v.task.state == "carry"
    zone = v.d.site("place_zone").xpos
    v.steer(v.grasp, np.r_[zone[:2], v.grasp()[2]], axes="xy", tol=0.015, max_s=40.0)
    v.steer(v.grasp, np.r_[v.grasp()[:2], zone[2] + 0.12], axes="z")
    v.press(PAGE_DOWN)  # the clicker opens it
    (trial,) = v.task.trials
    assert trial["result"] == "success" and trial["drops"] == 0


def test_mode2_opens_the_door():
    v = Visitor("door")
    handle = v.d.site("rs_handle_site").xpos.copy()
    v.steer(v.grasp, np.r_[v.grasp()[0], handle[1:]], axes="yz")
    v.steer(v.grasp, handle, axes="x")
    v.press(SPACE)
    v.hold({0: frame(), 1: frame(pitch=-30, roll=20)}, 10.0)  # pull back while moving right
    (trial,) = v.task.trials
    assert trial["result"] == "success"


def test_mode2_feeds_a_bite():
    v = Visitor("feeding")
    tip = lambda: v.d.site("utensil_tip").xpos.copy()  # noqa: E731
    plate, mouth = v.d.site("plate_zone").xpos.copy(), v.d.site("mouth_box").xpos.copy()
    v.steer(tip, np.r_[plate[:2], tip()[2]], axes="xy")
    v.steer(tip, plate - [0, 0, 0.005], axes="z")
    assert v.task.state == "carry"
    v.steer(tip, np.r_[tip()[:2], mouth[2]], axes="z")
    v.steer(tip, mouth, tol=0.008)
    v.hold(NEUTRAL, 1.5)
    (trial,) = v.task.trials
    assert trial["result"] == "success" and trial["points"] == 100 and v.task.face_hits == 0


# --- harness wiring ---

class _ClosedViewer:
    """Stands in for the MuJoCo viewer: records how it was opened, then 'closes'."""
    opened_with: dict = {}

    def __init__(self, model, data, **kwargs):
        _ClosedViewer.opened_with = kwargs
        self.cam = type("Cam", (), {"lookat": np.zeros(3)})()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def is_running(self):
        return False


def test_viewer_keys_reach_the_controller(monkeypatch):
    v = Visitor("pickup")
    monkeypatch.setattr(mujoco.viewer, "launch_passive", _ClosedViewer)
    v.h.run(v.c, console=None, bus=ImuBus(2), log=None)
    assert _ClosedViewer.opened_with["key_callback"] == v.c.on_key


def test_terminal_g_toggles_the_gripper(capsys):
    v = Visitor("pickup")
    v.h._handle("g", v.c, log=None)
    assert v.c.gripper.closed
    feeding = Visitor("feeding")
    feeding.h._handle("g", feeding.c, log=None)
    assert "holds the spoon" in capsys.readouterr().out
