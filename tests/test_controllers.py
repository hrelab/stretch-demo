"""Each controller's mapping, driven with synthetic IMU frames on the real
Stretch 4 model, plus one end-to-end run through the fake IMU pipeline."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import frame, harness_for, live_controller
from stretch_demo.controllers.hands_v2 import HandsV2Controller
from stretch_demo.controllers.hands_vel import HandsVelController
from stretch_demo.controllers.head import HeadController
from stretch_demo.imu.bus import ImuBus

def make(cls, task="pickup", **kwargs):
    harness = harness_for(task)
    controller = cls(ImuBus(2), harness.task.gripper_enabled, **kwargs)
    return live_controller(controller, harness), harness


def drive(controller, frames, seconds=1.0, t0=0.0):
    """Run control() + compose() for `seconds` of 2 ms ticks; returns the last ctrl."""
    dt = controller.dt
    ctrl = None
    for k in range(round(seconds / dt)):
        ctrl = controller.compose(controller.control(frames, dt, t0 + k * dt))
    return ctrl


@pytest.mark.parametrize("cls", (HandsVelController, HandsV2Controller))
def test_hand_pitch_tips_the_gripper_the_way_it_did_on_the_stretch3(cls):
    """mujoco-sim's hand mappings were tuned on a Stretch 3, where + wrist pitch
    raises the gripper; on the Stretch 4 + lowers it, so the sign is flipped:
    the same hand tilt (+ IMU pitch, which lowered the S3 gripper) must still lower it."""
    ctl, _ = make(cls)
    start = ctl.setpoints["wrist_pitch"]
    drive(ctl, {0: frame(pitch=20.0), 1: frame()}, seconds=0.5)
    assert ctl.setpoints["wrist_pitch"] > start + 0.05  # + is down on the Stretch 4


def base_level(controller) -> float:
    """The base's slewed strafe level (+ = to the robot's left)."""
    return float(controller.base.state[1])


# --- hands_vel ---

def test_hands_vel_right_hand_tilt_moves_lift_at_a_rate():
    ctl, _ = make(HandsVelController)
    start = ctl.setpoints["lift"]
    drive(ctl, {0: frame(), 1: frame(pitch=-40.0)}, seconds=1.0)  # LIFT_SIGN -1: tip back raises
    assert ctl.setpoints["lift"] == pytest.approx(start + 0.15, abs=0.03)
    held = ctl.setpoints["lift"]
    drive(ctl, {0: frame(), 1: frame()}, seconds=1.0)
    assert ctl.setpoints["lift"] == pytest.approx(held, abs=0.01)  # neutral: stays put


def test_hands_vel_left_yaw_past_40_strafes_the_base():
    ctl, _ = make(HandsVelController)
    drive(ctl, {0: frame(yaw=30.0), 1: frame()}, seconds=0.5)
    assert base_level(ctl) == 0.0
    drive(ctl, {0: frame(yaw=60.0), 1: frame()}, seconds=1.0)
    assert base_level(ctl) > 0.4


def test_hands_vel_right_yaw_latches_gripper():
    ctl, _ = make(HandsVelController)
    assert not ctl.gripper.closed
    drive(ctl, {0: frame(), 1: frame(yaw=-30.0)}, seconds=0.1)
    drive(ctl, {0: frame(), 1: frame()}, seconds=0.1)
    assert ctl.gripper.closed
    drive(ctl, {0: frame(), 1: frame(yaw=30.0)}, seconds=0.1)
    assert not ctl.gripper.closed


def test_hands_vel_feeding_keeps_spoon_and_ignores_gripper_yaw():
    ctl, _ = make(HandsVelController, task="feeding")
    start = ctl.setpoints["lift"]
    ctrl = drive(ctl, {0: frame(), 1: frame(pitch=-40.0, yaw=-30.0)}, seconds=0.5)
    assert ctl.gripper.closed and not ctl.frozen
    assert ctl.setpoints["lift"] > start  # no lift freeze: there is no gripper gesture
    assert ctrl[ctl.act["gripper"]] == ctl.gripper.lo


def test_hands_vel_wrist_is_relative_to_the_task_start_pose():
    ctl, _ = make(HandsVelController, task="door")
    drive(ctl, {0: frame(roll=10.0), 1: frame()}, seconds=0.5)
    assert ctl.setpoints["wrist_roll"] == pytest.approx(-1.5708 + np.deg2rad(10.0), abs=0.02)


# --- hands_v2 ---

def flick_frames(rate_dps):
    return {0: frame(), 1: frame(gyro=(0.0, 0.0, rate_dps))}


def test_hands_v2_hand2_yaw_flick_toggles_gripper_and_pauses_lift():
    ctl, _ = make(HandsV2Controller)
    drive(ctl, flick_frames(400.0), seconds=0.08)
    assert ctl.gripper.closed
    start = ctl.setpoints["lift"]
    drive(ctl, {0: frame(), 1: frame(pitch=-30.0)}, seconds=0.2, t0=0.1)
    assert ctl.setpoints["lift"] == pytest.approx(start)  # paused just after the flick


def test_hands_v2_hand1_yaw_drives_wrist_then_base():
    ctl, _ = make(HandsV2Controller)
    start = ctl.setpoints["wrist_yaw"]
    drive(ctl, {0: frame(yaw=30.0), 1: frame()}, seconds=0.5)
    assert ctl.setpoints["wrist_yaw"] > start + 0.1 and base_level(ctl) == 0.0
    turned = ctl.setpoints["wrist_yaw"]
    drive(ctl, {0: frame(yaw=60.0), 1: frame()}, seconds=1.0)
    assert ctl.setpoints["wrist_yaw"] == pytest.approx(turned, abs=0.02)  # base takes over
    assert base_level(ctl) > 0.4


def test_hands_v2_flick_does_nothing_when_feeding():
    ctl, _ = make(HandsV2Controller, task="feeding")
    drive(ctl, flick_frames(400.0), seconds=0.08)
    drive(ctl, flick_frames(0.0), seconds=0.5)
    drive(ctl, flick_frames(400.0), seconds=0.08, t0=1.0)
    assert ctl.gripper.closed


# --- head ---

def test_head_nod_down_lowers_lift():
    ctl, _ = make(HeadController)
    start = ctl.setpoints["lift"]
    drive(ctl, {0: frame(pitch=-30.0), 1: frame()}, seconds=1.0)  # imu pitch - = flexion
    assert ctl.setpoints["lift"] < start - 0.1


def test_head_turn_drives_wrist_then_base_past_60():
    ctl, _ = make(HeadController)
    start = ctl.setpoints["wrist_yaw"]
    drive(ctl, {0: frame(yaw=40.0), 1: frame()}, seconds=0.5)
    assert ctl.setpoints["wrist_yaw"] > start + 0.1 and base_level(ctl) == 0.0
    turned = ctl.setpoints["wrist_yaw"]
    drive(ctl, {0: frame(yaw=72.0), 1: frame()}, seconds=1.0)
    assert ctl.setpoints["wrist_yaw"] == pytest.approx(turned, abs=0.02)
    assert base_level(ctl) > 0.4


def test_head_shoulder_protraction_drives_arm_with_hardware_sign():
    ctl, _ = make(HeadController)
    start = ctl.setpoints["arm"]
    drive(ctl, {0: frame(), 1: frame(yaw=-20.0)}, seconds=0.5)  # protraction = -yaw (right)
    assert ctl.setpoints["arm"] < start - 0.03  # arm_sign -1, from the Stretch 4 sessions


def shrug(peak_dps, seconds):
    return {0: frame(), 1: frame(gyro=(peak_dps, 0.0, 0.0))}, seconds


def test_head_double_shrug_toggles_gripper():
    ctl, _ = make(HeadController)
    t = 0.0
    for frames, seconds in (shrug(90, 0.08), shrug(0, 0.3), shrug(90, 0.08), shrug(0, 0.3)):
        drive(ctl, frames, seconds, t0=t)
        t += seconds
    assert ctl.gripper.closed


def test_head_keeps_wrist_roll_fixed():
    ctl, _ = make(HeadController, task="door")
    drive(ctl, {0: frame(roll=30.0, pitch=10.0, yaw=20.0), 1: frame(roll=10.0)}, seconds=0.5)
    assert ctl.setpoints["wrist_roll"] == pytest.approx(-1.5708, abs=1e-6)


# --- end to end through the fake IMU pipeline ---

def test_calibrates_through_bus_then_follows_imus(fake_imus, clock):
    bus, source = fake_imus
    harness = harness_for("pickup")
    ctl = HandsVelController(bus, True)
    ctl.clock = clock
    ctl.setup(harness.model, harness.data, harness.act)
    for _ in range(400):
        source.publish_once()
        clock.advance(0.01)
        ctl.get_ctrl(harness.model, harness.data)
    assert not ctl.calibrating
    source.set_pose(1, 0.0, -40.0, 0.0)
    start = ctl.setpoints["lift"]
    for _ in range(500):
        source.publish_once()
        clock.advance(0.002)
        ctl.get_ctrl(harness.model, harness.data)
    assert ctl.setpoints["lift"] > start + 0.1


def test_silent_imu_holds_and_says_which(fake_imus, clock):
    bus, _ = fake_imus
    harness = harness_for("pickup")
    ctl = live_controller(HandsVelController(bus, True), harness)
    ctl.clock = clock
    ctl.trackers[0].ref = ctl.trackers[1].ref = Rotation.identity()  # calibrated, but no data arrives
    ctl.get_ctrl(harness.model, harness.data)
    assert "NO DATA" in ctl.status() and "left hand" in ctl.status()
