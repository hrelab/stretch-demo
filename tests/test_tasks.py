"""Scenes, scoring state machines, and that every task is physically doable
(scripted motions on the Stretch 4) and gets scored."""
import csv

import mujoco
import numpy as np
import pytest

from conftest import Scripted, harness_for
from stretch_demo.sim.robot import GRASP_BODY
from stretch_demo.tasks.feeding import DWELL_S
from stretch_demo.tasks.log import TrialLog

TASKS = ("feeding", "pickup", "door")


def robot_scene_contacts(harness) -> list:
    """Robot-vs-scene contacts (the floor aside)."""
    m, d = harness.model, harness.data
    root = m.body_rootid[m.body(GRASP_BODY).id]
    floor = m.geom("floor").id
    out = []
    for c in d.contact[: d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        mixed = (m.body_rootid[m.geom_bodyid[g1]] == root) != (m.body_rootid[m.geom_bodyid[g2]] == root)
        if mixed and floor not in (g1, g2):
            out.append((m.geom(g1).name, m.geom(g2).name))
    return out


@pytest.mark.parametrize("task", TASKS)
def test_scene_loads_with_robot_clear_of_everything(task):
    harness = harness_for(task)
    assert robot_scene_contacts(harness) == []


def test_spoon_points_out_of_the_gripper():
    d = harness_for("feeding").data
    out = d.site("utensil_tip").xpos - d.body(GRASP_BODY).xpos
    assert out == pytest.approx([0.15, 0.0, 0.0], abs=0.01)  # forward, level


@pytest.mark.parametrize("task, site", [("feeding", "mouth_box"), ("pickup", "bottle_center"),
                                        ("door", "rs_handle_site")])
def test_targets_are_in_front_of_the_arm(task, site):
    target = harness_for(task).data.site(site).xpos
    assert target[0] > 0.6 and abs(target[1]) < 0.12


# --- feeding scoring, with the spoon tip placed directly ---

def place_tip(harness, point) -> None:
    mujoco.mj_forward(harness.model, harness.data)
    harness.data.site_xpos[harness.task.tip] = point


def feed_steps(harness, points, t0=0.0, dt=0.1):
    t = t0
    for point in points:
        place_tip(harness, point)
        harness.task.step(harness.model, harness.data, t)
        t += dt
    return t


def feeding():
    harness = harness_for("feeding")
    harness.task.bind(harness.model, harness.data)
    d = harness.data
    return harness, d.site("plate_zone").xpos.copy(), d.site("mouth_box").xpos.copy()


def test_feeding_scores_a_bite_with_accuracy_and_efficiency():
    harness, plate, mouth = feeding()
    off_center = mouth + [0.01, 0.0, 0.0]
    halfway = (plate + mouth) / 2
    feed_steps(harness, [plate, halfway, off_center, off_center] + [mouth] * int(DWELL_S / 0.1 + 2))
    (trial,) = harness.task.trials
    assert trial["result"] == "success"
    assert trial["error_cm"] == pytest.approx(0.0, abs=0.05)  # closest approach during the dwell
    assert trial["time_s"] == pytest.approx(0.2, abs=0.01)
    assert trial["path_efficiency"] > 0.95
    assert harness.task.state == "scoop"


def test_feeding_touching_the_face_is_a_miss():
    harness, plate, mouth = feeding()
    forehead = mouth + [0.05, 0.0, 0.08]  # the face looks back at the robot (-x)
    feed_steps(harness, [plate, forehead])
    assert harness.task.trials[0]["result"].startswith("missed")


def test_feeding_needs_a_scoop_first_and_redip_restarts():
    harness, plate, mouth = feeding()
    feed_steps(harness, [mouth, mouth])
    assert harness.task.state == "scoop" and not harness.task.trials
    t = feed_steps(harness, [plate, (plate + mouth) / 2])
    feed_steps(harness, [plate], t0=t + 5.0)
    assert harness.task._carry["t0"] == pytest.approx(t + 5.0)


# --- physically doable ---

def lift_to_tip_height(run, z, tries=3):
    """Servo the lift until the spoon tip is at height z (the S4 lift sags)."""
    for _ in range(tries):
        error = z - run.d.site("utensil_tip").xpos[2]
        run.ramp("lift", run.ctrl[run.h.act["lift"]] + error)
        run.go(1.0)


def test_feeding_bite_is_reachable():
    run = Scripted(harness_for("feeding"))
    run.go(0.5)
    run.ramp("arm", 0.075)
    lift_to_tip_height(run, run.d.site("plate_zone").xpos[2] - 0.005)
    assert run.h.task.state == "carry"
    lift_to_tip_height(run, run.d.site("mouth_box").xpos[2])
    run.ramp("wrist_yaw", 0.19, rate=0.5)
    run.ramp("arm", 0.353, rate=0.1)
    run.go(DWELL_S + 0.5)
    assert [t["result"] for t in run.h.task.trials] == ["success"]


def test_floor_pickup_is_doable_and_scored():
    harness = harness_for("pickup")
    run = Scripted(harness)
    run.go(2.0, lift=0.0, arm=0.02)
    run.ramp("arm", 0.20, rate=0.1)
    run.go(2.0, gripper=run.gripper(closed=True))
    run.ramp("lift", 0.56)
    assert harness.task.state == "carry"
    run.go(5.6, base=1.0)
    run.go(1.5)
    run.ramp("lift", 0.47)
    run.go(1.0)
    run.go(1.0, gripper=run.gripper(closed=False))
    run.ramp("arm", 0.05, rate=0.1)  # back away, as a visitor would: open fingers keep it jiggling
    run.go(1.0)
    (trial,) = harness.task.trials
    assert trial["result"] == "success" and trial["drops"] == 0


def test_pickup_counts_a_drop():
    harness = harness_for("pickup")
    run = Scripted(harness)
    task = harness.task
    task.reset_objects(run.m, run.d)
    run.d.qpos[task.qadr + 2] += 0.3  # bottle teleported up = "lifted"
    run.go(0.05)
    assert task.state == "carry"
    run.go(1.5)                      # let go: falls back to the floor
    assert task.drops == 1 and task.state == "reach"


def test_door_is_doable_and_scored():
    harness = harness_for("door")
    run = Scripted(harness)
    handle_z = run.d.site("rs_handle_site").xpos[2]
    run.go(2.0, lift=run.ctrl[harness.act["lift"]] + handle_z - run.d.body(GRASP_BODY).xpos[2])
    run.go(3.0, arm=0.24)
    run.go(2.0, gripper=run.gripper(closed=True))
    arm = 0.24
    for _ in range(80):  # retract while backing along the handle's arc
        arm = max(0.0, arm - 0.002)
        run.go(0.1, arm=arm, base=-0.6)
    (trial,) = harness.task.trials
    assert trial["result"] == "success"
    assert harness.task.touch_s is not None
    harness.task.reset_objects(run.m, run.d)
    assert harness.task.angle(run.d) == 0.0 and harness.task.state == "reach"


# --- logging ---

def test_trial_log_writes_one_row_per_trial(tmp_path):
    log = TrialLog("feeding", "head", "P01", runs_dir=tmp_path)
    log.write([{"trial": 1, "result": "success", "time_s": 3.2}])
    log.next_visitor()
    log.write([{"trial": 1, "result": "missed", "time_s": 4.0}])
    rows = list(csv.DictReader(log.path.open()))
    assert [r["visitor"] for r in rows] == ["1", "2"] and rows[0]["who"] == "P01"
    assert "1/1 successful" in log.summary() and "0/1 successful" in log.summary()
