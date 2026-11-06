import numpy as np

from conftest import Scripted, harness_for
from stretch_demo.controllers.limits import JointWatch, limits_text
from stretch_demo.tracking.shaping import Setpoints

JOINTS = ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll")


def setpoints_for(harness, ctrl):
    ranges = harness.model.actuator_ctrlrange
    return Setpoints({j: float(ctrl[harness.act[j]]) for j in JOINTS},
                     {j: tuple(ranges[harness.act[j]]) for j in JOINTS})


def test_setpoints_remember_which_limit_a_write_pushed_past():
    sp = Setpoints({"lift": 0.5}, {"lift": (0.0, 1.0)})
    sp.step("lift", 0.7)
    assert sp["lift"] == 1.0 and sp.pushing["lift"] == 1
    sp.step("lift", 0.0)  # resting at the limit, not asking for more
    assert sp.pushing["lift"] == 0
    sp.set("lift", -0.2)
    assert sp.pushing["lift"] == -1


def test_at_rest_nothing_is_reported_stuck():
    harness = harness_for("pickup")  # the lift sags ~3 cm under its command: not "blocked"
    run = Scripted(harness)
    run.go(1.0)
    watch = JointWatch(harness.act)
    sp = setpoints_for(harness, run.ctrl)
    assert watch.check(sp, harness.data, 0.0) == watch.check(sp, harness.data, 5.0) == []


def test_wrist_pitched_into_the_table_is_reported_blocked():
    harness = harness_for("feeding")
    run = Scripted(harness)
    watch = JointWatch(harness.act)
    sp = setpoints_for(harness, run.ctrl)
    notes, t = [], 0.0
    while not notes and t < 4.0:  # pitch down at 60 deg/s, as hands_v2 at full tilt
        sp.step("wrist_pitch", np.deg2rad(60.0) * 0.01)
        run.go(0.01, wrist_pitch=sp["wrist_pitch"])
        t += 0.01
        notes = watch.check(sp, harness.data, t)
    assert notes == ["wrist pitch blocked"]
    assert sp["wrist_pitch"] < np.deg2rad(120.0)  # far short of its +245 deg limit


def test_limits_text_tells_the_visitor_what_to_do():
    assert limits_text([]) == ""
    assert limits_text(["wrist roll at its limit"]) == "WRIST ROLL AT ITS LIMIT -- move it back the other way"
