"""Door opening: grab the handle and pull the door open past GOAL_RAD.

  REACH  until the robot first touches the handle
  OPEN   until the hinge passes GOAL_RAD -> trial done
  DONE   the door stays open (the gripper may still be on the handle, so it
         isn't snapped shut automatically): `r` closes it for the next try

Pulling it open takes retracting the arm while strafing the base to the right
(the handle swings toward the robot on an arc), i.e. two drives at once.

Per trial: time to first touch the handle, time to open, widest angle.
"""
import numpy as np

from stretch_demo.sim.robot import GRASP_BODY
from stretch_demo.tasks.base import SCENES, Task, body_id, geom_id, touching

GOAL_RAD = 0.6  # of the hinge's 0.8 rad range

INSTRUCTIONS = {
    "reach": "grab the door handle",
    "open": "pull the door open",
    "done": "door open!  (operator: r to close it)",
}


class DoorTask(Task):
    name = "door"
    title = "OPEN THE DOOR"
    scene = SCENES / "door.xml"
    # Wrist pre-rolled 90 deg: the handle is a horizontal lever, and the fingers
    # close horizontally at roll 0, so they'd pinch it end to end and slip off
    # (tried: the door didn't move). Rolled, they close around it.
    # -90 rather than +90: the Stretch 4's roll range stops at +65 deg. The head
    # controller never changes wrist roll, so this is what it grips with.
    start_pose = {"lift": 0.61, "arm": 0.05, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": -1.5708}
    camera = ((0.602, -0.066, 0.75), 2.0, -55.0, -20.0)

    def __init__(self):
        super().__init__()
        self._reset_state()

    def _reset_state(self) -> None:
        self.state = "reach"
        self.touch_s: float | None = None
        self.peak = 0.0

    def bind(self, model, data) -> None:
        joint = model.joint("rs_door_hinge").id
        self.qadr, self.vadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        self.handle = {geom_id(model, "rs_handle"), geom_id(model, "rs_handle_base")}
        robot_root = model.body_rootid[body_id(model, GRASP_BODY)]
        self.is_robot = lambda g: model.body_rootid[model.geom_bodyid[g]] == robot_root
        self._model = model

    def reset_objects(self, model, data) -> None:
        super().reset_objects(model, data)
        data.qpos[self.qadr] = 0.0
        data.qvel[self.vadr] = 0.0
        self._reset_state()

    def angle(self, data) -> float:
        return float(data.qpos[self.qadr])

    def step(self, model, data, now: float) -> None:
        if self.state == "done":
            return
        elapsed = self.clock(now)
        angle = self.angle(data)
        self.peak = max(self.peak, angle)
        if self.touch_s is None and touching(model, data, self.handle, self.is_robot):
            self.touch_s = elapsed
            self.state = "open"
        if angle >= GOAL_RAD:
            self._done(elapsed)

    def _done(self, elapsed: float) -> None:
        touch = self.touch_s if self.touch_s is not None else elapsed
        self.finish_trial({"result": "success", "touch_s": round(touch, 2),
                           "open_s": round(elapsed, 2), "peak_deg": round(float(np.rad2deg(self.peak)), 1)})
        self.state = "done"

    def overlay(self, now: float) -> list[tuple[str, str]]:
        live = self.started_at is not None and self.state != "done"
        timer = f"   ({now - self.started_at:.1f} s)" if live else ""
        rows = [("task", self.title), ("now", INSTRUCTIONS[self.state] + timer),
                ("door", f"{np.rad2deg(self.peak):.0f} deg of {np.rad2deg(GOAL_RAD):.0f} needed")]
        if self.trials:
            last = self.trials[-1]
            rows.append(("opened", f"{len(self.trials)} time(s)"))
            rows.append(("last", f"handle at {last['touch_s']:.1f} s, open at {last['open_s']:.1f} s"))
        return rows
