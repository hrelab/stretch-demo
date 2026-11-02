"""Floor pickup: a bottle has been dropped on the floor in front of the robot.
Pick it up, strafe over to the side table and set it down in the green zone.

  REACH   bottle on the floor
  CARRY   bottle lifted clear of the floor (grasped)
  PLACED  bottle at rest inside the place zone -> trial done, and the bottle is
          put back on the floor after RESET_AFTER_S for the next try
  Setting it down (or dropping it) back on the floor counts a drop and returns
  to REACH. A bottle knocked over on the floor needs `r`.

Per trial: time to grasp, time to place, drops.
"""
import numpy as np

from stretch_demo.tasks.base import SCENES, Task, body_id, geom_id, site_contains, site_id, touching

LIFTED_DZ = 0.05     # m above its resting height, off the floor = grasped
DROPPED_DZ = 0.03
AT_REST_SPEED = 0.05  # m/s
SETTLE_S = 0.5
RESET_AFTER_S = 2.5
TIPPED_COS = np.cos(np.deg2rad(50.0))

INSTRUCTIONS = {
    "reach": "pick the bottle up off the floor (reach well around it, then close)",
    "carry": "put it on the side table (green zone)",
    "placed": "placed! next bottle coming...",
}


class PickupTask(Task):
    name = "pickup"
    title = "FLOOR PICKUP"
    scene = SCENES / "pickup.xml"
    start_pose = {"lift": 0.36, "arm": 0.05, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0}
    camera = ((0.602, 0.134, 0.35), 2.1, 50.0, -22.0)

    def __init__(self):
        super().__init__()
        self._reset_state()

    def _reset_state(self) -> None:
        self.state = "reach"
        self.grasp_s: float | None = None
        self.drops = 0
        self._still_since: float | None = None
        self._placed_at: float | None = None
        self.knocked_over = False

    def bind(self, model, data) -> None:
        self.body = body_id(model, "bottle")
        joint = model.body_jntadr[self.body]
        self.qadr, self.vadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        self.geoms = {geom_id(model, "bottle"), geom_id(model, "bottle_cap")}
        self.floor = geom_id(model, "floor")
        self.zone = site_id(model, "place_zone")
        self.start_qpos = data.qpos[self.qadr:self.qadr + 7].copy()
        self.rest_z = float(self.start_qpos[2])
        self._model = model

    def reset_objects(self, model, data) -> None:
        super().reset_objects(model, data)
        data.qpos[self.qadr:self.qadr + 7] = self.start_qpos
        data.qvel[self.vadr:self.vadr + 6] = 0.0
        self._reset_state()

    def step(self, model, data, now: float) -> None:
        elapsed = self.clock(now)
        self.knocked_over = self.state == "reach" and self.tipped(data)
        {"reach": self._reach, "carry": self._carry, "placed": self._placed}[self.state](data, now, elapsed)

    def _on_floor(self, data) -> bool:
        return touching(self._model, data, self.geoms, lambda g: g == self.floor)

    def _reach(self, data, now: float, elapsed: float) -> None:
        lifted = data.xpos[self.body][2] > self.rest_z + LIFTED_DZ
        if lifted and not self._on_floor(data):
            self.state = "carry"
            self.grasp_s = self.grasp_s if self.grasp_s is not None else elapsed

    def _carry(self, data, now: float, elapsed: float) -> None:
        pos = data.xpos[self.body]
        speed = float(np.linalg.norm(data.qvel[self.vadr:self.vadr + 3]))
        if site_contains(self._model, data, self.zone, pos) and speed < AT_REST_SPEED:
            self._still_since = self._still_since if self._still_since is not None else now
            if now - self._still_since >= SETTLE_S:
                self._place(now, elapsed)
            return
        self._still_since = None
        if pos[2] < self.rest_z + DROPPED_DZ and self._on_floor(data):
            self.drops += 1
            self.state = "reach"
            print("Dropped -- pick it up again.")

    def _place(self, now: float, elapsed: float) -> None:
        self.finish_trial({"result": "success", "grasp_s": round(self.grasp_s, 2),
                           "place_s": round(elapsed, 2), "drops": self.drops})
        self.state = "placed"
        self._placed_at = now

    def _placed(self, data, now: float, elapsed: float) -> None:
        if now - self._placed_at >= RESET_AFTER_S:
            self.reset_objects(self._model, data)

    def tipped(self, data) -> bool:
        z_axis = data.xmat[self.body].reshape(3, 3)[:, 2]
        return z_axis[2] < TIPPED_COS

    def overlay(self, now: float) -> list[tuple[str, str]]:
        live = self.started_at is not None and self.state != "placed"
        timer = f"   ({now - self.started_at:.1f} s)" if live else ""
        rows = [("task", self.title), ("now", INSTRUCTIONS[self.state] + timer)]
        if self.knocked_over:
            rows.append(("!", "bottle knocked over -- operator: r to stand it up"))
        if self.drops:
            rows.append(("drops", str(self.drops)))
        if self.trials:
            last = self.trials[-1]
            rows.append(("placed", f"{len(self.trials)} bottle(s)"))
            rows.append(("last", f"grasped at {last['grasp_s']:.1f} s, placed at {last['place_s']:.1f} s, "
                                 f"{last['drops']} drop(s)"))
        return rows
