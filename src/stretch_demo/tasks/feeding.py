"""Self-feeding: the robot holds a spoon; scoop from the plate, bring the bite to
the seated person's mouth (green box). The gripper stays closed on the spoon.

Each trial is one bite:
  SCOOP  spoon tip enters the plate zone -> the bite appears on the spoon
  CARRY  tip travels toward the mouth; time and path length accumulate
         (dipping back into the plate restarts the carry)
  MOUTH  tip inside the mouth box; for DWELL_S or until it leaves, the closest
         approach to the box center is the bite's ACCURACY
  A tip that touches the face outside the mouth box is a missed bite.

Per bite: plate->mouth time, accuracy (cm from mouth center), and path
efficiency (straight-line plate->mouth distance / distance actually travelled,
1.0 = perfectly direct).
"""
import numpy as np

from stretch_demo.tasks.base import SCENES, Task, mean, site_contains, site_id

DWELL_S = 1.0
FOOD_RGBA_ALPHA = 1.0

INSTRUCTIONS = {
    "scoop": "dip the spoon into the plate",
    "carry": "bring the bite to the mouth (green box)",
    "mouth": "hold it there...",
}


class FeedingTask(Task):
    name = "feeding"
    title = "SELF-FEEDING"
    scene = SCENES / "feeding.xml"
    tool = SCENES / "utensil.xml"
    gripper_enabled = False
    start_pose = {"lift": 0.76, "arm": 0.10, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0}
    camera = ((0.772, -0.066, 0.95), 2.0, 55.0, -18.0)

    def __init__(self):
        super().__init__()
        self.state = "scoop"
        self._carry: dict = {}

    def bind(self, model, data) -> None:
        self.tip = site_id(model, "utensil_tip")
        self.plate = site_id(model, "plate_zone")
        self.mouth = site_id(model, "mouth_box")
        self.face = site_id(model, "face_zone")
        self.food = site_id(model, "food_on_utensil")
        self._model = model
        self._show_food(False)

    def reset_objects(self, model, data) -> None:
        super().reset_objects(model, data)
        self.state = "scoop"
        self._show_food(False)

    def step(self, model, data, now: float) -> None:
        tip = data.site_xpos[self.tip].copy()
        self.clock(now)
        {"scoop": self._scoop, "carry": self._carry_step, "mouth": self._at_mouth}[self.state](data, tip, now)

    def _scoop(self, data, tip, now: float) -> None:
        if site_contains(self._model, data, self.plate, tip):
            self._start_carry(tip, now)

    def _start_carry(self, tip, now: float) -> None:
        self.state = "carry"
        self._carry = {"t0": now, "start": tip, "prev": tip, "path": 0.0}
        self._show_food(True)

    def _carry_step(self, data, tip, now: float) -> None:
        c = self._carry
        c["path"] += float(np.linalg.norm(tip - c["prev"]))
        c["prev"] = tip
        if site_contains(self._model, data, self.plate, tip):
            self._start_carry(tip, now)  # dipped again: the carry starts over
        elif site_contains(self._model, data, self.mouth, tip):
            self.state = "mouth"
            c.update(arrived=now, entry=tip, best=self._error(data, tip))
        elif site_contains(self._model, data, self.face, tip):
            self._end("missed (touched the face)", data, tip, now)

    def _at_mouth(self, data, tip, now: float) -> None:
        c = self._carry
        inside = site_contains(self._model, data, self.mouth, tip)
        if inside:
            c["best"] = min(c["best"], self._error(data, tip))
        if not inside or now - c["arrived"] >= DWELL_S:
            self._end("success", data, c["entry"], c["arrived"])

    def _end(self, result: str, data, end_tip, end_time: float) -> None:
        c = self._carry
        straight = float(np.linalg.norm(end_tip - c["start"]))
        self.finish_trial({
            "result": result,
            "time_s": round(end_time - c["t0"], 2),
            "error_cm": round(100 * c["best"], 2) if result == "success" else float("nan"),
            "path_efficiency": round(min(1.0, straight / max(c["path"], 1e-6)), 2),
        })
        self.state = "scoop"
        self._show_food(False)

    def _error(self, data, tip) -> float:
        return float(np.linalg.norm(tip - data.site_xpos[self.mouth]))

    def _show_food(self, shown: bool) -> None:
        self._model.site_rgba[self.food, 3] = FOOD_RGBA_ALPHA if shown else 0.0

    def overlay(self, now: float) -> list[tuple[str, str]]:
        rows = [("task", self.title), ("now", self._instruction(now))]
        ok = self.successes()
        if self.trials:
            rows.append(("bites", f"{len(ok)} of {len(self.trials)} delivered ({len(ok) / len(self.trials):.0%})"))
            last = self.trials[-1]
            rows.append(("last", _bite_text(last)))
        if ok:
            rows.append(("average", f"{mean(t['time_s'] for t in ok):.1f} s, "
                                    f"{mean(t['error_cm'] for t in ok):.1f} cm from center, "
                                    f"{mean(t['path_efficiency'] for t in ok):.0%} direct"))
        return rows

    def _instruction(self, now: float) -> str:
        text = INSTRUCTIONS[self.state]
        if self.state == "carry":
            text += f"   ({now - self._carry['t0']:.1f} s)"
        return text


def _bite_text(trial: dict) -> str:
    if trial["result"] != "success":
        return f"{trial['result']} after {trial['time_s']:.1f} s"
    return f"{trial['time_s']:.1f} s, {trial['error_cm']:.1f} cm from center, {trial['path_efficiency']:.0%} direct"
