"""Self-feeding, played as a round of ATTEMPTS bites for points.

The robot holds a spoon (the gripper stays closed on it). Each attempt:
  SCOOP  spoon tip enters the plate zone -> the bite appears on the spoon
  CARRY  tip travels toward the mouth; time and path length accumulate
         (dipping back into the plate restarts the carry, not a new attempt)
  MOUTH  tip inside the mouth box; for DWELL_S or until it leaves, the closest
         approach to the box center is the bite's accuracy -> +BITE_POINTS
  A spoon that pokes the face (outside the mouth box) while carrying ends the
  attempt as a miss.

Touching the person's face, by any part of the robot (it collides with the
head) or with the spoon tip outside the mouth box, costs FACE_PENALTY, at any
time in the round. One touch counts once: it re-arms after FACE_CLEAR_S clear.
The table also collides but isn't penalized: it only blocks the robot.

After ATTEMPTS attempts the round is over: the final score and time show on
the viewer, the operator types the visitor's initials in the terminal, and the
score goes on the leaderboard (tasks/leaderboard.py). `r` plays another round,
`n` starts the next visitor.

Logged per attempt: plate->mouth time, accuracy (cm from mouth center), path
efficiency (straight-line distance / distance travelled), face touches since
the previous attempt, and the points it earned (bite minus penalties).
"""
import numpy as np

from stretch_demo.sim.robot import GRASP_BODY
from stretch_demo.tasks.base import (SCENES, ContactDebounce, Task, body_id, geom_id,
                                     site_contains, site_id, touching)
from stretch_demo.tasks.leaderboard import clock_text, normalize_initials

ATTEMPTS = 5
BITE_POINTS = 100
FACE_PENALTY = 50
FACE_CLEAR_S = 0.5
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
        self.round = 0
        self._new_round()

    def _new_round(self) -> None:
        self.round += 1
        self.state = "scoop"
        self._carry: dict = {}
        self.attempts = 0
        self.bites = 0
        self.score = 0
        self.face_hits = 0
        self._hits_this_attempt = 0
        self.round_start: float | None = None
        self.round_end: float | None = None
        self.rank: int | None = None
        self._awaiting = False
        self._face = ContactDebounce(FACE_CLEAR_S)

    @property
    def round_over(self) -> bool:
        return self.attempts >= ATTEMPTS

    # --- harness interface ---

    def bind(self, model, data) -> None:
        self.tip = site_id(model, "utensil_tip")
        self.plate = site_id(model, "plate_zone")
        self.mouth = site_id(model, "mouth_box")
        self.face = site_id(model, "face_zone")
        self.food = site_id(model, "food_on_utensil")
        self.head = {geom_id(model, "head")}
        robot_root = model.body_rootid[body_id(model, GRASP_BODY)]
        self.is_robot = lambda g: model.body_rootid[model.geom_bodyid[g]] == robot_root
        self._model = model
        self._show_food(False)

    def reset_objects(self, model, data) -> None:
        """`r`: start a new round (play again)."""
        super().reset_objects(model, data)
        self._new_round()
        self._show_food(False)

    def new_visitor(self, model, data) -> None:
        self.round = 0
        super().new_visitor(model, data)

    def step(self, model, data, now: float) -> None:
        if self.round_over:
            return
        if self.round_start is None:
            self.round_start = now
        tip = data.site_xpos[self.tip].copy()
        self._check_face(model, data, tip, now)
        if not self.round_over:
            {"scoop": self._scoop, "carry": self._carry_step, "mouth": self._at_mouth}[self.state](data, tip, now)

    # --- face touches ---

    def _check_face(self, model, data, tip, now: float) -> None:
        spoon_in_face = self._spoon_in_face(data, tip)
        robot_on_head = touching(model, data, self.head, self.is_robot)
        if not self._face.update(spoon_in_face or robot_on_head, now):
            return
        self.face_hits += 1
        self._hits_this_attempt += 1
        self.score -= FACE_PENALTY
        print(f"Touched the face: -{FACE_PENALTY} (score {self.score})")
        if spoon_in_face and self.state == "carry":
            self._end("missed (touched the face)", tip, now, now)

    def _spoon_in_face(self, data, tip) -> bool:
        in_mouth = site_contains(self._model, data, self.mouth, tip)
        return not in_mouth and site_contains(self._model, data, self.face, tip)

    # --- one attempt ---

    def _scoop(self, data, tip, now: float) -> None:
        if site_contains(self._model, data, self.plate, tip):
            self._start_carry(tip, now)

    def _start_carry(self, tip, now: float) -> None:
        self.state = "carry"
        self._carry = {"t0": now, "start": tip, "prev": tip, "path": 0.0, "best": float("nan")}
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

    def _at_mouth(self, data, tip, now: float) -> None:
        c = self._carry
        inside = site_contains(self._model, data, self.mouth, tip)
        if inside:
            c["best"] = min(c["best"], self._error(data, tip))
        if not inside or now - c["arrived"] >= DWELL_S:
            self._end("success", c["entry"], c["arrived"], now)

    def _end(self, result: str, end_tip, end_time: float, now: float) -> None:
        c = self._carry
        bite = result == "success"
        points = (BITE_POINTS if bite else 0) - FACE_PENALTY * self._hits_this_attempt
        self.score += BITE_POINTS if bite else 0
        self.bites += bite
        self.attempts += 1
        straight = float(np.linalg.norm(end_tip - c["start"]))
        self.finish_trial({
            "round": self.round, "attempt": self.attempts, "result": result,
            "time_s": round(end_time - c["t0"], 2),
            "error_cm": round(100 * c["best"], 2) if bite else float("nan"),
            "path_efficiency": round(min(1.0, straight / max(c["path"], 1e-6)), 2),
            "face_hits": self._hits_this_attempt, "points": points,
        })
        self._hits_this_attempt = 0
        self.state = "scoop"
        self._show_food(False)
        if self.round_over:
            self._finish_round(now)

    def _finish_round(self, now: float) -> None:
        self.round_end = now
        self._awaiting = self.leaderboard is not None
        print(f"\nROUND OVER: {self.score} points in {clock_text(self.round_time(now))} "
              f"({self.bites}/{ATTEMPTS} bites, {self.face_hits} face touches).")
        if self._awaiting:
            print("Type the visitor's initials (up to 3) and press Enter -- or just Enter to skip.")

    def round_time(self, now: float) -> float:
        if self.round_start is None:
            return 0.0
        return (self.round_end if self.round_end is not None else now) - self.round_start

    def _error(self, data, tip) -> float:
        return float(np.linalg.norm(tip - data.site_xpos[self.mouth]))

    def _show_food(self, shown: bool) -> None:
        self._model.site_rgba[self.food, 3] = FOOD_RGBA_ALPHA if shown else 0.0

    # --- leaderboard ---

    def awaiting_initials(self) -> bool:
        return self._awaiting

    def submit_initials(self, text: str) -> None:
        self._awaiting = False
        initials = normalize_initials(text)
        if initials is None:
            print("Score not saved. r = play again, n = next visitor.")
            return
        self.rank = self.leaderboard.add(initials, self.score, self.round_time(self.round_end),
                                         bites=self.bites, face_hits=self.face_hits)
        print(f"{initials}: #{self.rank} on the {self.leaderboard.controller} board. "
              "r = play again, n = next visitor.")

    def board_rows(self) -> list[tuple[str, str]]:
        return self.leaderboard.rows() if self.leaderboard is not None else []

    # --- overlay ---

    def overlay(self, now: float) -> list[tuple[str, str]]:
        if self.round_over:
            return self._final_rows()
        rows = [("task", self.title), ("now", self._instruction(now)),
                ("bite", f"{self.attempts + 1} of {ATTEMPTS}"),
                ("score", f"{self.score}   ({self.bites} bites, {self.face_hits} face touches)"),
                ("time", clock_text(self.round_time(now)))]
        if self.trials and self.trials[-1]["round"] == self.round:
            rows.append(("last", _bite_text(self.trials[-1])))
        return rows

    def _final_rows(self) -> list[tuple[str, str]]:
        rows = [("task", self.title),
                ("FINAL", f"{self.score} points in {clock_text(self.round_time(self.round_end))}"),
                ("", f"{self.bites} of {ATTEMPTS} bites, {self.face_hits} face touches")]
        if self._awaiting:
            rows.append(("next", "operator: type initials + Enter"))
        elif self.rank is not None:
            rows.append(("rank", f"#{self.rank}!   (operator: r = play again, n = next visitor)"))
        else:
            rows.append(("next", "operator: r = play again, n = next visitor"))
        return rows

    def _instruction(self, now: float) -> str:
        text = INSTRUCTIONS[self.state]
        if self.state == "carry":
            text += f"   ({now - self._carry['t0']:.1f} s)"
        return text


def _bite_text(trial: dict) -> str:
    if trial["result"] != "success":
        return f"{trial['result']}: {trial['points']:+d}"
    return f"{trial['points']:+d}: {trial['time_s']:.1f} s, {trial['error_cm']:.1f} cm from center"
