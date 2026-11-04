"""What the three tasks share: start pose, geometry tests, trial bookkeeping.

A task never touches the controller. The harness calls step() a few times a
second while the controller is live (not while calibrating, so a trial's clock
never includes calibration), and shows overlay() on the viewer. Finished trials
are handed to the TrialLog through take_trials().

Times are wall-clock (time.monotonic): it's the visitor's experience being
measured, and the sim runs in real time anyway.
"""
from pathlib import Path

import mujoco
import numpy as np

SCENES = Path(__file__).resolve().parents[3] / "scenes"


class Task:
    name = "task"
    title = "TASK"
    scene: Path
    tool: Path | None = None
    gripper_enabled = True
    # Stretch 4 joint commands the task starts from, and whether the gripper starts closed.
    start_pose = {"lift": 0.66, "arm": 0.05, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0}
    start_gripper_closed = False
    # Viewer camera: (lookat xyz, distance, azimuth, elevation).
    camera = ((0.6, 0.0, 0.6), 2.0, 55.0, -20.0)

    def __init__(self):
        self.leaderboard = None         # set by the CLI; tasks with rounds use it
        self.trials: list[dict] = []    # this visitor's finished trials
        self._pending: list[dict] = []  # not yet handed to the log
        self.started_at: float | None = None

    # --- harness interface ---

    def start_ctrl(self, model, act) -> dict[str, float]:
        ctrl = dict(self.start_pose)
        lo, hi = model.actuator_ctrlrange[act["gripper"]]
        ctrl["gripper"] = lo if self.start_gripper_closed or not self.gripper_enabled else hi
        return ctrl

    def bind(self, model, data) -> None:
        """Look up ids and remember the objects' start state."""

    def reset_objects(self, model, data) -> None:
        """`r`: objects back to their start, trial restarted."""
        self.started_at = None

    def new_visitor(self, model, data) -> None:
        self.trials = []
        self.reset_objects(model, data)

    def step(self, model, data, now: float) -> None:
        raise NotImplementedError

    def overlay(self, now: float) -> list[tuple[str, str]]:
        raise NotImplementedError

    def awaiting_initials(self) -> bool:
        """True while a finished round waits for the operator to type initials."""
        return False

    def submit_initials(self, text: str) -> None:
        """The operator's line while awaiting_initials()."""

    def board_rows(self) -> list[tuple[str, str]]:
        """Extra overlay box (top right), e.g. the leaderboard. Empty = none."""
        return []

    def take_trials(self) -> list[dict]:
        out, self._pending = self._pending, []
        return out

    # --- helpers ---

    def clock(self, now: float) -> float:
        """Seconds since this trial started (starts on the first live tick)."""
        if self.started_at is None:
            self.started_at = now
        return now - self.started_at

    def finish_trial(self, record: dict) -> None:
        record = {"trial": len(self.trials) + 1, **record}
        self.trials.append(record)
        self._pending.append(record)
        self.started_at = None

    def successes(self) -> list[dict]:
        return [t for t in self.trials if t["result"] == "success"]


class ContactDebounce:
    """Turns a 'touching now?' signal into hit events: True once when a touch
    begins, then not again until it has been clear for clear_s (so one bump
    that chatters on and off counts once)."""

    def __init__(self, clear_s: float):
        self.clear_s = clear_s
        self.touching = False
        self.clear_since: float | None = None

    def update(self, touching_now: bool, now: float) -> bool:
        if touching_now:
            new = not self.touching
            self.touching, self.clear_since = True, None
            return new
        if self.touching and self.clear_since is None:
            self.clear_since = now
        elif self.touching and now - self.clear_since >= self.clear_s:
            self.touching = False
        return False


def site_id(model, name: str) -> int:
    return _require(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name), "site", name)


def body_id(model, name: str) -> int:
    return _require(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name), "body", name)


def geom_id(model, name: str) -> int:
    return _require(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name), "geom", name)


def _require(idx: int, kind: str, name: str) -> int:
    if idx < 0:
        raise ValueError(f"the scene has no {kind} named {name!r}")
    return idx


def _in_box(local, size) -> bool:
    return bool(np.all(np.abs(local) <= size))


def _in_cylinder(local, size) -> bool:
    return bool(np.hypot(local[0], local[1]) <= size[0] and abs(local[2]) <= size[1])


def _in_sphere(local, size) -> bool:
    return bool(np.linalg.norm(local) <= size[0])


_CONTAINS = {  # keyed by int: model.site_type holds plain ints
    int(mujoco.mjtGeom.mjGEOM_BOX): _in_box,
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): _in_cylinder,
    int(mujoco.mjtGeom.mjGEOM_SPHERE): _in_sphere,
}


def site_contains(model, data, site: int, point) -> bool:
    """Is a world point inside a box/cylinder/sphere site?"""
    local = data.site_xmat[site].reshape(3, 3).T @ (np.asarray(point) - data.site_xpos[site])
    return _CONTAINS[int(model.site_type[site])](local, model.site_size[site])


def touching(model, data, geoms_a: set[int], other) -> bool:
    """Any contact between a geom in geoms_a and a geom for which other(geom) is True."""
    for contact in data.contact[: data.ncon]:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if (g1 in geoms_a and other(g2)) or (g2 in geoms_a and other(g1)):
            return True
    return False


def mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")
