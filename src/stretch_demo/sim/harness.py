"""The sim loop: MuJoCo passive viewer + controller + task scoring + overlay.

Ported from mujoco-sim's sim/harness.py. Its run() (cognitive complexity 25) is
split into one method per job, and place_joints_at (12) per transmission type.
New: a task is scored every few ticks, its lines and the controller/IMU status
are drawn on the viewer, and terminal commands (stretch_demo.console) reset,
recalibrate or start the next visitor without restarting anything.
"""
import time

import mujoco
import mujoco.viewer

from stretch_demo.console import HELP, command_of
from stretch_demo.sim.robot import ACTUATORS
from stretch_demo.sim.scene import build_model

INITIAL_SETTLE_S = 1.0
TASK_EVERY_TICKS = 10      # ~50 Hz at the 2 ms timestep: plenty for scoring
OVERLAY_EVERY_S = 0.1


def place_joints_at(model, data, ctrl) -> None:
    """Set qpos so every position actuator already sits at its ctrl target (a
    teleport, not a commanded move). Actuators without a position servo's -kp
    bias term are left alone."""
    for i in range(model.nu):
        if model.actuator_biasprm[i, 1] == 0:  # no -kp bias term: not a position servo
            continue
        target = ctrl[i] / model.actuator_gear[i, 0]
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] = target
        elif model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
            _place_tendon(model, data, model.actuator_trnid[i, 0], target)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _place_tendon(model, data, tendon_id: int, target: float) -> None:
    """A fixed tendon (the arm's `extend`): spread the target over its joints by
    coefficient."""
    adr, num = model.tendon_adr[tendon_id], model.tendon_num[tendon_id]
    wraps = range(adr, adr + num)
    total = sum(model.wrap_prm[k] for k in wraps)
    for k in wraps:
        data.qpos[model.jnt_qposadr[model.wrap_objid[k]]] = target / total


class SimHarness:
    def __init__(self, task):
        self.task = task
        self.model = build_model(task.scene, task.tool)
        self.data = mujoco.MjData(self.model)
        self.act = self._resolve_actuators(ACTUATORS)
        self._place_start_pose()
        self._initial = (self.data.qpos.copy(), self.data.ctrl.copy())
        self._tick = 0
        self._next_overlay = 0.0
        self.quit = False

    # --- setup ---

    def _resolve_actuators(self, names) -> dict[str, int]:
        ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names}
        missing = [name for name, idx in ids.items() if idx < 0]
        if missing:
            raise RuntimeError(f"actuators {missing} not in the model: does the scene include stretch.xml?")
        return ids

    def _place_start_pose(self) -> None:
        """The task's start pose, then a short settle with the commands held."""
        m, d = self.model, self.data
        ctrl = d.ctrl.copy()
        for name, value in self.task.start_ctrl(m, self.act).items():
            ctrl[self.act[name]] = value
        place_joints_at(m, d, ctrl)
        for _ in range(round(INITIAL_SETTLE_S / m.opt.timestep)):
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
        d.time = 0.0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    def reset_all(self) -> None:
        """Robot and objects back to the start (next visitor)."""
        qpos, ctrl = self._initial
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)

    # --- the loop ---

    def run(self, controller, console, bus, log) -> None:
        controller.setup(self.model, self.data, self.act)
        self.task.bind(self.model, self.data)
        self._last_ctrl = None
        with mujoco.viewer.launch_passive(self.model, self.data,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            self._aim_camera(viewer)
            while viewer.is_running() and not self.quit:
                started = time.time()
                for line in console.take():
                    self._handle(line, controller, log)
                self._step_control(controller)
                self._step_task(controller, log)
                self._draw_overlay(viewer, controller, bus)
                viewer.sync()
                self._pace(started)

    def _step_control(self, controller) -> None:
        ctrl = controller.get_ctrl(self.model, self.data)
        if ctrl is not None:
            self._last_ctrl = ctrl
        if self._last_ctrl is not None:  # None = hold the last command
            self.data.ctrl[:] = self._last_ctrl
        mujoco.mj_step(self.model, self.data)
        self._tick += 1

    def _step_task(self, controller, log) -> None:
        # Not while calibrating: a trial's clock shouldn't include calibration.
        if self._tick % TASK_EVERY_TICKS or controller.calibrating:
            return
        self.task.step(self.model, self.data, controller.clock())
        log.write(self.task.take_trials())

    def _handle(self, line: str, controller, log) -> None:
        """One line from the operator: initials if the task is waiting for them,
        otherwise a command letter."""
        if self.task.awaiting_initials():
            self.task.submit_initials(line)
            return
        command = command_of(line)
        if command == "c":
            controller.recalibrate()
        elif command == "r":
            self.task.reset_objects(self.model, self.data)
            print("Task reset.")
        elif command == "n":
            self._next_visitor(controller, log)
        elif command == "q":
            self.quit = True
        elif line.strip():  # h, or anything unrecognized
            print(HELP)

    def _next_visitor(self, controller, log) -> None:
        self.reset_all()
        self.task.new_visitor(self.model, self.data)
        controller.setup(self.model, self.data, self.act)  # also recalibrates
        self._last_ctrl = None
        log.next_visitor()
        print(f"Visitor {log.visitor}: robot reset, hold still for calibration.")

    def _pace(self, started: float) -> None:
        # Always sleep a little, even when behind: a loop that never sleeps starves
        # the IMU and console threads of the GIL.
        remaining = self.model.opt.timestep - (time.time() - started)
        time.sleep(max(remaining, 0.001))

    # --- what the visitor sees ---

    def _aim_camera(self, viewer) -> None:
        lookat, distance, azimuth, elevation = self.task.camera
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = distance
        viewer.cam.azimuth = azimuth
        viewer.cam.elevation = elevation

    def _draw_overlay(self, viewer, controller, bus) -> None:
        now = time.monotonic()
        if now < self._next_overlay:
            return
        self._next_overlay = now + OVERLAY_EVERY_S
        status = [("controller", controller.name), ("", controller.status()), ("IMUs", imu_health(bus))]
        if controller.limits:
            status.insert(2, ("STUCK", controller.limits))
        boxes = [
            _text_box(mujoco.mjtGridPos.mjGRID_TOPLEFT, self.task.overlay(now)),
            _text_box(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, status),
        ]
        board = self.task.board_rows()
        if board:
            boxes.append(_text_box(mujoco.mjtGridPos.mjGRID_TOPRIGHT, board))
        viewer.set_texts(boxes)


def _text_box(gridpos, rows: list[tuple[str, str]]) -> tuple:
    """(label, value) rows -> the viewer's two-column text box."""
    labels = "\n".join(label for label, _ in rows)
    values = "\n".join(value for _, value in rows)
    return (mujoco.mjtFontScale.mjFONTSCALE_150, gridpos, labels, values)


def imu_health(bus) -> str:
    parts = []
    for index, (alive, rate) in enumerate(zip(bus.alive(), bus.rates())):
        parts.append(f"imu {index}: {rate:.0f} Hz" if alive else f"imu {index}: NO DATA")
    return "   ".join(parts)
