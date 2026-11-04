import functools

import numpy as np
import pytest

from stretch_demo.imu.bus import ImuBus
from stretch_demo.imu.fake_source import FakeSource
from stretch_demo.sim.base_drive import OmniDrive
from stretch_demo.sim.harness import SimHarness
from stretch_demo.tracking.imu_tracker import ImuFrame


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def fake_imus(clock):
    bus = ImuBus(2, clock=clock)
    return bus, FakeSource(bus)


@functools.cache
def _harness_parts(task_name: str):
    from stretch_demo.cli import TASKS
    harness = SimHarness(TASKS[task_name]())
    return harness, harness.data.qpos.copy(), harness.data.ctrl.copy()


def harness_for(task_name: str) -> SimHarness:
    """A built harness at its start state (models are cached: building is ~1 s)."""
    harness, qpos, ctrl = _harness_parts(task_name)
    harness.data.qpos[:] = qpos
    harness.data.qvel[:] = 0.0
    harness.data.ctrl[:] = ctrl
    harness.data.time = 0.0
    import mujoco
    mujoco.mj_forward(harness.model, harness.data)
    harness.task.__init__()
    harness.quit = False
    return harness


def frame(roll=0.0, pitch=0.0, yaw=0.0, gyro=(0.0, 0.0, 0.0)) -> ImuFrame:
    """An IMU frame from degrees (angles) and deg/s (gyro)."""
    return ImuFrame(np.deg2rad([roll, pitch, yaw]), np.deg2rad(gyro))


def live_controller(controller, harness):
    """Set a controller up on a harness and skip calibration."""
    controller.setup(harness.model, harness.data, harness.act)
    controller.calibrating = False
    controller.on_setup()
    return controller


class Scripted:
    """Drive a harness's robot directly by ctrl (no IMUs), stepping physics and
    scoring the task as the demo does -- proves a task is doable and scored."""

    def __init__(self, harness):
        self.h = harness
        self.m, self.d = harness.model, harness.data
        self.base = OmniDrive(self.m, self.d, harness.act)
        self.ctrl = self.d.ctrl.copy()
        self.t = 0.0
        harness.task.bind(self.m, self.d)

    def go(self, seconds: float, base: float = 0.0, **targets) -> None:
        import mujoco
        for name, value in targets.items():
            self.ctrl[self.h.act[name]] = value
        for i in range(round(seconds / self.m.opt.timestep)):
            self._apply(base)
            mujoco.mj_step(self.m, self.d)
            self.t += self.m.opt.timestep
            if i % 10 == 0:
                self.h.task.step(self.m, self.d, self.t)

    def _apply(self, base: float) -> None:
        ctrl = np.clip(self.ctrl, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
        self.base.drive(ctrl, 0.0, strafe=base, dt=self.m.opt.timestep)
        self.d.ctrl[:] = ctrl

    def ramp(self, name: str, end: float, rate: float = 0.15, base: float = 0.0) -> None:
        """Move one joint at a controller-like rate (a jump strips grasps)."""
        value = self.ctrl[self.h.act[name]]
        while abs(end - value) > 1e-9:
            value += float(np.clip(end - value, -rate * 0.01, rate * 0.01))
            self.go(0.01, base=base, **{name: value})

    def gripper(self, closed: bool) -> float:
        lo, hi = self.m.actuator_ctrlrange[self.h.act["gripper"]]
        return lo if closed else hi
