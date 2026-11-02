"""Stretch 4 omni base drive (from mujoco-sim sim/base_drive.py): turn a
normalized (forward, strafe, turn) command into wheel actuator ctrl.

Commands are fractions of full speed in [-1, 1]: forward = +x (the way the base
faces), strafe = +y (to its left), turn = counterclockwise seen from above.

Write the base AFTER clipping the rest of ctrl to ctrlrange: the wheel actuators
are unlimited, and MuJoCo reports their ctrlrange as [0, 0], so a blanket
np.clip would pin them at zero.
"""
import numpy as np

# Full speeds (mujoco-sim matched them to the Stretch 3 it drove before) and
# 0.5 s from stop to full speed.
OMNI_FULL_SPEED = np.array([0.088, 0.088, 0.55])  # vx m/s, vy m/s, wz rad/s
OMNI_SLEW_RATE = 2.0                                # full-speed fractions per second

# Omnibase geometry and wheel layout, from stretch4_mujoco (config.py "omnibase",
# utils.H0_from_driving_dir with forward_dir "calder").
OMNI_WHEEL_RADIUS = 0.10
OMNI_BASE_RADIUS = 0.174
OMNI_GAMMA = np.radians(30.0)
OMNI_WHEELS = ("left_wheel_vel", "back_wheel_vel", "right_wheel_vel")


class OmniDrive:
    """Stretch 4: three omniwheels on position actuators. Holonomic, so strafe works.

    The wheel actuators are position servos (gear 6), not velocity ones despite
    their `_vel` names, so this integrates wheel speed into a position target.
    """

    def __init__(self, model, data, act: dict[str, int]):
        self.idx = [act[name] for name in OMNI_WHEELS]
        self.gear = model.actuator_gear[self.idx, 0]
        self.position = data.ctrl[self.idx].copy()
        self.state = np.zeros(3)  # slewed (forward, strafe, turn)
        angles = OMNI_GAMMA + np.radians([0.0, 120.0, 240.0])
        # Wheel angular velocity = H @ (vx, vy, wz).
        self.H = np.column_stack(
            [np.sin(angles), -np.cos(angles), np.full(3, -OMNI_BASE_RADIUS)]
        ) / OMNI_WHEEL_RADIUS

    def drive(self, ctrl, forward: float, turn: float = 0.0, strafe: float = 0.0, *, dt: float) -> None:
        target = np.clip([forward, strafe, turn], -1.0, 1.0)
        max_delta = OMNI_SLEW_RATE * dt
        self.state += np.clip(target - self.state, -max_delta, max_delta)
        wheel_speed = self.H @ (OMNI_FULL_SPEED * self.state)
        self.position += wheel_speed * self.gear * dt
        ctrl[self.idx] = self.position

    @property
    def wheels(self) -> dict[str, float]:
        """Wheel position targets last written, for telemetry."""
        return dict(zip(OMNI_WHEELS, self.position.tolist()))

    def feedback_signals(self) -> dict[str, float]:
        forward, strafe, turn = self.state
        return {"base_forward": float(forward), "base_turn": float(turn),
                "base_strafe": float(strafe)}
