"""Small signal-shaping pieces shared by the controllers (from mujoco-sim's
retarget/imu_hands*.py, bio_m2_v2 and head_shoulder, where each was repeated)."""
import numpy as np

# Leaky baseline that soaks up magnetometer-referenced yaw drift (imu_hands'
# YAW_DRIFT_TAU_S): sustained drift over tens of seconds is cancelled, a gesture
# shorter than this still comes through.
YAW_DRIFT_TAU_S = 45.0


def shape(x: float, start: float, full: float, expo: float = 0.0) -> float:
    """Signed drive in [-1, 1]: 0 inside +-start, ramping to +-1 at +-full, bent by
    `expo` (0 = linear, 1 = cubic: finer control just past the deadband)."""
    u = np.clip((abs(x) - start) / (full - start), 0.0, 1.0)
    return float(np.sign(x) * ((1.0 - expo) * u + expo * u**3))


def deadband(value: float, band: float) -> float:
    return 0.0 if abs(value) < band else value


class YawDriftBaseline:
    def __init__(self, tau_s: float = YAW_DRIFT_TAU_S):
        self.tau_s = tau_s
        self.value = 0.0

    def correct(self, yaw: float, dt: float) -> float:
        """Track yaw slowly and return it with the drifted-away zero removed."""
        self.value += min(1.0, dt / self.tau_s) * (yaw - self.value)
        return yaw - self.value

    def reset(self) -> None:
        self.value = 0.0


class RollUnwrap:
    """Keeps roll continuous across +-180 deg (a forearm twist past it would
    otherwise jump by a full turn)."""

    def __init__(self):
        self.prev: float | None = None
        self.offset = 0.0

    def __call__(self, raw: float) -> float:
        if self.prev is not None:
            jump = raw - self.prev
            self.offset -= 2 * np.pi * np.round(jump / (2 * np.pi))
        self.prev = raw
        return raw + self.offset


class Ema:
    """Per-name exponential smoothing (alpha per tick)."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.values: dict[str, float] = {}

    def __call__(self, name: str, value: float) -> float:
        prev = self.values.get(name)
        out = value if prev is None else prev + self.alpha * (value - prev)
        self.values[name] = out
        return out


class Setpoints:
    """Joint commands, each clamped to its ctrlrange on every write -- a rate-
    integrated setpoint can't wind past a limit (what bit bio_m2/bio_m3)."""

    def __init__(self, initial: dict[str, float], limits: dict[str, tuple[float, float]]):
        self.values = dict(initial)
        self.limits = limits

    def set(self, name: str, value: float) -> None:
        lo, hi = self.limits[name]
        self.values[name] = float(np.clip(value, lo, hi))

    def step(self, name: str, delta: float) -> None:
        self.set(name, self.values[name] + delta)

    def __getitem__(self, name: str) -> float:
        return self.values[name]
