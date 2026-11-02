"""In-process IMU samples.

Same field names and units as hre_schemas.schemas.imu.ImuData (what ximu-api put
on the wire and port-listener stored in Redis), so the tracking code ported from
mujoco-sim reads them unchanged -- but plain frozen dataclasses, since nothing is
serialized any more.
"""
from dataclasses import dataclass

R3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class InertialSample:
    timestamp: int      # device clock, microseconds since power-on
    acceleration: R3    # g
    gyroscope: R3       # deg/s


@dataclass(frozen=True, slots=True)
class QuaternionSample:
    timestamp: int
    quaternion: tuple[float, float, float, float]  # (w, x, y, z), scalar FIRST


@dataclass(frozen=True, slots=True)
class MagnetometerSample:
    timestamp: int
    magnetometer: R3    # arbitrary units


@dataclass(frozen=True, slots=True)
class ImuBatch:
    """Everything one IMU produced since the previous batch. `index` is the IMU's
    position in --imu-ip (0 = first), nothing about where it is worn."""

    index: int
    inertial: tuple[InertialSample, ...] = ()
    quaternion: tuple[QuaternionSample, ...] = ()
    magnetometer: tuple[MagnetometerSample, ...] = ()
