import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stretch_demo.imu.bus import ImuBus
from stretch_demo.imu.types import ImuBatch, InertialSample, QuaternionSample
from stretch_demo.imu.ximu_source import ImuConnectError, SensorState, XimuSource, _wait_for_tick
from stretch_demo.tracking.imu_tracker import ImuTracker, NeutralCalibration, latest_quaternion


def batch(index, t, gyro=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    return ImuBatch(index, (InertialSample(t, (0.0, 0.0, 1.0), gyro),), (QuaternionSample(t, quat),))


def test_bus_goes_stale_after_ttl(clock):
    bus = ImuBus(2, clock=clock)
    assert bus.latest(0) is None
    bus.publish(batch(0, 1))
    assert bus.latest(0) is not None and bus.latest(1) is None
    clock.advance(0.6)
    assert bus.latest(0) is None
    assert bus.alive() == [False, False]


def test_bus_merges_recent_batches_so_no_gyro_peak_is_lost(clock):
    bus = ImuBus(1, clock=clock)
    bus.publish(batch(0, 1, gyro=(0.0, 0.0, 500.0)))  # a flick peak...
    bus.publish(batch(0, 2))                          # ...then a quiet batch on top
    merged = bus.latest(0)
    assert [s.timestamp for s in merged.inertial] == [1, 2]
    assert max(s.gyroscope[2] for s in merged.inertial) == 500.0


def test_bus_ignores_unknown_index_and_reports_rates(clock):
    bus = ImuBus(1, clock=clock)
    bus.publish(batch(5, 1))
    for t in range(400):
        bus.publish(batch(0, t))
    clock.advance(1.0)
    assert bus.rates() == [pytest.approx(400.0)]


def test_quaternion_is_read_scalar_first():
    # 90 deg about z, (w, x, y, z): read scalar-last it would be a different rotation.
    q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    rot = latest_quaternion(batch(0, 1, quat=q))
    assert np.allclose(rot.as_euler("ZYX", degrees=True), [90, 0, 0], atol=1e-6)


def _calibrate(source, bus, clock, trackers, seconds=3.0):
    calibration = NeutralCalibration(trackers)
    done = False
    for _ in range(int(seconds * 100)):
        source.publish_once()
        clock.advance(0.01)
        done = calibration.step({i: bus.latest(i) for i in trackers}, clock())
        if done:
            break
    return done


def test_tracker_calibrates_then_reads_angles_from_neutral(fake_imus, clock):
    bus, source = fake_imus
    source.set_pose(0, 0.0, 0.0, 30.0)  # starts turned: that becomes neutral
    trackers = {0: ImuTracker("a"), 1: ImuTracker("b")}
    assert _calibrate(source, bus, clock, trackers)
    source.set_pose(0, 10.0, -20.0, 30.0 + 15.0)
    source.publish_once()
    roll, pitch, yaw = np.rad2deg(trackers[0].update(bus.latest(0)).angles)
    assert (roll, pitch, yaw) == pytest.approx((10.0, -20.0, 15.0), abs=0.5)


def test_calibration_waits_while_the_imu_moves(fake_imus, clock):
    bus, source = fake_imus
    trackers = {0: ImuTracker("a")}
    calibration = NeutralCalibration(trackers)
    for k in range(300):
        source.set_pose(0, 10.0 * np.sin(k / 10), 0.0, 0.0)
        source.publish_once()
        clock.advance(0.01)
        assert not calibration.step({0: bus.latest(0)}, clock())
    assert "moved" in calibration.progress(clock())


def test_tracker_gyro_is_bias_removed_peak(fake_imus, clock):
    bus, source = fake_imus
    trackers = {0: ImuTracker("a"), 1: ImuTracker("b")}
    assert _calibrate(source, bus, clock, trackers)
    source.set_pose(1, 0.0, 0.0, 3.0)  # 3 deg in one 10 ms batch = 300 deg/s about z
    source.publish_once()
    gyro = np.rad2deg(trackers[1].update(bus.latest(1)).gyro)
    assert gyro[2] == pytest.approx(300.0, rel=0.02)
    assert abs(gyro[0]) < 1 and abs(gyro[1]) < 1


def test_device_heading_swing_with_still_gyros_is_ignored(fake_imus, clock):
    """Seen on the real x-IMU3s: the device's heading swung 28 deg (and showed as
    roll on a tilted IMU) while the gyros read ~0. Only gyro-backed motion counts."""
    bus, source = fake_imus
    source.set_pose(0, 0.0, 70.0, 0.0)  # worn tilted, so heading leaks into roll
    trackers = {0: ImuTracker("a"), 1: ImuTracker("b")}
    assert _calibrate(source, bus, clock, trackers)
    swung = Rotation.from_euler("z", 28, degrees=True) * Rotation.from_euler("ZYX", [0, 70, 0], degrees=True)
    batch = bus.latest(0)
    q = tuple(swung.as_quat(scalar_first=True))
    still = [InertialSample(s.timestamp + 10_000_000, s.acceleration, (0.0, 0.0, 0.0)) for s in batch.inertial]
    fake = ImuBatch(0, tuple(still), (QuaternionSample(still[-1].timestamp, q),))
    device_angles = np.rad2deg((trackers[0].ref.inv() * swung).as_euler("ZYX"))
    assert max(abs(device_angles[0]), abs(device_angles[2])) > 20  # what the old tracker used
    angles = np.rad2deg(trackers[0].update(fake).angles)
    assert np.abs(angles).max() < 1.0


def test_gyro_rotation_still_tracks_real_motion(fake_imus, clock):
    bus, source = fake_imus
    trackers = {0: ImuTracker("a"), 1: ImuTracker("b")}
    assert _calibrate(source, bus, clock, trackers)
    poses = [(0.0, 0.0, 2.0 * k) for k in range(1, 16)]       # turn 30 deg about z,
    poses += [(0.0, 20.0 * k / 15, 30.0) for k in range(1, 16)]  # then tip 20 deg
    for roll, pitch, yaw in poses:
        source.set_pose(0, roll, pitch, yaw)
        source.publish_once()
        frame = trackers[0].update(bus.latest(0))
    roll, pitch, yaw = np.rad2deg(frame.angles)
    assert (roll, pitch, yaw) == pytest.approx((0.0, 20.0, 30.0), abs=1.0)


class FakeConnection:
    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []  # pass one list to share the order

    def add_inertial_callback(self, cb):
        self.calls.append("add")
        return 1

    add_quaternion_callback = add_magnetometer_callback = add_inertial_callback

    def remove_callback(self, cid):
        self.calls.append("remove")

    def send_command(self, cmd):
        self.calls.append("strobe")

    def close(self):
        self.calls.append("close")


def test_sensor_state_drains_into_a_batch():
    sensor = SensorState(FakeConnection(), 1, "ip")
    assert sensor.drain() is None

    class Msg:
        timestamp, w, x, y, z = 5, 1.0, 0.0, 0.0, 0.0
        accelerometer_x = accelerometer_y = gyroscope_x = gyroscope_y = gyroscope_z = 0.0
        accelerometer_z = 1.0

    sensor.on_inertial(Msg)
    sensor.on_quaternion(Msg)
    drained = sensor.drain()
    assert drained.index == 1 and len(drained.inertial) == 1 and drained.quaternion[0].quaternion == (1, 0, 0, 0)
    assert sensor.drain() is None


def test_shutdown_unregisters_every_sensor_before_closing_any():
    source = XimuSource(ImuBus(2), ips=("a", "b"))
    calls = []
    connections = [FakeConnection(calls), FakeConnection(calls)]
    source.sensors = [SensorState(c, i, ip) for i, (c, ip) in enumerate(zip(connections, "ab"))]
    for sensor in source.sensors:
        sensor.register()
    source.stop()
    order = [call for call in calls if call in ("remove", "close")]
    assert order == ["remove"] * 6 + ["close"] * 2
    assert source.sensors == []


def test_connect_failure_names_the_ip(monkeypatch):
    class Failing:
        def __init__(self, config):
            pass

        def open(self):
            raise RuntimeError("timeout")

    fake_sdk = type("ximu3", (), {"Connection": Failing, "TcpConnectionConfig": lambda ip, port: None})
    monkeypatch.setattr("stretch_demo.imu.ximu_source._import_ximu3", lambda: fake_sdk)
    with pytest.raises(ImuConnectError, match="10.0.0.9"):
        XimuSource(ImuBus(1), ips=("10.0.0.9",)).start()


def test_drain_loop_skips_missed_ticks_instead_of_spinning():
    assert _wait_for_tick(0.0, 0.01) > 0.0
