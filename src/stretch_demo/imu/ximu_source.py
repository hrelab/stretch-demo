"""x-IMU3 devices -> ImuBus, in this process.

A port of ximu-api/src/ximu_api/main.py with the ZMQ socket (and port-listener and
Redis after it) replaced by ImuBus.publish. Same devices, commands, callbacks and
100 Hz drain loop; the loop runs on a daemon thread so the sim owns the main
thread.

Sensors are found by serial number by default (see imu/discovery.py), so their
IPs can be anything; pass ips to connect to fixed addresses instead.

ximu3 is imported lazily so `--imu fake` runs without the SDK installed.
"""
import threading
import time

from stretch_demo.imu import discovery
from stretch_demo.imu.bus import ImuBus
from stretch_demo.imu.types import ImuBatch, InertialSample, MagnetometerSample, QuaternionSample

DEFAULT_SERIALS = ("6A3409FC", "6A32706F")  # ximu-api's SENSOR_SERIALS: imu 0, imu 1
DISCOVERY_TIMEOUT_S = 10.0
SENSOR_PORT = 7000  # with fixed ips; a looked-up sensor uses the port it announces
EMIT_HZ = 100
STARTUP_COMMANDS = ['{"colour":null}', '{"strobe":null}']  # blink: "this one is connected"


class ImuConnectError(RuntimeError):
    pass


class SensorState:
    """Per-sensor buffers. ximu3 callbacks (SDK threads) append under the lock;
    drain() swaps the lists out and builds the batch off the lock."""

    def __init__(self, connection, index: int, ip: str):
        self.connection = connection
        self.index = index
        self.ip = ip
        self.lock = threading.Lock()
        self._inertial: list[InertialSample] = []
        self._quaternion: list[QuaternionSample] = []
        self._magnetometer: list[MagnetometerSample] = []
        self._callback_ids: list[int] = []

    def on_inertial(self, m) -> None:
        sample = InertialSample(
            m.timestamp,
            (m.accelerometer_x, m.accelerometer_y, m.accelerometer_z),
            (m.gyroscope_x, m.gyroscope_y, m.gyroscope_z),
        )
        with self.lock:
            self._inertial.append(sample)

    def on_quaternion(self, m) -> None:
        sample = QuaternionSample(m.timestamp, (m.w, m.x, m.y, m.z))
        with self.lock:
            self._quaternion.append(sample)

    def on_magnetometer(self, m) -> None:
        sample = MagnetometerSample(m.timestamp, (m.x, m.y, m.z))
        with self.lock:
            self._magnetometer.append(sample)

    def register(self) -> None:
        self._callback_ids = [
            self.connection.add_inertial_callback(self.on_inertial),
            self.connection.add_quaternion_callback(self.on_quaternion),
            self.connection.add_magnetometer_callback(self.on_magnetometer),
        ]

    def unregister(self) -> None:
        for callback_id in self._callback_ids:
            self.connection.remove_callback(callback_id)
        self._callback_ids = []

    def drain(self) -> ImuBatch | None:
        with self.lock:
            inertial, self._inertial = self._inertial, []
            quaternion, self._quaternion = self._quaternion, []
            magnetometer, self._magnetometer = self._magnetometer, []
        if not (inertial or quaternion or magnetometer):
            return None
        return ImuBatch(self.index, tuple(inertial), tuple(quaternion), tuple(magnetometer))


class XimuSource:
    def __init__(self, bus: ImuBus, ips=None, port: int = SENSOR_PORT, hz: float = EMIT_HZ,
                 serials=DEFAULT_SERIALS):
        self.bus = bus
        self.ips = tuple(ips) if ips else None  # None: look the serials up
        self.serials = tuple(serials)
        self.port = port
        self.period = 1.0 / hz
        self.sensors: list[SensorState] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def describe(self) -> str:
        if self.ips:
            return ", ".join(f"imu {i} = {ip}" for i, ip in enumerate(self.ips))
        return ", ".join(f"imu {i} = serial {serial}" for i, serial in enumerate(self.serials))

    def start(self) -> None:
        """Connect every sensor (blocking, with the SDK's own retries), blink it,
        register the callbacks and start draining. All or nothing: on a failed
        connection the ones already open are closed and ImuConnectError says which."""
        ximu3 = _import_ximu3()
        for index, (ip, port) in enumerate(self._endpoints(ximu3)):
            self._connect(ximu3, index, ip, port)
        for sensor in self.sensors:
            sensor.connection.send_commands(STARTUP_COMMANDS)
            sensor.register()
        self._thread = threading.Thread(target=self._loop, name="ximu-drain", daemon=True)
        self._thread.start()

    def _endpoints(self, ximu3) -> list[tuple[str, int]]:
        """(ip, port) per sensor, imu 0 first: the fixed ips, or each serial's
        announced address."""
        if self.ips:
            return [(ip, self.port) for ip in self.ips]
        print(f"Looking for x-IMU3 serial(s) {', '.join(self.serials)} "
              f"(up to {DISCOVERY_TIMEOUT_S:.0f} s) ...", flush=True)
        try:
            found = discovery.discover(self.serials, DISCOVERY_TIMEOUT_S,
                                       ximu3.NetworkAnnouncement().get_messages)
        except LookupError as error:
            raise ImuConnectError(
                f"{error}\nAre they on, and on the same network as this computer? "
                "(--imu-serial for other sensors, --imu-ip to skip the lookup, or rehearse with --imu fake)"
            ) from error
        return [(e.ip, e.port) for e in found]

    def _connect(self, ximu3, index: int, ip: str, port: int) -> None:
        print(f"Connecting to x-IMU3 {ip}:{port} as imu {index} ...", flush=True)
        connection = ximu3.Connection(ximu3.TcpConnectionConfig(ip, port))
        try:
            connection.open()
        except RuntimeError as error:
            self._close_all()
            raise ImuConnectError(
                f"could not connect to imu {index} at {ip}:{port} ({error}). Is it on, "
                "and on this network? (--imu-serial / --imu-ip; or rehearse with --imu fake)"
            ) from error
        self.sensors.append(SensorState(connection, index, ip))
        print(f"  connected: imu {index} ({ip}) -- it should be blinking", flush=True)

    def _loop(self) -> None:
        next_emit = time.perf_counter()
        while not self._stop.is_set():
            next_emit = _wait_for_tick(next_emit, self.period)
            for sensor in self.sensors:
                batch = sensor.drain()
                if batch is not None:
                    self.bus.publish(batch)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._close_all()

    def _close_all(self) -> None:
        # Callbacks off BEFORE closing: ximu-api found that closing first can fire a
        # callback into a torn-down connection ("FATAL: exception not rethrown /
        # Aborted (core dumped)").
        for sensor in self.sensors:
            sensor.unregister()
        for sensor in self.sensors:
            _quietly(sensor.connection.send_command, '{"strobe":null}')
            _quietly(sensor.connection.close)
        self.sensors = []


def _wait_for_tick(next_emit: float, period: float) -> float:
    """Sleep until the next tick; if behind, skip the missed ticks instead of
    spinning to catch up (the bug ximu-api fixed)."""
    next_emit += period
    now = time.perf_counter()
    if next_emit < now:
        return next_emit + (int((now - next_emit) // period) + 1) * period
    time.sleep(next_emit - now)
    return next_emit


def _quietly(call, *args) -> None:
    try:
        call(*args)
    except RuntimeError:
        pass


def _import_ximu3():
    try:
        import ximu3
    except ImportError as error:
        raise ImuConnectError(
            f"the x-IMU3 SDK (ximu3) isn't importable: {error}. Run `uv sync` in stretch-demo, "
            "or rehearse with --imu fake."
        ) from error
    return ximu3
