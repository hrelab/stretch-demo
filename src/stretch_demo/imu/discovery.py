"""Find x-IMU3s by serial number instead of by IP. A port of
ximu-api/src/ximu_api/discovery.py.

Every x-IMU3 on Wi-Fi broadcasts a network announcement about once a second
(device name, serial number, IP, TCP port, ...) -- what the x-IMU3 GUI's device
search listens to. Looking sensors up by serial means the IP can be anything
(lab network, phone hotspot, travel router; DHCP or static), and imu 0/imu 1
can't get swapped, because each serial is one physical sensor.
"""
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Endpoint:
    serial: str
    ip: str
    port: int


def latest_by_serial(messages: Iterable) -> dict:
    """Last announcement per serial: a sensor that rejoined may be on a new IP."""
    return {m.serial_number: m for m in messages}


def describe(messages: Iterable) -> str:
    rows = [
        f"  {m.serial_number:<12} {m.ip_address}:{m.tcp_port:<6} {m.device_name!r}"
        f"  (signal {m.rssi}%, battery {m.battery}%)"
        for m in sorted(messages, key=lambda m: m.serial_number)
    ]
    return "seen:\n" + "\n".join(rows) if rows else "no x-IMU3 announcements seen at all"


def resolve(serials: Sequence[str], seen: dict) -> list[Endpoint]:
    """Endpoints in the order of `serials` (that order is the imu index).
    Raises LookupError naming what's missing and listing everything seen."""
    missing = [s for s in serials if s not in seen]
    if missing:
        raise LookupError(f"not found on the network: {', '.join(missing)}\n{describe(seen.values())}")
    return [Endpoint(s, seen[s].ip_address, seen[s].tcp_port) for s in serials]


def discover(serials: Sequence[str], timeout_s: float, get_messages: Callable[[], list],
             poll_s: float = 0.25) -> list[Endpoint]:
    """Wait up to `timeout_s` for every serial to announce itself."""
    seen: dict = {}
    deadline = time.monotonic() + timeout_s
    while True:
        seen.update(latest_by_serial(get_messages()))
        if all(s in seen for s in serials) or time.monotonic() >= deadline:
            return resolve(serials, seen)
        time.sleep(poll_s)
