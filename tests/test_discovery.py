"""Finding the x-IMU3s by serial number, against fake announcements and a fake SDK."""
from types import SimpleNamespace

import pytest

from stretch_demo.cli import make_source, parse_args
from stretch_demo.imu import discovery
from stretch_demo.imu.bus import ImuBus
from stretch_demo.imu.ximu_source import DEFAULT_SERIALS, ImuConnectError, XimuSource


def announcement(serial, ip, port=7000):
    return SimpleNamespace(serial_number=serial, ip_address=ip, tcp_port=port,
                           device_name="x-IMU3", rssi=80, battery=90)


def test_order_follows_serials_not_announcements():
    seen = discovery.latest_by_serial([announcement("B", "172.20.10.2"), announcement("A", "172.20.10.13")])
    assert [e.ip for e in discovery.resolve(["A", "B"], seen)] == ["172.20.10.13", "172.20.10.2"]


def test_latest_announcement_wins():
    seen = discovery.latest_by_serial([announcement("A", "192.168.0.161"), announcement("A", "172.20.10.13")])
    assert seen["A"].ip_address == "172.20.10.13"


def test_missing_serial_is_named_with_everything_seen():
    with pytest.raises(LookupError, match="not found on the network: C") as raised:
        discovery.resolve(["A", "C"], {"B": announcement("B", "172.20.10.2"), "A": announcement("A", "1.1.1.1")})
    assert "172.20.10.2" in str(raised.value)


def test_discover_waits_for_every_serial():
    rounds = iter([[], [announcement("A", "10.0.0.1")], [announcement("B", "10.0.0.2", port=7001)]])
    found = discovery.discover(["A", "B"], 5.0, lambda: next(rounds, []), poll_s=0)
    assert [(e.ip, e.port) for e in found] == [("10.0.0.1", 7000), ("10.0.0.2", 7001)]


def test_discover_gives_up_after_timeout():
    with pytest.raises(LookupError):
        discovery.discover(["A"], 0.05, lambda: [], poll_s=0.01)


class FakeConnection:
    opened: list = []

    def __init__(self, config):
        self.config = config

    def open(self):
        FakeConnection.opened.append(self.config)

    def __getattr__(self, name):  # callbacks, commands, close: no-ops
        return lambda *args, **kwargs: 0


def fake_sdk(messages):
    return SimpleNamespace(
        Connection=FakeConnection,
        TcpConnectionConfig=lambda ip, port: (ip, port),
        NetworkAnnouncement=lambda: SimpleNamespace(get_messages=lambda: messages),
    )


def test_source_connects_to_each_serials_announced_address(monkeypatch):
    sdk = fake_sdk([announcement("S1", "172.20.10.2"), announcement("S0", "172.20.10.13", port=7001)])
    monkeypatch.setattr("stretch_demo.imu.ximu_source._import_ximu3", lambda: sdk)
    FakeConnection.opened = []
    source = XimuSource(ImuBus(2), serials=("S0", "S1"))
    source.start()
    source.stop()
    assert FakeConnection.opened == [("172.20.10.13", 7001), ("172.20.10.2", 7000)]


def test_source_reports_missing_serial(monkeypatch):
    monkeypatch.setattr("stretch_demo.imu.ximu_source._import_ximu3", lambda: fake_sdk([]))
    monkeypatch.setattr("stretch_demo.imu.ximu_source.DISCOVERY_TIMEOUT_S", 0.05)
    with pytest.raises(ImuConnectError, match="not found on the network: S0"):
        XimuSource(ImuBus(2), serials=("S0", "S1")).start()


def test_cli_looks_up_default_serials_unless_given_ips():
    args = parse_args(["--env", "door", "--controller", "head"])
    source = make_source(args, ImuBus(2))
    assert source.ips is None and source.serials == DEFAULT_SERIALS
    assert "serial" in source.describe()
    args = parse_args(["--env", "door", "--controller", "head", "--imu-serial", "X", "Y"])
    assert make_source(args, ImuBus(2)).serials == ("X", "Y")


def test_cli_serial_and_ip_are_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--env", "door", "--controller", "head", "--imu-ip", "a", "b", "--imu-serial", "X", "Y"])
