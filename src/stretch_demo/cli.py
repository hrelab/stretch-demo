"""`uv run demo`: the whole showcase in one terminal.

  x-IMU3s --(ximu3 SDK threads)--> ImuBus --> controller --> MuJoCo viewer
                                                       task scoring --> runs/*.csv

replacing ximu-api + port-listener + Redis + `uv run sim` (four processes).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

from stretch_demo.console import HELP, Console
from stretch_demo.imu.bus import ImuBus
from stretch_demo.imu.fake_source import FakeSource
from stretch_demo.imu.ximu_source import DEFAULT_SERIALS, ImuConnectError, XimuSource
from stretch_demo.sim.harness import SimHarness
from stretch_demo.tasks.door import DoorTask
from stretch_demo.tasks.feeding import FeedingTask
from stretch_demo.tasks.leaderboard import Leaderboard
from stretch_demo.tasks.log import RUNS_DIR, TrialLog
from stretch_demo.tasks.pickup import PickupTask
from stretch_demo.tracking.mobility import MobilitySettings

TASKS = {task.name: task for task in (FeedingTask, PickupTask, DoorTask)}


def _hands_vel(bus, task, args):
    from stretch_demo.controllers.hands_vel import HandsVelController
    return HandsVelController(bus, task.gripper_enabled)


def _hands_v2(bus, task, args):
    from stretch_demo.controllers.hands_v2 import HandsV2Controller
    return HandsV2Controller(bus, task.gripper_enabled)


def _head(bus, task, args):
    from stretch_demo.controllers.head import HeadController
    return HeadController(bus, task.gripper_enabled, mobility(args))


def _ee_mode2(bus, task, args):
    from stretch_demo.controllers.ee.mode2 import EeMode2Controller
    from stretch_demo.controllers.ee.settings import PostwristSettings
    return EeMode2Controller(bus, task.gripper_enabled, PostwristSettings.load(args.ee_profile))


CONTROLLERS = {"hands_vel": _hands_vel, "hands_v2": _hands_v2, "head": _head, "ee_mode2": _ee_mode2}


def mobility(args) -> MobilitySettings:
    settings = MobilitySettings.load(args.mobility) if args.mobility else MobilitySettings()
    if args.shoulder:
        settings.update({"shoulder_side": args.shoulder})
    return settings


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="demo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=sorted(TASKS), help="which task")
    parser.add_argument("--controller", required=True, choices=sorted(CONTROLLERS),
                        help="hands_vel / hands_v2 / ee_mode2: IMUs on both hands; head: head + shoulder")
    parser.add_argument("--imu", default="ximu", choices=("ximu", "fake"),
                        help="ximu: the real x-IMU3s; fake: two still synthetic IMUs, to rehearse (default: %(default)s)")
    sensors = parser.add_mutually_exclusive_group()
    sensors.add_argument("--imu-serial", nargs=2, metavar=("S0", "S1"), default=list(DEFAULT_SERIALS),
                         help="x-IMU3 serial numbers, imu 0 then imu 1, found on whatever network they're on "
                         "(default: %(default)s). hands: left/hand 1 first; head: head first, shoulder second")
    sensors.add_argument("--imu-ip", nargs=2, metavar=("IP0", "IP1"),
                         help="connect to these addresses instead of looking the serials up, imu 0 first")
    parser.add_argument("--who", default="", help="participant/visitor label written into the CSV")
    parser.add_argument("--leaderboard", default=str(RUNS_DIR / "leaderboard.json"), metavar="PATH",
                        help="high-score file for the feeding rounds (default: %(default)s)")
    head = parser.add_argument_group("--controller head")
    head.add_argument("--mobility", metavar="PATH", help="per-user ranges JSON, e.g. profiles/p00.json")
    head.add_argument("--shoulder", choices=("left", "right"), help="shoulder the IMU is on (default: right)")
    ee = parser.add_argument_group("--controller ee_mode2")
    ee.add_argument("--ee-profile", metavar="PATH",
                    help="mujoco-sim end-effector profile JSON, e.g. profiles/imu_ee_mode2_faster.json")
    return parser.parse_args(argv)


def make_source(args, bus):
    if args.imu == "fake":
        return FakeSource(bus)
    return XimuSource(bus, args.imu_ip, serials=args.imu_serial)


def main(argv=None) -> None:
    _relaunch_with_mjpython_on_macos()
    args = parse_args(argv)
    task = TASKS[args.env]()
    task.leaderboard = Leaderboard(args.leaderboard, args.controller)
    bus = ImuBus(2)
    source = make_source(args, bus)
    print(f"stretch-demo: {task.title} on the Stretch 4 with {args.controller} ({source.describe()})")
    try:
        source.start()
    except ImuConnectError as error:
        sys.exit(f"IMU setup failed: {error}")
    log = TrialLog(task.name, args.controller, args.who)
    try:
        harness = SimHarness(task)
        controller = CONTROLLERS[args.controller](bus, task, args)
        print(HELP)
        harness.run(controller, Console().start(), bus, log)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        print(log.summary())


def _relaunch_with_mjpython_on_macos() -> None:
    """The MuJoCo viewer must run on the macOS main UI thread (as in mujoco-sim)."""
    if sys.platform != "darwin" or os.environ.get("MJPYTHON_BIN"):
        return
    mjpython = shutil.which("mjpython") or str(Path(sys.prefix) / "bin" / "mjpython")
    os.execv(mjpython, [mjpython, "-m", "stretch_demo.cli", *sys.argv[1:]])


if __name__ == "__main__":
    main()
