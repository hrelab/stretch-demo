"""CLI wiring, the operator console, and the complexity rule."""
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stretch_demo.cli import CONTROLLERS, TASKS, make_source, parse_args
from stretch_demo.console import Console
from stretch_demo.imu.bus import ImuBus
from stretch_demo.imu.fake_source import FakeSource
from stretch_demo.imu.ximu_source import XimuSource

ROOT = Path(__file__).resolve().parents[1]
MAX_COMPLEXITY = 10


@pytest.mark.parametrize("controller", sorted(CONTROLLERS))
def test_every_controller_builds_for_every_task(controller):
    for env, task_cls in TASKS.items():
        args = parse_args(["--env", env, "--controller", controller, "--imu", "fake"])
        built = CONTROLLERS[controller](ImuBus(2), task_cls(), args)
        assert built.gripper_enabled == (env != "feeding")


def test_head_controller_takes_mobility_profile_and_shoulder():
    args = parse_args(["--env", "door", "--controller", "head",
                       "--mobility", str(ROOT / "profiles" / "p00.json"), "--shoulder", "left"])
    built = CONTROLLERS["head"](ImuBus(2), TASKS["door"](), args)
    assert built.settings.shoulder_side == "left"


def test_imu_source_choice():
    fake = parse_args(["--env", "door", "--controller", "head", "--imu", "fake"])
    real = parse_args(["--env", "door", "--controller", "head", "--imu-ip", "10.0.0.1", "10.0.0.2"])
    assert isinstance(make_source(fake, ImuBus(2)), FakeSource)
    source = make_source(real, ImuBus(2))
    assert isinstance(source, XimuSource) and source.ips == ("10.0.0.1", "10.0.0.2")


def test_console_queues_commands_and_quits(capsys):
    console = Console(io.StringIO("c\n  R \nq\nzzz\n"))
    for line in console.stream:
        console.push(line)
    assert console.take() == ["c", "r"] and console.quit_requested
    assert "Commands" in capsys.readouterr().out  # unknown input prints the help


def _complexipy() -> str:
    found = shutil.which("complexipy") or str(Path(sys.executable).parent / "complexipy")
    if not Path(found).exists():
        pytest.skip("complexipy not installed (it's in the dev group: uv sync)")
    return found


def registered_exceptions() -> set[str]:
    """Functions allowed above the limit: `path::function` in COMPLEXITY.md's table."""
    text = (ROOT / "COMPLEXITY.md").read_text()
    return set(re.findall(r"^\|\s*`([^`]+::[^`]+)`", text, flags=re.MULTILINE))


def test_no_function_exceeds_complexity_10_unless_registered(tmp_path):
    out = tmp_path / "complexity.json"
    subprocess.run([_complexipy(), "src", "tests", "--output-format", "json", "--output", str(out),
                    "-q", "--max-complexity-allowed", str(MAX_COMPLEXITY)], cwd=ROOT, check=False)
    rows = json.loads(out.read_text())
    over = {f"{r['path']}::{r['function_name']}": r["complexity"] for r in rows if r["complexity"] > MAX_COMPLEXITY}
    unregistered = {name: score for name, score in over.items() if name not in registered_exceptions()}
    assert not unregistered, f"over {MAX_COMPLEXITY}, rewrite or justify in COMPLEXITY.md: {unregistered}"
