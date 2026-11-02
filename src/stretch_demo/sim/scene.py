"""Build a MuJoCo model from a task scene + the Stretch 4.

Scenes `<include file="stretch.xml"/>`; the robot's stretch.xml and meshes are
handed to MuJoCo from memory (as in mujoco-sim's harness), along with the
scene's robosuite textures. Optionally, a tool (a file holding one <body>, e.g.
scenes/utensil.xml) is mounted on the grasp body: a child body is held rigidly,
with no weld constraint to drift or fight the gripper.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from stretch_demo.sim.robot import GRASP_BODY, MODEL_DIR


def build_model(scene_path: Path, tool_path: Path | None = None) -> mujoco.MjModel:
    scene_path = Path(scene_path)
    assets = _robot_assets(tool_path)
    assets.update(_scene_assets(scene_path.parent, assets))
    return mujoco.MjModel.from_xml_string(scene_path.read_text(), assets=assets)


def _robot_assets(tool_path: Path | None) -> dict[str, bytes]:
    assets = {"stretch.xml": _robot_xml(tool_path)}
    seen = {"stretch.xml"}
    for path in sorted((MODEL_DIR / "assets").iterdir()):
        key = f"assets/{path.name}"
        if path.is_file() and key.lower() not in seen:  # some mesh names differ only by case
            assets[key] = path.read_bytes()
            seen.add(key.lower())
    return assets


def _scene_assets(scene_dir: Path, existing: dict) -> dict[str, bytes]:
    root = scene_dir / "robosuite_assets"
    seen = {key.lower() for key in existing}
    found = {}
    for path in sorted(root.rglob("*")) if root.is_dir() else ():
        key = path.relative_to(scene_dir).as_posix()
        if path.is_file() and key.lower() not in seen:
            found[key] = path.read_bytes()
    return found


def _robot_xml(tool_path: Path | None) -> bytes:
    xml = (MODEL_DIR / "stretch.xml").read_bytes()
    if tool_path is None:
        return xml
    root = ET.fromstring(xml)
    _find_body(root, GRASP_BODY).append(ET.fromstring(Path(tool_path).read_text()))
    return ET.tostring(root, encoding="utf-8")


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"no <body name={name!r}> found")
