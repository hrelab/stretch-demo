"""Tells the visitor when a joint won't move any further, so a joint that has
stopped doesn't look like a frozen controller.

Two cases, per driven joint:
  AT LIMIT  the controller is asking for more than the joint's range (Setpoints
            clamped the last write in that direction)
  BLOCKED   the joint has lagged its command for BLOCKED_AFTER_S: it's pressed
            against the robot itself, the table, the floor... The Stretch 4's
            wrist ranges are huge (pitch -65..+245 deg), so with rate control the
            wrist usually hits something long before a limit.
Either way the fix is the same: move that joint back the other way.
"""
import numpy as np

LABELS = {"lift": "lift", "arm": "arm", "wrist_yaw": "wrist yaw",
          "wrist_pitch": "wrist pitch", "wrist_roll": "wrist roll"}
# How far behind its command a joint has to be to count as blocked. Generous
# enough for servo lag in normal motion and the Stretch 4 lift's ~4 cm sag.
BLOCKED_GAP = {"lift": 0.07, "arm": 0.05, "wrist_yaw": np.deg2rad(20.0),
               "wrist_pitch": np.deg2rad(20.0), "wrist_roll": np.deg2rad(20.0)}
BLOCKED_AFTER_S = 0.5


class JointWatch:
    def __init__(self, act: dict[str, int]):
        self.act = act
        self._lagging_since: dict[str, float] = {}

    def check(self, setpoints, data, now: float) -> list[str]:
        """'wrist pitch at its limit' / 'arm blocked' for each stuck joint."""
        notes = []
        for joint, label in LABELS.items():
            if setpoints.pushing.get(joint):
                notes.append(f"{label} at its limit")
            elif self._blocked(joint, setpoints[joint], data, now):
                notes.append(f"{label} blocked")
        return notes

    def _blocked(self, joint: str, command: float, data, now: float) -> bool:
        gap = abs(command - float(data.actuator_length[self.act[joint]]))
        if gap < BLOCKED_GAP[joint]:
            self._lagging_since.pop(joint, None)
            return False
        since = self._lagging_since.setdefault(joint, now)
        return now - since >= BLOCKED_AFTER_S

    def reset(self) -> None:
        self._lagging_since.clear()


def limits_text(notes: list[str]) -> str:
    if not notes:
        return ""
    return ", ".join(notes).upper() + " -- move it back the other way"
