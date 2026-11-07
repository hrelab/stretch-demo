"""Operator input typed in the demo's one terminal (a line + Enter).

Terminal, not viewer keys: MuJoCo's viewer already binds nearly every letter to a
rendering toggle, and the operator is at the terminal anyway.

Lines are queued whole; the harness decides what a line means: normally a
command letter (COMMANDS), but after a feeding round it's the visitor's initials,
so typing "Q" or "N" as initials doesn't quit or reset.
"""
import queue
import sys
import threading

HELP = """Commands (type the letter, then Enter):
  c  recalibrate neutral (visitor holds still ~2 s)
  r  reset the task / play another round
  n  next visitor: robot and task reset, scores cleared, recalibrate
  g  open / close the gripper (operator override; ee_mode2: also space or the clicker)
  q  quit (or close the viewer window)
  h  this help
After a feeding round, the next line typed is the visitor's initials."""

COMMANDS = {"c", "r", "n", "g", "q", "h"}


def command_of(line: str) -> str | None:
    """'  R ' -> 'r'; None for a blank or unknown line."""
    letter = line.strip().lower()[:1]
    return letter if letter in COMMANDS else None


class Console:
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdin
        self._queue: queue.Queue[str] = queue.Queue()

    def start(self) -> "Console":
        if self.stream is not None and not self.stream.closed:
            threading.Thread(target=self._read, name="console", daemon=True).start()
        return self

    def _read(self) -> None:
        for line in self.stream:
            self._queue.put(line.rstrip("\n"))

    def take(self) -> list[str]:
        out = []
        while not self._queue.empty():
            out.append(self._queue.get_nowait())
        return out
