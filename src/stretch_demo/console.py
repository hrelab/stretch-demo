"""Operator commands typed in the demo's one terminal (letter + Enter).

Terminal, not viewer keys: MuJoCo's viewer already binds nearly every letter to a
rendering toggle, and the operator is at the terminal anyway.
"""
import queue
import sys
import threading

HELP = """Commands (type the letter, then Enter):
  c  recalibrate neutral (visitor holds still ~2 s)
  r  reset the task objects / start the trial over
  n  next visitor: robot and task reset, scores cleared, recalibrate
  q  quit (or close the viewer window)
  h  this help"""

COMMANDS = {"c", "r", "n", "q", "h"}


class Console:
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdin
        self._queue: queue.Queue[str] = queue.Queue()
        self.quit_requested = False

    def start(self) -> "Console":
        if self.stream is not None and not self.stream.closed:
            threading.Thread(target=self._read, name="console", daemon=True).start()
        return self

    def _read(self) -> None:
        for line in self.stream:
            self.push(line)

    def push(self, line: str) -> None:
        command = line.strip().lower()[:1]
        if command == "q":
            self.quit_requested = True
        elif command == "h" or (command and command not in COMMANDS):
            print(HELP)
        elif command:
            self._queue.put(command)

    def take(self) -> list[str]:
        out = []
        while not self._queue.empty():
            out.append(self._queue.get_nowait())
        return out
