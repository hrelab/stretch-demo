"""High scores with initials, kept in one JSON file so they survive restarts
during the event.

Every finished round is kept, tagged with its controller; the board shown is
per controller (they differ a lot in difficulty, so mixing them isn't fair).
Ranking: higher score first, then the faster time.
"""
import json
import time
from pathlib import Path

SHOWN = 5


def normalize_initials(text: str) -> str | None:
    """'a.b c' -> 'ABC': letters and digits, upper case, at most 3. None if empty."""
    kept = "".join(ch for ch in text.upper() if ch.isalnum())[:3]
    return kept or None


def _rank_key(entry: dict) -> tuple:
    return (-entry["score"], entry["time_s"])


class Leaderboard:
    def __init__(self, path: Path, controller: str):
        self.path = Path(path)
        self.controller = controller
        self.latest: dict | None = None  # the entry added last, to highlight

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            # Never lose the event's scores silently: keep the bad file aside.
            aside = self.path.with_suffix(f".unreadable-{int(time.time())}.json")
            self.path.rename(aside)
            print(f"Leaderboard {self.path} was unreadable ({error}); moved to {aside}, starting fresh.")
            return []

    def entries(self) -> list[dict]:
        """This controller's entries, best first."""
        mine = [e for e in self._load() if e.get("controller") == self.controller]
        return sorted(mine, key=_rank_key)

    def add(self, initials: str, score: int, time_s: float, **extra) -> int:
        """Save a round; returns its rank (1 = best) on this controller's board."""
        entry = {"initials": initials, "score": score, "time_s": round(time_s, 1),
                 "controller": self.controller, "date": time.strftime("%Y-%m-%d %H:%M"), **extra}
        everything = self._load() + [entry]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(everything, indent=1) + "\n")
        tmp.replace(self.path)  # atomic: a crash mid-write can't corrupt the board
        self.latest = entry
        return self.entries().index(entry) + 1

    def rows(self) -> list[tuple[str, str]]:
        """The top SHOWN as viewer rows, the latest entry marked."""
        rows = [("TOP SCORES", self.controller)]
        for rank, entry in enumerate(self.entries()[:SHOWN], start=1):
            mark = "  <-- new" if entry == self.latest else ""
            rows.append((f"{rank}. {entry['initials']}", f"{entry['score']} pts  {clock_text(entry['time_s'])}{mark}"))
        if len(rows) == 1:
            rows.append(("", "no scores yet"))
        return rows


def clock_text(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"
