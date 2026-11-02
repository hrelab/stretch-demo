"""One CSV row per finished trial, written as it happens (nothing lost on a crash)."""
import csv
import time
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"


class TrialLog:
    def __init__(self, env: str, controller: str, who: str, runs_dir: Path = RUNS_DIR):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = Path(runs_dir) / f"{env}_{controller}_{stamp}.csv"
        self.fixed = {"env": env, "controller": controller, "who": who}
        self.visitor = 1
        self.rows: list[dict] = []

    def next_visitor(self) -> None:
        self.visitor += 1

    def write(self, trials: list[dict]) -> None:
        for trial in trials:
            row = {"time": time.strftime("%H:%M:%S"), **self.fixed, "visitor": self.visitor, **trial}
            self._append(row)
            self.rows.append(row)
            print("Trial: " + ", ".join(f"{k}={_fmt(v)}" for k, v in trial.items()))

    def _append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(row)

    def summary(self) -> str:
        if not self.rows:
            return "No trials recorded."
        visitors = sorted({row["visitor"] for row in self.rows})
        lines = [f"{len(self.rows)} trials from {len(visitors)} visitor(s), saved to {self.path}"]
        for visitor in visitors:
            rows = [r for r in self.rows if r["visitor"] == visitor]
            ok = sum(r["result"] == "success" for r in rows)
            lines.append(f"  visitor {visitor}: {ok}/{len(rows)} successful")
        return "\n".join(lines)


def _fmt(value) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)
