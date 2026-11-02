"""Thread-safe hand-off from the IMU threads to the sim loop.

Replaces the old Redis keys (`SET imu:N ... PX 500` by port-listener, `GET imu:N`
by the controllers). Two differences, both on purpose:

- It keeps the last few batches, not just the newest, and `latest()` merges them.
  With Redis, a batch overwritten before the sim read it was lost, and with it any
  gyro peak in it (that's what the flick/shrug detectors look for). The trackers
  de-duplicate samples by device timestamp, so handing them some already-seen
  samples again is harmless.
- Staleness is the same 0.5 s as IMU_KEY_TTL_MS: a silent IMU reads as None, which
  makes every controller hold still instead of acting on old data.
"""
import threading
import time
from collections import deque

from stretch_demo.imu.types import ImuBatch

TTL_S = 0.5
KEEP_BATCHES = 8  # ~80 ms at 100 Hz: covers a slow viewer frame


class ImuBus:
    def __init__(self, count: int, ttl_s: float = TTL_S, clock=time.monotonic):
        self.count = count
        self.ttl_s = ttl_s
        self.clock = clock
        self._lock = threading.Lock()
        self._recent = [deque(maxlen=KEEP_BATCHES) for _ in range(count)]
        self._last_seen = [float("-inf")] * count
        self._inertial_counts = [0] * count
        self._rate_start = clock()
        self._rates = [0.0] * count

    def publish(self, batch: ImuBatch) -> None:
        if not 0 <= batch.index < self.count:
            return
        now = self.clock()
        with self._lock:
            self._recent[batch.index].append(batch)
            self._last_seen[batch.index] = now
            self._inertial_counts[batch.index] += len(batch.inertial)

    def latest(self, index: int) -> ImuBatch | None:
        """The recent batches of IMU `index` merged into one, or None if it has
        been silent for longer than the TTL."""
        with self._lock:
            if self.clock() - self._last_seen[index] > self.ttl_s:
                return None
            batches = tuple(self._recent[index])
        return ImuBatch(
            index=index,
            inertial=tuple(s for b in batches for s in b.inertial),
            quaternion=tuple(s for b in batches for s in b.quaternion),
            magnetometer=tuple(s for b in batches for s in b.magnetometer),
        )

    def rates(self) -> list[float]:
        """Inertial samples per second per IMU (~400 Hz on a healthy x-IMU3),
        refreshed at most once a second."""
        now = self.clock()
        with self._lock:
            elapsed = now - self._rate_start
            if elapsed >= 1.0:
                self._rates = [n / elapsed for n in self._inertial_counts]
                self._inertial_counts = [0] * self.count
                self._rate_start = now
            return list(self._rates)

    def alive(self) -> list[bool]:
        now = self.clock()
        with self._lock:
            return [now - seen <= self.ttl_s for seen in self._last_seen]
