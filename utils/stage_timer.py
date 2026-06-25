import time
import threading
from typing import Dict

class StageTimer:
    """Utility to enforce per‑stage runtime limits.

    Usage:
        timer = StageTimer({"extraction": 15, "entity": 8, "claim": 15,
                           "blindspot": 15, "research": 30, "evidence": 20,
                           "report": 5})
        with timer.time_limit("extraction"):
            ...  # stage code
    """

    def __init__(self, limits: Dict[str, int]):
        """Initialize with a mapping of stage name -> max seconds."""
        self.limits = limits
        self._start_times: Dict[str, float] = {}
        self._timers: Dict[str, threading.Timer] = {}
        self._expired: Dict[str, bool] = {}

    def start(self, stage: str) -> None:
        """Record the start time for *stage* and schedule a timeout raise."""
        if stage not in self.limits:
            raise ValueError(f"Stage '{stage}' has no configured time limit")
        self._start_times[stage] = time.time()
        self._expired[stage] = False
        # schedule timer to set flag after limit seconds
        timeout = self.limits[stage]
        timer = threading.Timer(timeout, self._expire, args=(stage,))
        timer.start()
        self._timers[stage] = timer

    def _expire(self, stage: str) -> None:
        self._expired[stage] = True

    def stop(self, stage: str) -> None:
        """Cancel the timer for *stage* and clear tracking data."""
        if stage in self._timers:
            self._timers[stage].cancel()
            del self._timers[stage]
        self._start_times.pop(stage, None)
        self._expired.pop(stage, None)

    def elapsed(self, stage: str) -> float:
        """Return elapsed seconds for *stage* or 0 if not started."""
        if stage not in self._start_times:
            return 0.0
        return time.time() - self._start_times[stage]

    def check(self, stage: str) -> None:
        """Raise ``TimeoutError`` if the stage exceeded its limit."""
        if self._expired.get(stage, False):
            raise TimeoutError(f"Stage '{stage}' exceeded its time budget of {self.limits[stage]} seconds")

    # Context manager convenience
    class _StageContext:
        def __init__(self, timer: "StageTimer", stage: str):
            self.timer = timer
            self.stage = stage

        def __enter__(self):
            self.timer.start(self.stage)
            return self.timer

        def __exit__(self, exc_type, exc, tb):
            self.timer.stop(self.stage)
            # If the stage timed‑out, raise after cleanup
            if self.timer._expired.get(self.stage, False):
                raise TimeoutError(f"Stage '{self.stage}' exceeded its time budget of {self.timer.limits[self.stage]} seconds")
            return False  # propagate any other exception

    def time_limit(self, stage: str):
        """Return a context manager that enforces the budget for *stage*.

        Example:
            with stage_timer.time_limit("claim"):
                run_claim_analysis()
        """
        return self._StageContext(self, stage)
