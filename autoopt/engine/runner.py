"""
autoopt.engine.runner
=====================
High-precision execution benchmark runner with statistical outlier rejection.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple


class AutoOptRunner:
    """Runs compiled binaries with precise wall-clock timing and outlier filtering."""

    def __init__(self, runs: int = 5, warmup: bool = True, timeout: int = 30):
        self.runs = max(1, runs)
        self.warmup = warmup
        self.timeout = timeout

    def benchmark(self, binary_path: Path) -> Tuple[Optional[float], List[float], str]:
        """
        Executes binary multiple times.
        Returns: (median_time_seconds, all_times_seconds, stdout_sample)
        """
        if not binary_path.exists():
            return None, [], "Binary not found"

        cmd = [str(binary_path)]

        # Warmup run to prime OS file buffers and CPU caches
        if self.warmup:
            try:
                subprocess.run(cmd, capture_output=True, timeout=self.timeout)
            except Exception:
                pass

        times: List[float] = []
        last_stdout = ""

        for _ in range(self.runs):
            try:
                t0 = time.perf_counter()
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
                t1 = time.perf_counter()

                if res.returncode == 0:
                    times.append(t1 - t0)
                    last_stdout = res.stdout
                else:
                    return None, times, f"Runtime error code {res.returncode}: {res.stderr}"
            except subprocess.TimeoutExpired:
                return None, times, f"Execution timed out (> {self.timeout}s)"
            except Exception as exc:
                return None, times, f"Execution exception: {exc}"

        if not times:
            return None, [], "Zero successful execution runs"

        times.sort()
        median_time = times[len(times) // 2]
        return median_time, times, last_stdout
