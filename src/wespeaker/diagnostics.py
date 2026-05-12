"""Performance diagnostics utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """性能计时统计类."""

    _timings: dict[str, float] = field(default_factory=dict)
    _start_times: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def start(self, operation: str) -> None:
        """开始计时某个操作."""
        self._start_times[operation] = time.perf_counter()

    def end(self, operation: str) -> float:
        """结束计时某个操作，返回耗时（秒）."""
        if operation not in self._start_times:
            return 0.0
        elapsed = time.perf_counter() - self._start_times[operation]
        del self._start_times[operation]

        if operation not in self._timings:
            self._timings[operation] = 0.0
            self._counts[operation] = 0
        self._timings[operation] += elapsed
        self._counts[operation] += 1
        return elapsed

    def get_timings(self) -> dict[str, float]:
        """获取所有操作的总耗时."""
        return self._timings.copy()

    def get_summary(self) -> dict:
        """获取性能统计摘要."""
        total_time = sum(self._timings.values())
        return {
            "total_operations": len(self._timings),
            "total_time": total_time,
            "operations": {
                op: {
                    "total_time": self._timings[op],
                    "count": self._counts[op],
                    "avg_time": self._timings[op] / self._counts[op],
                }
                for op in self._timings
            },
        }
