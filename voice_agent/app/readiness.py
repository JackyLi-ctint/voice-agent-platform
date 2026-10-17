"""In-memory liveness and readiness state with critical/optional checks."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CheckResult:
    ok: bool
    critical: bool
    detail: str


class ReadinessState:
    """Thread-safe process readiness snapshot updated by application lifespan."""

    def __init__(
        self,
        service_name: str,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._service_name = service_name
        self._clock = clock
        self._lock = threading.RLock()
        self._startup_complete = False
        self._stopping = False
        self._started_at: float | None = None
        self._checks: dict[str, CheckResult] = {}

    def begin_startup(self) -> None:
        with self._lock:
            self._startup_complete = False
            self._stopping = False
            self._started_at = self._clock()
            self._checks.clear()

    def set_check(
        self,
        name: str,
        *,
        ok: bool,
        critical: bool,
        detail: str = "",
    ) -> None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("readiness check name must not be empty")
        with self._lock:
            self._checks[normalized] = CheckResult(
                ok=bool(ok), critical=bool(critical), detail=str(detail)
            )

    def mark_started(self) -> None:
        with self._lock:
            self._startup_complete = True
            self._stopping = False
            if self._started_at is None:
                self._started_at = self._clock()

    def mark_stopping(self) -> None:
        with self._lock:
            self._stopping = True

    def liveness(self) -> dict[str, object]:
        with self._lock:
            return {
                "ok": True,
                "service": self._service_name,
                "status": "stopping" if self._stopping else "alive",
            }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            checks = {name: asdict(result) for name, result in sorted(self._checks.items())}
            critical_ok = all(
                result.ok for result in self._checks.values() if result.critical
            )
            optional_degraded = any(
                not result.ok for result in self._checks.values() if not result.critical
            )
            ready = self._startup_complete and not self._stopping and critical_ok
            if self._stopping:
                status = "stopping"
            elif not ready:
                status = "not_ready"
            elif optional_degraded:
                status = "degraded"
            else:
                status = "ready"
            return {
                "ok": ready,
                "service": self._service_name,
                "status": status,
                "startup_complete": self._startup_complete,
                "checks": checks,
            }
