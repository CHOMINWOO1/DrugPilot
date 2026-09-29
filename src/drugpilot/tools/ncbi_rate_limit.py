"""Process-wide admission control for NCBI E-utilities requests."""

from __future__ import annotations

from threading import Lock
from time import monotonic, sleep
from typing import Callable


class NCBIRateLimiter:
    """Serialize request admission at NCBI's key-aware etiquette interval."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self._clock = clock
        self._sleeper = sleeper
        self._lock = Lock()
        self._last_admitted_at: float | None = None

    def acquire(self, *, api_key_configured: bool) -> None:
        minimum_interval = 0.11 if api_key_configured else 0.34
        with self._lock:
            now = self._clock()
            if self._last_admitted_at is not None:
                remaining = minimum_interval - (now - self._last_admitted_at)
                if remaining > 0:
                    self._sleeper(remaining)
            self._last_admitted_at = self._clock()


_PROCESS_NCBI_RATE_LIMITER = NCBIRateLimiter()


def get_process_ncbi_rate_limiter() -> NCBIRateLimiter:
    return _PROCESS_NCBI_RATE_LIMITER


__all__ = ["NCBIRateLimiter", "get_process_ncbi_rate_limiter"]
