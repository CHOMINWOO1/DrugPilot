from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator

from drugpilot.domain.models import ExecutionEvent


class EventBroker:
    """In-process fan-out for the local dashboard's server-sent events."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[ExecutionEvent | None]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def publish(self, event: ExecutionEvent) -> None:
        async with self._lock:
            subscribers = tuple(self._subscribers.get(event.run_id, ()))
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # The persisted audit ledger remains authoritative.  A slow UI
                # can reconnect and fetch the current RunSnapshot.
                pass

    async def close(self, run_id: str) -> None:
        async with self._lock:
            subscribers = tuple(self._subscribers.get(run_id, ()))
        for queue in subscribers:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def stream(self, run_id: str) -> AsyncIterator[str]:
        queue: asyncio.Queue[ExecutionEvent | None] = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers[run_id].add(queue)
        try:
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    yield "event: closed\ndata: {}\n\n"
                    return
                yield (
                    f"id: {event.event_id}\n"
                    "event: workflow\n"
                    f"data: {event.model_dump_json()}\n\n"
                )
        finally:
            async with self._lock:
                self._subscribers[run_id].discard(queue)
                if not self._subscribers[run_id]:
                    self._subscribers.pop(run_id, None)

