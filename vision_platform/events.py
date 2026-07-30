from __future__ import annotations

from threading import RLock
from typing import Callable

from vision_platform.models import TaskEvent


EventHandler = Callable[[TaskEvent], None]


class EventBus:
    """Small synchronous event bus with thread-safe subscription snapshots."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        with self._lock:
            self._handlers.append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._handlers:
                    self._handlers.remove(handler)

        return unsubscribe

    def publish(self, event: TaskEvent) -> None:
        with self._lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            handler(event)
