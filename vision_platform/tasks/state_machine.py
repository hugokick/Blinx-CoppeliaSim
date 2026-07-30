from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from vision_platform.events import EventBus
from vision_platform.models import TaskEvent


_ALLOWED: dict[str | None, frozenset[str]] = {
    None: frozenset({"IDLE"}),
    "IDLE": frozenset({"ACQUIRE", "SAFE_STOP"}),
    "ACQUIRE": frozenset({"DETECT", "SAFE_STOP"}),
    "DETECT": frozenset({"TRANSFORM", "SAFE_STOP"}),
    "TRANSFORM": frozenset({"APPROACH", "SAFE_STOP"}),
    "APPROACH": frozenset({"DESCEND", "SAFE_STOP"}),
    "DESCEND": frozenset({"ATTACH", "SAFE_STOP"}),
    "ATTACH": frozenset({"LIFT", "SAFE_STOP"}),
    "LIFT": frozenset({"TRANSFER", "SAFE_STOP"}),
    "TRANSFER": frozenset({"RELEASE", "SAFE_STOP"}),
    "RELEASE": frozenset({"VERIFY", "SAFE_STOP"}),
    "VERIFY": frozenset({"ACQUIRE", "COMPLETE", "SAFE_STOP"}),
    "COMPLETE": frozenset(),
    "SAFE_STOP": frozenset(),
}


class TaskStateMachine:
    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        clock: Callable[[], float] = time.time,
        step_waiter: Callable[[TaskEvent], None] | None = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.clock = clock
        self.step_waiter = step_waiter
        self.state: str | None = None
        self.events: list[TaskEvent] = []

    def transition(
        self,
        state: str,
        message: str,
        *,
        data: Mapping[str, Any] | None = None,
        error_code: str | None = None,
    ) -> TaskEvent:
        if state != "SAFE_STOP" and state not in _ALLOWED.get(
            self.state, frozenset()
        ):
            raise RuntimeError(
                f"Invalid task transition: {self.state!r} -> {state!r}"
            )
        event = TaskEvent(
            state=state,
            message=message,
            timestamp_s=float(self.clock()),
            data=dict(data or {}),
            error_code=error_code,
        )
        self.state = state
        self.events.append(event)
        self.event_bus.publish(event)
        if self.step_waiter is not None and state not in {"COMPLETE", "SAFE_STOP"}:
            self.step_waiter(event)
        return event
