from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class AttachmentEvidence:
    attached: bool
    object_handle: int | None = None
    distance_m: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolService(Protocol):
    def on(self) -> AttachmentEvidence:
        ...

    def off(self) -> None:
        ...

    def is_attached(self) -> bool:
        ...


class BackendPumpTool:
    """Idempotent pump commands for the real BLX-compatible backend."""

    def __init__(
        self,
        backend: Any,
        *,
        attachment_check: Callable[[], bool] | None = None,
    ) -> None:
        self._backend = backend
        self._attachment_check = attachment_check
        self._enabled = False

    def on(self) -> AttachmentEvidence:
        if not self._enabled:
            self._backend.pump_on()
            self._enabled = True
        return AttachmentEvidence(
            attached=self.is_attached(),
            metadata={
                "verification": (
                    "sensor"
                    if self._attachment_check is not None
                    else "command_acknowledged"
                )
            },
        )

    def off(self) -> None:
        if not self._enabled:
            return
        self._backend.pump_off()
        self._enabled = False

    def is_attached(self) -> bool:
        if not self._enabled:
            return False
        if self._attachment_check is None:
            return True
        return bool(self._attachment_check())
