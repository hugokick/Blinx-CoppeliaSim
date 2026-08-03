from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = 1
ALLOWED_COMMANDS = frozenset(
    {
        "camera.capture",
        "camera.profile.apply",
        "camera.profile.get",
        "camera.profile.reset",
        "context.log",
        "context.sleep",
        "context.checkpoint",
        "experiment.info",
        "robot.home",
        "robot.move_world",
        "robot.pose",
        "tool.on",
        "tool.off",
        "vision2d.analyze",
        "vision2d.code_routes",
        "vision2d.ocr_sorting",
        "vision2d.ocr_sort_entry",
        "vision2d.template_match",
    }
)


class RunState(str, Enum):
    EMPTY = "EMPTY"
    LOADED = "LOADED"
    VALIDATED = "VALIDATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    PASSED = "PASSED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RESETTING = "RESETTING"


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must not be empty")
    return value


@dataclass(frozen=True)
class CommandMessage:
    command_id: str
    name: str
    args: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.command_id, "command_id")
        if self.name not in ALLOWED_COMMANDS:
            raise ValueError(f"COMMAND_NOT_ALLOWED: {self.name}")
        if not isinstance(self.args, Mapping):
            raise ValueError("args must be a mapping")
        object.__setattr__(self, "args", dict(self.args))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "command",
            "command_id": self.command_id,
            "name": self.name,
            "args": dict(self.args),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommandMessage":
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")
        version = payload.get("schema_version")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise ValueError("PROTOCOL_VERSION_UNSUPPORTED")
        if payload.get("kind") != "command":
            raise ValueError("PROTOCOL_KIND_INVALID")
        return cls(
            command_id=_require_text(payload.get("command_id"), "command_id"),
            name=_require_text(payload.get("name"), "name"),
            args=payload.get("args", {}),
        )


@dataclass(frozen=True)
class ResponseMessage:
    command_id: str
    status: str
    value: Any
    error: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        _require_text(self.command_id, "command_id")
        if self.status not in {"PASS", "FAIL", "CANCELLED"}:
            raise ValueError(f"Invalid response status: {self.status}")
        if self.status == "PASS" and self.error is not None:
            raise ValueError("PASS response must not contain error")
        if self.status != "PASS" and self.error is None:
            raise ValueError("Non-PASS response must contain error")
        if self.error is not None:
            if not isinstance(self.error, Mapping):
                raise ValueError("error must be a mapping")
            object.__setattr__(self, "error", dict(self.error))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "result",
            "command_id": self.command_id,
            "status": self.status,
            "value": self.value,
            "error": dict(self.error) if self.error is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResponseMessage":
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")
        version = payload.get("schema_version")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise ValueError("PROTOCOL_VERSION_UNSUPPORTED")
        if payload.get("kind") != "result":
            raise ValueError("PROTOCOL_KIND_INVALID")
        error = payload.get("error")
        return cls(
            command_id=_require_text(payload.get("command_id"), "command_id"),
            status=_require_text(payload.get("status"), "status"),
            value=payload.get("value"),
            error=error,
        )
