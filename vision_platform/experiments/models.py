from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(nested) for nested in value)
    return value


def _freeze_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return _freeze(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw(nested) for nested in value]
    return value


@dataclass(frozen=True)
class ExperimentAcceptance:
    probe_kind: str
    automated_checks: tuple[str, ...]
    human_checks: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "automated_checks", tuple(self.automated_checks))
        object.__setattr__(self, "human_checks", tuple(self.human_checks))


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    pack_id: str
    title: str
    version: str
    scene: Path
    scene_manifest: Path
    student_template: Path
    guide: Path
    capabilities: tuple[str, ...]
    workspace: Mapping[str, Any]
    public_parameters: Mapping[str, Any]
    acceptance: ExperimentAcceptance
    hardware_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(
            self,
            "workspace",
            _freeze_mapping(self.workspace, "workspace"),
        )
        object.__setattr__(
            self,
            "public_parameters",
            _freeze_mapping(self.public_parameters, "public_parameters"),
        )


@dataclass(frozen=True)
class CapabilityReport:
    available: tuple[str, ...]
    missing: tuple[str, ...]
    reasons: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", tuple(self.available))
        object.__setattr__(self, "missing", tuple(self.missing))
        object.__setattr__(
            self,
            "reasons",
            _freeze_mapping(self.reasons, "reasons"),
        )

    @property
    def ready(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class ExperimentRunContext:
    experiment_id: str
    experiment_version: str
    scene_path: Path
    scene_sha256: str
    scene_manifest_path: Path
    public_parameters: Mapping[str, Any]
    hardware_status: str = "PENDING_HARDWARE"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "public_parameters",
            _freeze_mapping(self.public_parameters, "public_parameters"),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_version": self.experiment_version,
            "scene_sha256": self.scene_sha256,
            "public_parameters": _thaw(self.public_parameters),
            "hardware_status": self.hardware_status,
        }
