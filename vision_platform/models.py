from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class Frame:
    image_bgr: np.ndarray
    width: int
    height: int
    timestamp_s: float
    source: str
    sequence_id: int
    intrinsics: Mapping[str, Any] | None = None
    depth_m: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a numpy array")
        if self.image_bgr.ndim != 3 or self.image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have HxWx3 BGR shape")
        if self.image_bgr.shape[:2] != (self.height, self.width):
            raise ValueError("frame dimensions do not match image")
        if self.depth_m is not None and self.depth_m.shape != (self.height, self.width):
            raise ValueError("depth dimensions do not match image")
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")


@dataclass(frozen=True)
class Detection:
    detection_id: str
    center_px: tuple[float, float]
    color: str
    shape: str
    angle_deg: float
    area_px: float
    confidence: float
    contour: np.ndarray | None = None
    annotated_roi: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.center_px) != 2:
            raise ValueError("center_px must contain x and y")
        object.__setattr__(
            self,
            "center_px",
            (float(self.center_px[0]), float(self.center_px[1])),
        )
        if self.area_px < 0:
            raise ValueError("area_px must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class CalibrationRecord:
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    pixel_points: tuple[tuple[float, float], ...]
    world_points_mm: tuple[tuple[float, float], ...]
    image_size: tuple[int, int]
    plane_z_mm: float
    source: str
    scene_version: str | None = None
    rms_error_mm: float | None = None
    max_error_mm: float | None = None
    validation_sample_count: int | None = None
    created_at: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class TaskEvent:
    state: str
    message: str
    timestamp_s: float
    data: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    status: str
    events: tuple[TaskEvent, ...] | list[TaskEvent]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "events": [event.to_dict() for event in self.events],
            "metrics": _jsonable(self.metrics),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "artifacts": _jsonable(self.artifacts),
        }
