from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


def _finite_positive(value: float, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite positive")
    return normalized


@dataclass(frozen=True)
class PixelScale:
    mm_per_pixel_x: float
    mm_per_pixel_y: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mm_per_pixel_x",
            _finite_positive(self.mm_per_pixel_x, "mm_per_pixel_x"),
        )
        object.__setattr__(
            self,
            "mm_per_pixel_y",
            _finite_positive(self.mm_per_pixel_y, "mm_per_pixel_y"),
        )


@dataclass(frozen=True)
class Vision2DConfig:
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.40
    saturation_min: int = 60
    value_min: int = 40
    gaussian_kernel_size: int = 5
    morphology_kernel_size: int = 3
    border_margin_px: int = 1
    polygon_epsilon_ratio: float = 0.03
    square_aspect_min: float = 0.85
    circle_circularity_min: float = 0.70
    pixel_scale: PixelScale | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.min_area_ratio < self.max_area_ratio <= 1.0:
            raise ValueError("area ratios must satisfy 0 < min < max <= 1")
        for name in ("gaussian_kernel_size", "morphology_kernel_size"):
            value = int(getattr(self, name))
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be positive odd")
        if not 0 <= self.saturation_min <= 255:
            raise ValueError("saturation_min must be between 0 and 255")
        if not 0 <= self.value_min <= 255:
            raise ValueError("value_min must be between 0 and 255")
        if self.border_margin_px < 0:
            raise ValueError("border_margin_px must be non-negative")
        if not 0.0 < self.polygon_epsilon_ratio < 1.0:
            raise ValueError("polygon_epsilon_ratio must be between 0 and 1")
        if not 0.0 < self.square_aspect_min <= 1.0:
            raise ValueError("square_aspect_min must be between 0 and 1")
        if not 0.0 < self.circle_circularity_min <= 1.0:
            raise ValueError("circle_circularity_min must be between 0 and 1")


@dataclass(frozen=True)
class TargetMeasurement:
    detection_id: str
    center_px: tuple[float, float]
    axis_aligned_bbox_px: tuple[int, int, int, int]
    rotated_box_px: tuple[tuple[float, float], ...]
    long_side_px: float
    short_side_px: float
    angle_deg: float | None
    area_px2: float
    perimeter_px: float
    size_mm: tuple[float, float] | None
    area_mm2: float | None
    perimeter_mm: float | None
    color: str
    shape: str
    vertex_count: int
    circularity: float
    aspect_ratio: float
    quality_flags: tuple[str, ...] = ()
    contour: np.ndarray = field(
        repr=False,
        compare=False,
        default_factory=lambda: np.empty((0, 1, 2), dtype=np.int32),
    )


@dataclass(frozen=True)
class RejectedTarget:
    candidate_index: int
    center_px: tuple[float, float] | None
    area_px2: float
    code: str
    reason: str
    contour: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class Vision2DResult:
    status: str
    image_size: tuple[int, int]
    targets: tuple[TargetMeasurement, ...] = ()
    rejected_targets: tuple[RejectedTarget, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "PARTIAL", "NO_TARGETS", "REJECTED"}:
            raise ValueError("status is not a supported Vision2D result status")
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError("image_size must contain positive width and height")


@dataclass(frozen=True)
class Vision2DAnalysis:
    result: Vision2DResult
    intermediate_images: Mapping[str, np.ndarray]
