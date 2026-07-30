from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

from vision_platform.errors import CalibrationError
from vision_platform.models import CalibrationRecord


@dataclass(frozen=True)
class CalibrationMetrics:
    rms_error_mm: float
    max_error_mm: float
    sample_count: int


@dataclass
class AffineCalibration:
    matrix: np.ndarray
    pixel_points: tuple[tuple[float, float], ...]
    world_points_mm: tuple[tuple[float, float], ...]
    image_size: tuple[int, int]
    plane_z_mm: float
    source: str = "unknown"
    scene_version: str | None = None
    metrics: CalibrationMetrics | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (2, 3):
            raise CalibrationError("Affine matrix must be 2x3")
        self.matrix = matrix
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.plane_z_mm = float(self.plane_z_mm)
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def fit(
        cls,
        pixel_points: Sequence[Sequence[float]],
        world_points_mm: Sequence[Sequence[float]],
        *,
        image_size: tuple[int, int],
        plane_z_mm: float,
        source: str = "unknown",
        scene_version: str | None = None,
    ) -> "AffineCalibration":
        if len(pixel_points) != len(world_points_mm):
            raise CalibrationError(
                "Pixel and world points must have the same number of samples"
            )
        if len(pixel_points) < 3:
            raise CalibrationError("At least three calibration points are required")

        pixels = np.asarray(pixel_points, dtype=np.float64)
        worlds = np.asarray(world_points_mm, dtype=np.float64)
        if pixels.shape != (len(pixel_points), 2):
            raise CalibrationError("Pixel points must be Nx2")
        if worlds.shape != (len(world_points_mm), 2):
            raise CalibrationError("World points must be Nx2")

        design = np.column_stack([pixels, np.ones(len(pixels))])
        if np.linalg.matrix_rank(design) < 3:
            raise CalibrationError("Calibration pixel points are collinear")
        world_design = np.column_stack([worlds, np.ones(len(worlds))])
        if np.linalg.matrix_rank(world_design) < 3:
            raise CalibrationError("Calibration world points are collinear")

        coefficients, _, _, _ = np.linalg.lstsq(design, worlds, rcond=None)
        matrix = np.round(coefficients.T, 12)
        return cls(
            matrix=matrix,
            pixel_points=tuple(
                (float(point[0]), float(point[1])) for point in pixels
            ),
            world_points_mm=tuple(
                (float(point[0]), float(point[1])) for point in worlds
            ),
            image_size=(int(image_size[0]), int(image_size[1])),
            plane_z_mm=float(plane_z_mm),
            source=source,
            scene_version=scene_version,
        )

    def pixel_to_world(
        self,
        pixel: Sequence[float],
    ) -> tuple[float, float]:
        if len(pixel) != 2:
            raise CalibrationError("Pixel coordinate must contain x and y")
        homogeneous = np.array(
            [float(pixel[0]), float(pixel[1]), 1.0],
            dtype=np.float64,
        )
        world = self.matrix @ homogeneous
        return (round(float(world[0]), 9), round(float(world[1]), 9))

    def evaluate(
        self,
        validation_pairs: Iterable[
            tuple[Sequence[float], Sequence[float]]
        ],
    ) -> CalibrationMetrics:
        errors: list[float] = []
        for pixel, expected_world in validation_pairs:
            predicted = np.asarray(self.pixel_to_world(pixel), dtype=np.float64)
            expected = np.asarray(expected_world, dtype=np.float64)
            if expected.shape != (2,):
                raise CalibrationError("Validation world coordinate must be XY")
            errors.append(float(np.linalg.norm(predicted - expected)))
        if not errors:
            raise CalibrationError("At least one validation point is required")
        values = np.asarray(errors, dtype=np.float64)
        metrics = CalibrationMetrics(
            rms_error_mm=float(np.sqrt(np.mean(np.square(values)))),
            max_error_mm=float(np.max(values)),
            sample_count=len(errors),
        )
        self.metrics = metrics
        return metrics

    def validate_image_size(self, image_size: tuple[int, int]) -> None:
        requested = (int(image_size[0]), int(image_size[1]))
        if requested != self.image_size:
            raise CalibrationError(
                f"Calibration image size {self.image_size} does not match {requested}",
                expected=self.image_size,
                actual=requested,
            )

    def to_record(self) -> CalibrationRecord:
        metrics = self.metrics
        return CalibrationRecord(
            matrix=tuple(
                tuple(float(value) for value in row) for row in self.matrix
            ),
            pixel_points=self.pixel_points,
            world_points_mm=self.world_points_mm,
            image_size=self.image_size,
            plane_z_mm=self.plane_z_mm,
            source=self.source,
            scene_version=self.scene_version,
            rms_error_mm=metrics.rms_error_mm if metrics else None,
            max_error_mm=metrics.max_error_mm if metrics else None,
            validation_sample_count=metrics.sample_count if metrics else None,
            created_at=self.created_at,
        )
