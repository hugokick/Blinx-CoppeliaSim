"""Perspective intrinsics derivation for the square D1-01 sensor."""

from __future__ import annotations

import math

from vision_platform.rgbd.models import CameraIntrinsics

from .errors import RgbdSimContractError
from .models import RgbdSensorMetadata


def expected_intrinsics(
    width: int, height: int, perspective_angle_rad: float
) -> tuple[float, float]:
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or type(perspective_angle_rad) not in {int, float}
        or not math.isfinite(float(perspective_angle_rad))
        or not 0.0 < float(perspective_angle_rad) < math.pi
    ):
        raise RgbdSimContractError(
            "RGBD_SIM_INTRINSICS_INVALID", "resolution or perspective angle is invalid"
        )
    angle = float(perspective_angle_rad)
    focal = max(width, height) / (2.0 * math.tan(angle / 2.0))
    return float(focal), float(focal)


def derive_intrinsics(metadata: RgbdSensorMetadata) -> CameraIntrinsics:
    if not isinstance(metadata, RgbdSensorMetadata):
        raise RgbdSimContractError(
            "RGBD_SIM_INTRINSICS_INVALID", "metadata type is invalid"
        )
    width, height = metadata.resolution
    fx, fy = expected_intrinsics(width, height, metadata.perspective_angle_rad)
    return CameraIntrinsics(
        width,
        height,
        fx,
        fy,
        (width - 1) / 2.0,
        (height - 1) / 2.0,
    )


__all__ = ["derive_intrinsics", "expected_intrinsics"]
