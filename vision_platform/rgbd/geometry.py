"""Pinhole-camera deprojection for metric RGB-D samples."""

from __future__ import annotations

import math

from .errors import RgbdContractError
from .models import CameraIntrinsics, Point3M


def deproject_pixel(
    intrinsics: CameraIntrinsics,
    *,
    u_px: float,
    v_px: float,
    depth_m: float,
) -> Point3M:
    if not isinstance(intrinsics, CameraIntrinsics):
        raise RgbdContractError(
            "RGBD_DEPROJECTION_INVALID", "intrinsics are invalid"
        )
    normalized: list[float] = []
    for name, raw in (("u_px", u_px), ("v_px", v_px), ("depth_m", depth_m)):
        if type(raw) not in {int, float} or not math.isfinite(float(raw)):
            raise RgbdContractError(
                "RGBD_DEPROJECTION_INVALID", f"{name} must be finite"
            )
        normalized.append(float(raw))
    u_value, v_value, depth_value = normalized
    if (
        not 0.0 <= u_value <= intrinsics.width_px - 1
        or not 0.0 <= v_value <= intrinsics.height_px - 1
        or depth_value <= 0.0
    ):
        raise RgbdContractError(
            "RGBD_DEPROJECTION_INVALID", "pixel or depth is outside its domain"
        )
    x_m = (u_value - intrinsics.cx_px) * depth_value / intrinsics.fx_px
    y_m = (v_value - intrinsics.cy_px) * depth_value / intrinsics.fy_px
    return Point3M(x_m, y_m, depth_value)


__all__ = ["deproject_pixel"]
