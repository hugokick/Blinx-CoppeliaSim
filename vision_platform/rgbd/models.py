"""Immutable data contracts shared by the pure RGB-D algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import RgbdContractError


def _immutable_c_copy(source: np.ndarray) -> np.ndarray:
    """Copy an array onto a read-only bytes backing with C layout."""
    contiguous = np.array(source, dtype=source.dtype, copy=True, order="C")
    backing = contiguous.tobytes(order="C")
    return np.frombuffer(backing, dtype=contiguous.dtype).reshape(
        contiguous.shape, order="C"
    )


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    pixel_center_convention: str = "integer_center_top_left_zero"

    def __post_init__(self) -> None:
        if (
            type(self.width_px) is not int
            or type(self.height_px) is not int
            or self.width_px <= 0
            or self.height_px <= 0
        ):
            raise RgbdContractError(
                "RGBD_INTRINSICS_INVALID", "image size must be positive integers"
            )
        values = {
            "fx_px": self.fx_px,
            "fy_px": self.fy_px,
            "cx_px": self.cx_px,
            "cy_px": self.cy_px,
        }
        normalized: dict[str, float] = {}
        for name, raw in values.items():
            if type(raw) not in {int, float} or not math.isfinite(float(raw)):
                raise RgbdContractError(
                    "RGBD_INTRINSICS_INVALID", f"{name} must be finite"
                )
            normalized[name] = float(raw)
        if normalized["fx_px"] <= 0.0 or normalized["fy_px"] <= 0.0:
            raise RgbdContractError(
                "RGBD_INTRINSICS_INVALID", "focal lengths must be positive"
            )
        if self.pixel_center_convention != "integer_center_top_left_zero":
            raise RgbdContractError(
                "RGBD_INTRINSICS_INVALID", "pixel centre convention is unsupported"
            )
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class RgbdFrame:
    image_bgr: np.ndarray
    depth_m: np.ndarray

    def __post_init__(self) -> None:
        source_image = self.image_bgr
        source_depth = self.depth_m
        if (
            not isinstance(source_image, np.ndarray)
            or source_image.dtype != np.uint8
            or source_image.ndim != 3
            or source_image.shape[2] != 3
            or source_image.shape[0] <= 0
            or source_image.shape[1] <= 0
        ):
            raise RgbdContractError(
                "RGBD_FRAME_INVALID", "image_bgr must be non-empty uint8 HxWx3"
            )
        if (
            not isinstance(source_depth, np.ndarray)
            or source_depth.dtype != np.float32
            or source_depth.ndim != 2
            or source_depth.shape != source_image.shape[:2]
            or not np.all(np.isfinite(source_depth))
            or np.any(source_depth < 0.0)
        ):
            raise RgbdContractError(
                "RGBD_FRAME_INVALID",
                "depth_m must be finite non-negative float32 HxW matching image_bgr",
            )
        image = _immutable_c_copy(source_image)
        depth = _immutable_c_copy(source_depth)
        object.__setattr__(self, "image_bgr", image)
        object.__setattr__(self, "depth_m", depth)


@dataclass(frozen=True)
class DepthSample:
    status: str
    u_px: int
    v_px: int
    window_size: int
    valid_count: int
    depth_m: float | None
    failure_code: str | None

    def __post_init__(self) -> None:
        if (
            type(self.u_px) is not int
            or type(self.v_px) is not int
            or self.u_px < 0
            or self.v_px < 0
            or type(self.window_size) is not int
            or self.window_size not in {1, 3, 5, 7, 9}
            or type(self.valid_count) is not int
            or not 0 <= self.valid_count <= self.window_size * self.window_size
        ):
            raise RgbdContractError(
                "RGBD_SAMPLE_INVALID", "sample index, window, or count is invalid"
            )
        if self.status == "PASS":
            if (
                self.valid_count <= 0
                or type(self.depth_m) is not float
                or not math.isfinite(self.depth_m)
                or self.depth_m <= 0.0
                or self.failure_code is not None
            ):
                raise RgbdContractError(
                    "RGBD_SAMPLE_INVALID", "PASS sample fields are inconsistent"
                )
        elif self.status == "NO_VALID_DEPTH":
            if self.depth_m is not None or self.failure_code != "NO_VALID_DEPTH":
                raise RgbdContractError(
                    "RGBD_SAMPLE_INVALID", "missing-depth fields are inconsistent"
                )
        else:
            raise RgbdContractError(
                "RGBD_SAMPLE_INVALID", "sample status is unsupported"
            )


@dataclass(frozen=True)
class Point3M:
    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise RgbdContractError(
                    "RGBD_POINT_INVALID", f"{name} must be finite"
                )
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class RgbdMeasurement:
    status: str
    sample: DepthSample
    camera_frame_id: str
    target_frame_id: str | None
    point_camera_m: Point3M | None
    point_target_m: Point3M | None
    failure_code: str | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.sample, DepthSample):
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "sample type is invalid"
            )
        if self.point_camera_m is not None and not isinstance(self.point_camera_m, Point3M):
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "camera point type is invalid"
            )
        if self.point_target_m is not None and not isinstance(self.point_target_m, Point3M):
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "target point type is invalid"
            )
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "schema version must be 1"
            )
        if (
            type(self.camera_frame_id) is not str
            or not self.camera_frame_id
            or len(self.camera_frame_id) > 80
        ):
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "camera_frame_id is invalid"
            )
        if self.target_frame_id is not None and (
            type(self.target_frame_id) is not str
            or not self.target_frame_id
            or len(self.target_frame_id) > 80
        ):
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "target_frame_id is invalid"
            )
        if self.status == "PASS":
            if (
                self.sample.status != "PASS"
                or not isinstance(self.point_camera_m, Point3M)
                or self.failure_code is not None
                or (self.target_frame_id is None) != (self.point_target_m is None)
            ):
                raise RgbdContractError(
                    "RGBD_MEASUREMENT_INVALID", "PASS fields are inconsistent"
                )
        elif self.status == "NO_VALID_DEPTH":
            if (
                self.sample.status != "NO_VALID_DEPTH"
                or self.point_camera_m is not None
                or self.point_target_m is not None
                or self.failure_code != "NO_VALID_DEPTH"
            ):
                raise RgbdContractError(
                    "RGBD_MEASUREMENT_INVALID", "failure fields are inconsistent"
                )
        else:
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "measurement status is unsupported"
            )


__all__ = [
    "CameraIntrinsics",
    "DepthSample",
    "Point3M",
    "RgbdFrame",
    "RgbdMeasurement",
]
