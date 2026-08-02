"""Immutable data contracts shared by the pure RGB-D algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import RgbdContractError


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
        image = np.array(source_image, dtype=np.uint8, copy=True, order="C")
        depth = np.array(source_depth, dtype=np.float32, copy=True, order="C")
        image.setflags(write=False)
        depth.setflags(write=False)
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


__all__ = ["CameraIntrinsics", "DepthSample", "RgbdFrame"]
