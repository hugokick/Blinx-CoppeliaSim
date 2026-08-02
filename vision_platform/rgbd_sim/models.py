"""Immutable metadata and capture contracts for the D1-01 adapter."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame

from .errors import RgbdSimContractError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _immutable_c_copy(source: np.ndarray) -> np.ndarray:
    contiguous = np.array(source, dtype=source.dtype, copy=True, order="C")
    backing = contiguous.tobytes(order="C")
    return np.frombuffer(backing, dtype=contiguous.dtype).reshape(
        contiguous.shape, order="C"
    )


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


@dataclass(frozen=True)
class RgbdSensorMetadata:
    sensor_path: str
    resolution: tuple[int, int]
    near_clip_m: float
    far_clip_m: float
    perspective_angle_rad: float
    scene_path: str
    scene_sha256: str
    sequence_id: int
    timestamp_s: float
    expected_source_depth_model: str
    explicit_handling: bool = True
    perspective: bool = True
    rgb_enabled: bool = True
    depth_enabled: bool = True
    vertical_flip: str = "vertical_flip"
    color_conversion: str = "RGB_to_BGR"
    depth_unit: str = "metre"
    pixel_center_convention: str = "integer_center_top_left_zero"

    def __post_init__(self) -> None:
        if type(self.sensor_path) is not str or not self.sensor_path.startswith("/"):
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "sensor_path must be an absolute simulator path"
            )
        if (
            type(self.resolution) is not tuple
            or len(self.resolution) != 2
            or any(type(value) is not int or value <= 0 for value in self.resolution)
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "resolution must contain positive integers"
            )
        if (
            not _finite_number(self.near_clip_m)
            or not _finite_number(self.far_clip_m)
            or float(self.near_clip_m) <= 0.0
            or float(self.far_clip_m) <= float(self.near_clip_m)
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "near/far clipping values are invalid"
            )
        if (
            not _finite_number(self.perspective_angle_rad)
            or not 0.0 < float(self.perspective_angle_rad) < math.pi
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "perspective angle is invalid"
            )
        if type(self.scene_path) is not str or not self.scene_path or self.scene_path.startswith("/"):
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "scene_path must be a relative path"
            )
        if type(self.scene_sha256) is not str or _SHA256.fullmatch(self.scene_sha256) is None:
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "scene_sha256 must be lowercase SHA-256"
            )
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "sequence_id must be nonnegative"
            )
        if not _finite_number(self.timestamp_s):
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "timestamp_s must be finite"
            )
        if self.expected_source_depth_model not in {"optical_z", "ray_range"}:
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "source depth model is unsupported"
            )
        if (
            type(self.explicit_handling) is not bool
            or not self.explicit_handling
            or type(self.perspective) is not bool
            or not self.perspective
            or type(self.rgb_enabled) is not bool
            or not self.rgb_enabled
            or type(self.depth_enabled) is not bool
            or not self.depth_enabled
            or self.vertical_flip != "vertical_flip"
            or self.color_conversion != "RGB_to_BGR"
            or self.depth_unit != "metre"
            or self.pixel_center_convention != "integer_center_top_left_zero"
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_METADATA_INVALID", "sensor conversion conventions are invalid"
            )
        object.__setattr__(self, "resolution", tuple(self.resolution))
        for name in ("near_clip_m", "far_clip_m", "perspective_angle_rad", "timestamp_s"):
            object.__setattr__(self, name, float(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class RgbdSourceCapture:
    metadata: RgbdSensorMetadata
    image_bgr: np.ndarray
    source_depth_m: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, RgbdSensorMetadata):
            raise RgbdSimContractError(
                "RGBD_SIM_SOURCE_INVALID", "metadata type is invalid"
            )
        image = self.image_bgr
        depth = self.source_depth_m
        width, height = self.metadata.resolution
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape != (height, width, 3)
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_SOURCE_INVALID", "image_bgr shape or dtype is invalid"
            )
        if (
            not isinstance(depth, np.ndarray)
            or depth.dtype != np.float32
            or depth.ndim != 2
            or depth.shape != (height, width)
            or not np.all(np.isfinite(depth))
            or np.any(depth < 0.0)
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_SOURCE_INVALID", "source_depth_m shape or values are invalid"
            )
        object.__setattr__(self, "image_bgr", _immutable_c_copy(image))
        object.__setattr__(self, "source_depth_m", _immutable_c_copy(depth))


@dataclass(frozen=True, eq=False)
class RgbdSimCapture:
    source: RgbdSourceCapture
    frame: RgbdFrame
    intrinsics: CameraIntrinsics
    observed_source_depth_model: str
    output_depth_model: str = "optical_z"

    def __post_init__(self) -> None:
        if not isinstance(self.source, RgbdSourceCapture):
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "source capture type is invalid"
            )
        if not isinstance(self.frame, RgbdFrame) or not isinstance(
            self.intrinsics, CameraIntrinsics
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "frame or intrinsics type is invalid"
            )
        width, height = self.source.metadata.resolution
        if (
            self.frame.image_bgr.shape != (height, width, 3)
            or self.frame.depth_m.shape != (height, width)
            or self.intrinsics.width_px != width
            or self.intrinsics.height_px != height
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "capture dimensions do not match"
            )
        if self.observed_source_depth_model not in {"optical_z", "ray_range"}:
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "observed source model is invalid"
            )
        if self.observed_source_depth_model != self.source.metadata.expected_source_depth_model:
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "observed source model disagrees with expectation"
            )
        if self.output_depth_model != "optical_z":
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "output depth model must be optical_z"
            )


def _summary(depth: np.ndarray) -> dict[str, Any]:
    valid = depth[depth > 0.0]
    if valid.size == 0:
        return {"valid_count": 0, "invalid_count": int(depth.size)}
    return {
        "valid_count": int(valid.size),
        "invalid_count": int(depth.size - valid.size),
        "min_depth_m": float(np.min(valid)),
        "median_depth_m": float(np.median(valid)),
        "max_depth_m": float(np.max(valid)),
    }


def _metadata_to_dict(metadata: RgbdSensorMetadata) -> dict[str, Any]:
    return {
        "sensor_path": metadata.sensor_path,
        "resolution": [metadata.resolution[0], metadata.resolution[1]],
        "near_clip_m": metadata.near_clip_m,
        "far_clip_m": metadata.far_clip_m,
        "perspective_angle_rad": metadata.perspective_angle_rad,
        "scene_path": metadata.scene_path,
        "scene_sha256": metadata.scene_sha256,
        "sequence_id": metadata.sequence_id,
        "timestamp_s": metadata.timestamp_s,
        "expected_source_depth_model": metadata.expected_source_depth_model,
        "explicit_handling": metadata.explicit_handling,
        "perspective": metadata.perspective,
        "rgb_enabled": metadata.rgb_enabled,
        "depth_enabled": metadata.depth_enabled,
        "vertical_flip": metadata.vertical_flip,
        "color_conversion": metadata.color_conversion,
        "depth_unit": metadata.depth_unit,
        "pixel_center_convention": metadata.pixel_center_convention,
    }


def source_capture_to_dict(source: RgbdSourceCapture) -> dict[str, Any]:
    if not isinstance(source, RgbdSourceCapture):
        raise RgbdSimContractError("RGBD_SIM_SOURCE_INVALID", "source type is invalid")
    payload = _metadata_to_dict(source.metadata)
    payload["source_depth_summary"] = _summary(source.source_depth_m)
    return payload


def capture_to_dict(capture: RgbdSimCapture) -> dict[str, Any]:
    if not isinstance(capture, RgbdSimCapture):
        raise RgbdSimContractError("RGBD_SIM_CAPTURE_INVALID", "capture type is invalid")
    payload = _metadata_to_dict(capture.source.metadata)
    payload.update(
        {
            "observed_source_depth_model": capture.observed_source_depth_model,
            "output_depth_model": capture.output_depth_model,
            "intrinsics": {
                "fx_px": capture.intrinsics.fx_px,
                "fy_px": capture.intrinsics.fy_px,
                "cx_px": capture.intrinsics.cx_px,
                "cy_px": capture.intrinsics.cy_px,
                "pixel_center_convention": capture.intrinsics.pixel_center_convention,
            },
            "output_depth_summary": _summary(capture.frame.depth_m),
        }
    )
    return payload


__all__ = [
    "RgbdSensorMetadata",
    "RgbdSourceCapture",
    "RgbdSimCapture",
    "capture_to_dict",
    "source_capture_to_dict",
]
