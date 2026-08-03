"""Immutable metadata and capture contracts for the D1-01 adapter."""

from __future__ import annotations

import math
import hashlib
import json
import re
import weakref
from dataclasses import dataclass, field
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


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _source_digest(source: RgbdSourceCapture) -> str:
    return _array_digest(source.image_bgr, source.source_depth_m)


def _frame_digest(frame: RgbdFrame) -> str:
    return _array_digest(frame.image_bgr, frame.depth_m)


_CAPTURE_REGISTRY: dict[
    int,
    tuple[weakref.ReferenceType["RgbdSimCapture"], tuple[object, ...]],
] = {}
_CAPTURE_CAPABILITY = object()


def _register_capture(capture: "RgbdSimCapture") -> None:
    key = id(capture)
    trusted = (
        capture.source,
        capture.frame,
        capture.intrinsics,
        capture.observed_source_depth_model,
        capture.output_depth_model,
        capture._proof_scene_sha256,
        capture._proof_source_digest,
        capture._proof_sequence_id,
        capture._proof_anchor_digest,
        capture._proof_frame_digest,
    )
    _CAPTURE_REGISTRY[key] = (
        weakref.ref(capture, lambda reference, key=key: _CAPTURE_REGISTRY.pop(key, None)),
        trusted,
    )


def _untrusted_capture_error(code: str, message: str) -> RgbdSimContractError:
    return RgbdSimContractError(code, message)


@dataclass(frozen=True, init=False, eq=False)
class RgbdSimCapture:
    source: RgbdSourceCapture
    frame: RgbdFrame
    intrinsics: CameraIntrinsics
    observed_source_depth_model: str
    output_depth_model: str = "optical_z"
    _proof_scene_sha256: str = field(repr=False, compare=False)
    _proof_source_digest: str = field(repr=False, compare=False)
    _proof_sequence_id: int = field(repr=False, compare=False)
    _proof_anchor_digest: str = field(repr=False, compare=False)
    _proof_frame_digest: str = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RgbdSimContractError(
            "RGBD_SIM_CAPTURE_UNOBSERVED",
            "captures must be produced by source-depth normalization",
        )

    def _validate_payload(self) -> None:
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
        if (
            type(self._proof_scene_sha256) is not str
            or _SHA256.fullmatch(self._proof_scene_sha256) is None
            or type(self._proof_source_digest) is not str
            or _SHA256.fullmatch(self._proof_source_digest) is None
            or type(self._proof_sequence_id) is not int
            or self._proof_sequence_id < 0
            or type(self._proof_anchor_digest) is not str
            or _SHA256.fullmatch(self._proof_anchor_digest) is None
            or type(self._proof_frame_digest) is not str
            or _SHA256.fullmatch(self._proof_frame_digest) is None
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "capture normalization proof is invalid"
            )
        metadata = self.source.metadata
        if (
            self._proof_scene_sha256 != metadata.scene_sha256
            or self._proof_sequence_id != metadata.sequence_id
            or self._proof_source_digest != _source_digest(self.source)
            or self._proof_frame_digest != _frame_digest(self.frame)
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_BINDING_INVALID",
                "capture no longer matches its normalization proof",
            )

    @classmethod
    def _from_normalization(
        cls,
        *,
        source: RgbdSourceCapture,
        frame: RgbdFrame,
        intrinsics: CameraIntrinsics,
        observed_source_depth_model: str,
        output_depth_model: str = "optical_z",
        proof_scene_sha256: str,
        proof_source_digest: str,
        proof_sequence_id: int,
        proof_anchor_digest: str,
        _capability: object | None = None,
    ) -> "RgbdSimCapture":
        if _capability is not _CAPTURE_CAPABILITY:
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_UNOBSERVED",
                "captures must be produced by source-depth normalization",
            )
        capture = object.__new__(cls)
        object.__setattr__(capture, "source", source)
        object.__setattr__(capture, "frame", frame)
        object.__setattr__(capture, "intrinsics", intrinsics)
        object.__setattr__(capture, "observed_source_depth_model", observed_source_depth_model)
        object.__setattr__(capture, "output_depth_model", output_depth_model)
        object.__setattr__(capture, "_proof_scene_sha256", proof_scene_sha256)
        object.__setattr__(capture, "_proof_source_digest", proof_source_digest)
        object.__setattr__(capture, "_proof_sequence_id", proof_sequence_id)
        object.__setattr__(capture, "_proof_anchor_digest", proof_anchor_digest)
        object.__setattr__(capture, "_proof_frame_digest", _frame_digest(frame))
        capture._validate_payload()
        _register_capture(capture)
        return capture


def _assert_capture_trusted(capture: RgbdSimCapture) -> None:
    if not isinstance(capture, RgbdSimCapture):
        raise _untrusted_capture_error("RGBD_SIM_CAPTURE_INVALID", "capture type is invalid")
    entry = _CAPTURE_REGISTRY.get(id(capture))
    if entry is None or entry[0]() is not capture:
        raise _untrusted_capture_error(
            "RGBD_SIM_CAPTURE_UNOBSERVED",
            "capture is not a live normalized source-depth result",
        )
    trusted = entry[1]
    current = (
        getattr(capture, "source", None),
        getattr(capture, "frame", None),
        getattr(capture, "intrinsics", None),
        getattr(capture, "observed_source_depth_model", None),
        getattr(capture, "output_depth_model", None),
        getattr(capture, "_proof_scene_sha256", None),
        getattr(capture, "_proof_source_digest", None),
        getattr(capture, "_proof_sequence_id", None),
        getattr(capture, "_proof_anchor_digest", None),
        getattr(capture, "_proof_frame_digest", None),
    )
    identity_mismatch = any(
        left is not right for left, right in zip(current[:3], trusted[:3])
    )
    value_mismatch = any(
        type(left) is not type(right) or left != right
        for left, right in zip(current[3:], trusted[3:])
    )
    if identity_mismatch or value_mismatch:
        raise _untrusted_capture_error(
            "RGBD_SIM_CAPTURE_BINDING_INVALID",
            "capture fields no longer match the normalization proof",
        )
    try:
        capture._validate_payload()
    except Exception as exc:
        if isinstance(exc, RgbdSimContractError) and exc.code == "RGBD_SIM_CAPTURE_BINDING_INVALID":
            raise
        raise _untrusted_capture_error(
            "RGBD_SIM_CAPTURE_BINDING_INVALID",
            "capture no longer satisfies its normalization proof",
        ) from exc


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
    _assert_capture_trusted(capture)
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
