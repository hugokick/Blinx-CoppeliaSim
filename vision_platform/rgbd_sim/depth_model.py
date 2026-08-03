"""Independent source-depth model observation and optical-Z normalization."""

from __future__ import annotations

import math
import hashlib
import json
import weakref
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .errors import RgbdSimContractError
from .intrinsics import derive_intrinsics
from .models import (
    RgbdSimCapture,
    RgbdSourceCapture,
    _CAPTURE_CAPABILITY,
    _source_digest,
)


@dataclass(frozen=True)
class DepthAnchor:
    u_px: int
    v_px: int
    optical_z_m: float
    ray_range_m: float
    tolerance_m: float

    def __post_init__(self) -> None:
        if (
            type(self.u_px) is not int
            or type(self.v_px) is not int
            or type(self.optical_z_m) not in {int, float}
            or type(self.ray_range_m) not in {int, float}
            or type(self.tolerance_m) not in {int, float}
            or not math.isfinite(float(self.optical_z_m))
            or not math.isfinite(float(self.ray_range_m))
            or not math.isfinite(float(self.tolerance_m))
            or float(self.optical_z_m) <= 0.0
            or float(self.ray_range_m) <= 0.0
            or float(self.tolerance_m) <= 0.0
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_DEPTH_MODEL_INVALID", "depth anchor values are invalid"
            )


_OBSERVATION_REGISTRY: dict[int, weakref.ReferenceType["SourceDepthModelObservation"]] = {}
_OBSERVATION_CAPABILITY = object()


def _anchor_digest(anchors: Sequence[DepthAnchor]) -> str:
    canonical = [
        [
            anchor.u_px,
            anchor.v_px,
            float(anchor.optical_z_m),
            float(anchor.ray_range_m),
            float(anchor.tolerance_m),
        ]
        for anchor in anchors
    ]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def _register_observation(observation: "SourceDepthModelObservation") -> None:
    key = id(observation)
    _OBSERVATION_REGISTRY[key] = weakref.ref(
        observation, lambda reference, key=key: _OBSERVATION_REGISTRY.pop(key, None)
    )


def _is_registered_observation(observation: object) -> bool:
    reference = _OBSERVATION_REGISTRY.get(id(observation))
    return reference is not None and reference() is observation


@dataclass(frozen=True, init=False)
class SourceDepthModelObservation:
    model: str
    optical_error_m: float
    ray_range_error_m: float
    scene_sha256: str
    source_digest: str
    sequence_id: int
    anchor_digest: str

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_UNOBSERVED",
            "observations must be produced by source-depth measurement",
        )

    @classmethod
    def _from_measurement(
        cls,
        *,
        model: str,
        optical_error_m: float,
        ray_range_error_m: float,
        scene_sha256: str,
        source_digest: str,
        sequence_id: int,
        anchor_digest: str,
        _capability: object | None = None,
    ) -> "SourceDepthModelObservation":
        if _capability is not _OBSERVATION_CAPABILITY:
            raise RgbdSimContractError(
                "RGBD_SIM_DEPTH_MODEL_UNOBSERVED",
                "observations must be produced by source-depth measurement",
            )
        if model not in {"optical_z", "ray_range"}:
            raise RgbdSimContractError(
                "RGBD_SIM_DEPTH_MODEL_INVALID", "observed source model is invalid"
            )
        if not all(
            type(error) in {int, float} and math.isfinite(float(error))
            for error in (optical_error_m, ray_range_error_m)
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_DEPTH_MODEL_INVALID", "observation errors must be finite"
            )
        if (
            type(scene_sha256) is not str
            or len(scene_sha256) != 64
            or any(character not in "0123456789abcdef" for character in scene_sha256)
            or type(source_digest) is not str
            or len(source_digest) != 64
            or any(character not in "0123456789abcdef" for character in source_digest)
            or type(anchor_digest) is not str
            or len(anchor_digest) != 64
            or any(character not in "0123456789abcdef" for character in anchor_digest)
            or type(sequence_id) is not int
            or sequence_id < 0
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_DEPTH_MODEL_INVALID", "observation binding is invalid"
            )
        observation = object.__new__(cls)
        object.__setattr__(observation, "model", model)
        object.__setattr__(observation, "optical_error_m", float(optical_error_m))
        object.__setattr__(observation, "ray_range_error_m", float(ray_range_error_m))
        object.__setattr__(observation, "scene_sha256", scene_sha256)
        object.__setattr__(observation, "source_digest", source_digest)
        object.__setattr__(observation, "sequence_id", sequence_id)
        object.__setattr__(observation, "anchor_digest", anchor_digest)
        _register_observation(observation)
        return observation


def observe_source_depth_model(
    source: RgbdSourceCapture,
    anchors: Sequence[DepthAnchor],
) -> SourceDepthModelObservation:
    if not isinstance(source, RgbdSourceCapture) or not isinstance(anchors, Sequence):
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_INVALID", "source or anchors are invalid"
        )
    if not anchors:
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_INVALID", "at least one depth anchor is required"
        )
    width, height = source.metadata.resolution
    optical_errors: list[float] = []
    ray_errors: list[float] = []
    for anchor in anchors:
        if not isinstance(anchor, DepthAnchor) or not (
            0 <= anchor.u_px < width and 0 <= anchor.v_px < height
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_DEPTH_MODEL_INVALID", "depth anchor is outside the frame"
            )
        observed = float(source.source_depth_m[anchor.v_px, anchor.u_px])
        optical_errors.append(abs(observed - float(anchor.optical_z_m)))
        ray_errors.append(abs(observed - float(anchor.ray_range_m)))
    optical_ok = all(error <= float(anchor.tolerance_m) for error, anchor in zip(optical_errors, anchors))
    ray_ok = all(error <= float(anchor.tolerance_m) for error, anchor in zip(ray_errors, anchors))
    if optical_ok == ray_ok:
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_INVALID",
            "source-depth model is ambiguous or matches neither geometry",
        )
    optical_error = max(optical_errors)
    ray_error = max(ray_errors)
    if not math.isfinite(optical_error) or not math.isfinite(ray_error):
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_INVALID", "observation errors must be finite"
        )
    return SourceDepthModelObservation._from_measurement(
        model="optical_z" if optical_ok else "ray_range",
        optical_error_m=optical_error,
        ray_range_error_m=ray_error,
        scene_sha256=source.metadata.scene_sha256,
        source_digest=_source_digest(source),
        sequence_id=source.metadata.sequence_id,
        anchor_digest=_anchor_digest(anchors),
        _capability=_OBSERVATION_CAPABILITY,
    )


def normalize_source_capture(
    source: RgbdSourceCapture,
    observed: SourceDepthModelObservation,
    anchors: Sequence[DepthAnchor] | None = None,
) -> RgbdSimCapture:
    if not isinstance(source, RgbdSourceCapture):
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_INVALID", "source capture is invalid"
        )
    if not isinstance(observed, SourceDepthModelObservation) or not _is_registered_observation(observed):
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_UNOBSERVED",
            "normalization requires a live source-depth observation",
        )
    if anchors is None or not isinstance(anchors, Sequence) or not anchors:
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_BINDING_INVALID",
            "normalization requires the measured anchor contract",
        )
    if not all(isinstance(anchor, DepthAnchor) for anchor in anchors):
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_BINDING_INVALID", "anchor contract is invalid"
        )
    if (
        not all(
            type(error) in {int, float} and math.isfinite(float(error))
            for error in (observed.optical_error_m, observed.ray_range_error_m)
        )
        or observed.scene_sha256 != source.metadata.scene_sha256
        or observed.sequence_id != source.metadata.sequence_id
        or observed.source_digest != _source_digest(source)
        or observed.anchor_digest != _anchor_digest(anchors)
    ):
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_BINDING_INVALID",
            "observation does not belong to this source frame and anchor contract",
        )
    model = observed.model
    if model not in {"optical_z", "ray_range"}:
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_INVALID", "observed source model is invalid"
        )
    if model != source.metadata.expected_source_depth_model:
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_INVALID", "observed model disagrees with scene expectation"
        )
    intrinsics = derive_intrinsics(source.metadata)
    if model == "optical_z":
        depth = np.array(source.source_depth_m, dtype=np.float32, copy=True, order="C")
    else:
        height, width = source.source_depth_m.shape
        u = np.arange(width, dtype=np.float64)[None, :]
        v = np.arange(height, dtype=np.float64)[:, None]
        x_normalized = (u - intrinsics.cx_px) / intrinsics.fx_px
        y_normalized = (v - intrinsics.cy_px) / intrinsics.fy_px
        scale = np.sqrt(1.0 + x_normalized**2 + y_normalized**2)
        depth = np.divide(
            source.source_depth_m.astype(np.float64),
            scale,
            out=np.zeros_like(source.source_depth_m, dtype=np.float64),
            where=source.source_depth_m > 0.0,
        ).astype(np.float32)
    from vision_platform.rgbd.models import RgbdFrame

    frame = RgbdFrame(source.image_bgr, depth)
    return RgbdSimCapture._from_normalization(
        source=source,
        frame=frame,
        intrinsics=intrinsics,
        observed_source_depth_model=model,
        proof_scene_sha256=observed.scene_sha256,
        proof_source_digest=observed.source_digest,
        proof_sequence_id=observed.sequence_id,
        proof_anchor_digest=observed.anchor_digest,
        _capability=_CAPTURE_CAPABILITY,
    )


__all__ = [
    "DepthAnchor",
    "SourceDepthModelObservation",
    "normalize_source_capture",
    "observe_source_depth_model",
]
