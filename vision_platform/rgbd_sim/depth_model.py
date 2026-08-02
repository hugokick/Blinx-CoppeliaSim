"""Independent source-depth model observation and optical-Z normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import RgbdSimContractError
from .intrinsics import derive_intrinsics
from .models import RgbdSimCapture, RgbdSourceCapture


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


@dataclass(frozen=True)
class SourceDepthModelObservation:
    model: str
    optical_error_m: float
    ray_range_error_m: float

    def __post_init__(self) -> None:
        if self.model not in {"optical_z", "ray_range"}:
            raise RgbdSimContractError(
                "RGBD_SIM_DEPTH_MODEL_INVALID", "observed source model is invalid"
            )


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
    return SourceDepthModelObservation(
        "optical_z" if optical_ok else "ray_range",
        max(optical_errors),
        max(ray_errors),
    )


def normalize_source_capture(
    source: RgbdSourceCapture,
    observed: SourceDepthModelObservation | str,
) -> RgbdSimCapture:
    if not isinstance(source, RgbdSourceCapture):
        raise RgbdSimContractError(
            "RGBD_SIM_DEPTH_MODEL_INVALID", "source capture is invalid"
        )
    model = observed.model if isinstance(observed, SourceDepthModelObservation) else observed
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
    return RgbdSimCapture(
        source=source,
        frame=frame,
        intrinsics=intrinsics,
        observed_source_depth_model=model,
    )


__all__ = [
    "DepthAnchor",
    "SourceDepthModelObservation",
    "normalize_source_capture",
    "observe_source_depth_model",
]
