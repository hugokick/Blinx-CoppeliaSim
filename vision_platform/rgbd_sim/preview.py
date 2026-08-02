"""Pure, deterministic preview helpers for the D1-01 RGB-D adapter."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import RgbdSimContractError


def _validate_bgr(image_bgr: object) -> np.ndarray:
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.shape[0] <= 0
        or image_bgr.shape[1] <= 0
    ):
        raise RgbdSimContractError(
            "RGBD_SIM_PREVIEW_INVALID", "image_bgr must be non-empty uint8 HxWx3"
        )
    return image_bgr


def copy_bgr_preview(image_bgr: np.ndarray) -> np.ndarray:
    """Return a C-contiguous copy without exposing the caller's image storage."""

    source = _validate_bgr(image_bgr)
    return np.array(source, dtype=np.uint8, copy=True, order="C")


def _validate_depth(depth_m: object) -> np.ndarray:
    if (
        not isinstance(depth_m, np.ndarray)
        or depth_m.dtype != np.float32
        or depth_m.ndim != 2
        or depth_m.shape[0] <= 0
        or depth_m.shape[1] <= 0
        or not np.all(np.isfinite(depth_m))
        or np.any(depth_m < 0.0)
    ):
        raise RgbdSimContractError(
            "RGBD_SIM_PREVIEW_INVALID",
            "depth_m must be non-empty finite non-negative float32 HxW",
        )
    return depth_m


def _bound(value: object, name: str) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise RgbdSimContractError(
            "RGBD_SIM_PREVIEW_INVALID", f"{name} must be finite or None"
        )
    result = float(value)
    if result < 0.0:
        raise RgbdSimContractError(
            "RGBD_SIM_PREVIEW_INVALID", f"{name} must be non-negative"
        )
    return result


def colorize_depth(
    depth_m: np.ndarray,
    *,
    minimum_m: float | None = 0.0,
    maximum_m: float | None = None,
) -> np.ndarray:
    """Colorize metric depth into deterministic BGR ``uint8 HxWx3`` pixels.

    Zero is reserved for invalid depth and is rendered magenta. Valid values
    are clipped to the requested range, or to the observed finite range when
    a bound is omitted. The input is never modified.
    """

    source = _validate_depth(depth_m)
    lower = _bound(minimum_m, "minimum_m")
    upper = _bound(maximum_m, "maximum_m")
    valid = source > 0.0
    valid_values = source[valid].astype(np.float64, copy=False)

    if lower is None:
        lower = float(np.min(valid_values)) if valid_values.size else 0.0
    if upper is None:
        upper = float(np.max(valid_values)) if valid_values.size else max(lower, 1.0)
    if upper < lower or (maximum_m is not None and upper <= lower):
        raise RgbdSimContractError(
            "RGBD_SIM_PREVIEW_INVALID", "maximum_m must be greater than minimum_m"
        )

    # OpenCV's colormap is applied only to an independent uint8 work buffer.
    output = np.empty((*source.shape, 3), dtype=np.uint8)
    output[...] = (255, 0, 255)
    if not valid_values.size:
        return output

    clipped = np.clip(source.astype(np.float64, copy=False), lower, upper)
    span = upper - lower
    if span <= 0.0:
        scaled = np.full(source.shape, 127, dtype=np.uint8)
    else:
        scaled_float = ((clipped - lower) / span) * 255.0
        scaled = np.rint(np.clip(scaled_float, 0.0, 255.0)).astype(np.uint8)

    import cv2

    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    output[valid] = colored[valid]
    return output


def depth_summary(depth_m: np.ndarray) -> dict[str, Any]:
    """Return a JSON-native summary without embedding the depth array."""

    source = _validate_depth(depth_m)
    valid = source[source > 0.0]
    if valid.size == 0:
        return {"valid_count": 0, "invalid_count": int(source.size)}
    return {
        "valid_count": int(valid.size),
        "invalid_count": int(source.size - valid.size),
        "min_depth_m": float(np.min(valid)),
        "median_depth_m": float(np.median(valid)),
        "max_depth_m": float(np.max(valid)),
    }


# Small aliases keep the package vocabulary convenient at call sites while
# retaining one implementation and one contract.
copy_bgr = copy_bgr_preview
depth_to_preview = colorize_depth


__all__ = [
    "colorize_depth",
    "copy_bgr",
    "copy_bgr_preview",
    "depth_summary",
    "depth_to_preview",
]
