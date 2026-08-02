"""Bounded metric-depth sampling for validated RGB-D frames."""

from __future__ import annotations

import numpy as np

from .errors import RgbdContractError
from .models import DepthSample, RgbdFrame


def sample_depth(
    frame: RgbdFrame,
    u_px: int,
    v_px: int,
    *,
    window_size: int = 1,
    min_valid_count: int = 1,
) -> DepthSample:
    if not isinstance(frame, RgbdFrame):
        raise RgbdContractError("RGBD_SAMPLE_INVALID", "frame is invalid")
    if type(u_px) is not int or type(v_px) is not int:
        raise RgbdContractError(
            "RGBD_SAMPLE_INVALID", "pixel indices must be built-in integers"
        )
    height, width = frame.depth_m.shape
    if not 0 <= u_px < width or not 0 <= v_px < height:
        raise RgbdContractError("RGBD_SAMPLE_INVALID", "pixel is out of bounds")
    if (
        type(window_size) is not int
        or window_size not in {1, 3, 5, 7, 9}
        or type(min_valid_count) is not int
        or min_valid_count <= 0
        or min_valid_count > window_size * window_size
    ):
        raise RgbdContractError(
            "RGBD_SAMPLE_INVALID", "window or valid-count limit is invalid"
        )
    radius = window_size // 2
    left = max(0, u_px - radius)
    right = min(width, u_px + radius + 1)
    top = max(0, v_px - radius)
    bottom = min(height, v_px + radius + 1)
    window = frame.depth_m[top:bottom, left:right]
    valid = window[window > 0.0]
    valid_count = int(valid.size)
    if valid_count < min_valid_count:
        return DepthSample(
            "NO_VALID_DEPTH",
            u_px,
            v_px,
            window_size,
            valid_count,
            None,
            "NO_VALID_DEPTH",
        )
    return DepthSample(
        "PASS",
        u_px,
        v_px,
        window_size,
        valid_count,
        float(np.median(valid)),
        None,
    )


__all__ = ["sample_depth"]
