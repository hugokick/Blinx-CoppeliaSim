from __future__ import annotations

import cv2
import numpy as np

from vision_platform.vision2d.models import RejectedTarget, Vision2DConfig


def _touches_border(
    contour: np.ndarray,
    *,
    width: int,
    height: int,
    margin: int,
) -> bool:
    points = contour.reshape(-1, 2)
    return bool(
        np.any(points[:, 0] <= margin)
        or np.any(points[:, 0] >= width - 1 - margin)
        or np.any(points[:, 1] <= margin)
        or np.any(points[:, 1] >= height - 1 - margin)
    )


def _center_or_none(contour: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        return None
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def segment_contours(
    cleaned_mask: np.ndarray,
    config: Vision2DConfig,
) -> tuple[tuple[np.ndarray, ...], tuple[RejectedTarget, ...]]:
    if cleaned_mask.ndim != 2 or cleaned_mask.dtype != np.uint8:
        raise ValueError("cleaned_mask must be a uint8 HxW image")
    height, width = cleaned_mask.shape
    image_area = float(width * height)
    minimum = image_area * config.min_area_ratio
    maximum = image_area * config.max_area_ratio
    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    accepted: list[np.ndarray] = []
    rejected: list[RejectedTarget] = []
    for index, contour in enumerate(contours, start=1):
        area = float(cv2.contourArea(contour))
        code = None
        reason = None
        if len(contour) < 3:
            code = "INSUFFICIENT_CONTOUR_POINTS"
            reason = "candidate contour has fewer than three points"
        elif area < minimum:
            code = "AREA_TOO_SMALL"
            reason = "candidate area is below configured minimum"
        elif area > maximum:
            code = "AREA_TOO_LARGE"
            reason = "candidate area is above configured maximum"
        elif _touches_border(
            contour,
            width=width,
            height=height,
            margin=config.border_margin_px,
        ):
            code = "TOUCHES_IMAGE_BORDER"
            reason = "candidate touches the configured image border"
        elif _center_or_none(contour) is None:
            code = "DEGENERATE_MOMENT"
            reason = "candidate contour has a degenerate spatial moment"
        else:
            _, (box_width, box_height), _ = cv2.minAreaRect(contour)
            if box_width <= 0.0 or box_height <= 0.0:
                code = "DEGENERATE_ROTATED_BOX"
                reason = "candidate rotated box has a zero-length edge"
        if code is None:
            accepted.append(cv2.approxPolyDP(contour, 0.5, True))
        else:
            rejected.append(
                RejectedTarget(
                    candidate_index=index,
                    center_px=_center_or_none(contour),
                    area_px2=area,
                    code=code,
                    reason=reason,
                    contour=contour.copy(),
                )
            )
    return tuple(accepted), tuple(rejected)
