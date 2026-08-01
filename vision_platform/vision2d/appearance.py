from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision_platform.vision2d.models import Vision2DConfig


@dataclass(frozen=True)
class AppearanceMeasurement:
    color: str
    shape: str
    vertex_count: int
    circularity: float
    aspect_ratio: float


def _color_label(
    hsv: np.ndarray,
    contour: np.ndarray,
    *,
    saturation_min: int,
) -> str:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        return "unknown"
    hue = float(np.median(pixels[:, 0]))
    saturation = float(np.median(pixels[:, 1]))
    if saturation < float(saturation_min):
        return "unknown"
    if hue <= 10.0 or hue >= 170.0:
        return "red"
    if 15.0 <= hue <= 40.0:
        return "yellow"
    if 40.0 < hue <= 90.0:
        return "green"
    if 90.0 < hue <= 140.0:
        return "blue"
    return "unknown"


def measure_appearance(
    hsv: np.ndarray,
    contour: np.ndarray,
    config: Vision2DConfig,
) -> AppearanceMeasurement:
    perimeter = float(cv2.arcLength(contour, True))
    area = float(cv2.contourArea(contour))
    approximation = cv2.approxPolyDP(
        contour,
        config.polygon_epsilon_ratio * perimeter,
        True,
    )
    vertex_count = len(approximation)
    _, (width, height), _ = cv2.minAreaRect(contour)
    longer = max(float(width), float(height))
    shorter = min(float(width), float(height))
    aspect_ratio = shorter / longer if longer > 0.0 else 0.0
    circularity = (
        4.0 * math.pi * area / (perimeter * perimeter)
        if perimeter > 0.0
        else 0.0
    )
    if vertex_count == 3:
        shape = "triangle"
    elif vertex_count == 4:
        shape = "square" if aspect_ratio >= config.square_aspect_min else "rectangle"
    elif vertex_count >= 5 and circularity >= config.circle_circularity_min:
        shape = "circle"
    elif vertex_count >= 5:
        shape = "polygon"
    else:
        shape = "unknown"
    return AppearanceMeasurement(
        color=_color_label(
            hsv,
            contour,
            saturation_min=config.saturation_min,
        ),
        shape=shape,
        vertex_count=vertex_count,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
    )
