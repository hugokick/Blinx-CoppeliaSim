from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision_platform.vision2d.models import PixelScale


@dataclass(frozen=True)
class GeometryMeasurement:
    center_px: tuple[float, float]
    axis_aligned_bbox_px: tuple[int, int, int, int]
    rotated_box_px: tuple[tuple[float, float], ...]
    long_side_px: float
    short_side_px: float
    angle_deg: float
    area_px2: float
    perimeter_px: float
    size_mm: tuple[float, float] | None
    area_mm2: float | None
    perimeter_mm: float | None


def _normalize_axis_angle(angle_deg: float) -> float:
    return ((float(angle_deg) + 90.0) % 180.0) - 90.0


def _scaled_length(vector: np.ndarray, scale: PixelScale) -> float:
    return math.hypot(
        float(vector[0]) * scale.mm_per_pixel_x,
        float(vector[1]) * scale.mm_per_pixel_y,
    )


def measure_geometry(
    contour: np.ndarray,
    *,
    pixel_scale: PixelScale | None,
) -> GeometryMeasurement:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        raise ValueError("contour has degenerate moment")
    center = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    x, y, width, height = cv2.boundingRect(contour)
    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
    edges = [box[(index + 1) % 4] - box[index] for index in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    long_index = int(np.argmax(lengths))
    short_index = int(np.argmin(lengths))
    long_edge = edges[long_index]
    short_edge = edges[short_index]
    long_side = lengths[long_index]
    short_side = lengths[short_index]
    angle = _normalize_axis_angle(
        math.degrees(math.atan2(float(long_edge[1]), float(long_edge[0])))
    )
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    size_mm = None
    area_mm2 = None
    perimeter_mm = None
    if pixel_scale is not None:
        size_mm = (
            _scaled_length(long_edge, pixel_scale),
            _scaled_length(short_edge, pixel_scale),
        )
        area_mm2 = (
            area
            * pixel_scale.mm_per_pixel_x
            * pixel_scale.mm_per_pixel_y
        )
        points = contour.reshape(-1, 2).astype(np.float64)
        closed = np.vstack((points, points[0]))
        perimeter_mm = sum(
            _scaled_length(vector, pixel_scale)
            for vector in np.diff(closed, axis=0)
        )
    return GeometryMeasurement(
        center_px=(round(center[0], 6), round(center[1], 6)),
        axis_aligned_bbox_px=(int(x), int(y), int(width), int(height)),
        rotated_box_px=tuple(
            (round(float(point[0]), 6), round(float(point[1]), 6))
            for point in box
        ),
        long_side_px=long_side,
        short_side_px=short_side,
        angle_deg=angle,
        area_px2=area,
        perimeter_px=perimeter,
        size_mm=size_mm,
        area_mm2=area_mm2,
        perimeter_mm=perimeter_mm,
    )
