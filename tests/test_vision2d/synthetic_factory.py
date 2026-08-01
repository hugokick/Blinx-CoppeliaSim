from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


COLORS_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "gray": (160, 160, 160),
}


@dataclass(frozen=True)
class SyntheticObject:
    shape: str
    color: str
    center_px: tuple[int, int]
    size_px: tuple[int, int]
    angle_deg: float


def make_scene(
    image_size: tuple[int, int],
    objects: tuple[SyntheticObject, ...],
    *,
    background_bgr: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, tuple[SyntheticObject, ...]]:
    width, height = image_size
    image = np.full((height, width, 3), background_bgr, dtype=np.uint8)
    for item in objects:
        color = COLORS_BGR[item.color]
        if item.shape in {"rectangle", "square"}:
            rectangle = (item.center_px, item.size_px, item.angle_deg)
            points = np.rint(cv2.boxPoints(rectangle)).astype(np.int32)
            cv2.fillConvexPoly(image, points, color)
        elif item.shape == "circle":
            cv2.circle(image, item.center_px, item.size_px[0] // 2, color, -1)
        elif item.shape == "triangle":
            radius = item.size_px[0] / 2.0
            angles = np.deg2rad(np.array([-90.0, 30.0, 150.0]) + item.angle_deg)
            points = np.column_stack(
                (
                    item.center_px[0] + radius * np.cos(angles),
                    item.center_px[1] + radius * np.sin(angles),
                )
            )
            cv2.fillConvexPoly(image, np.rint(points).astype(np.int32), color)
        else:
            raise ValueError(f"unsupported synthetic shape: {item.shape}")
    return image, objects


def add_gaussian_noise(
    image_bgr: np.ndarray,
    *,
    sigma: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image_bgr.shape)
    return np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def adjust_brightness(image_bgr: np.ndarray, *, factor: float) -> np.ndarray:
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("factor must be finite positive")
    return np.clip(
        image_bgr.astype(np.float32) * float(factor),
        0,
        255,
    ).astype(np.uint8)
