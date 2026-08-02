"""Original deterministic in-memory RGB-D fixtures for D1 tests."""

from __future__ import annotations

import numpy as np


def _base(width: int, height: int, depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("synthetic size must be positive integers")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (32, 96, 160)
    depth = np.full((height, width), depth_m, dtype=np.float32)
    return image, depth


def make_plane(width: int, height: int, depth_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    return _base(width, height, depth_m)


def make_step(
    width: int, height: int, left_m: float = 0.8, right_m: float = 1.2
) -> tuple[np.ndarray, np.ndarray]:
    image, depth = _base(width, height, left_m)
    depth[:, width // 2 :] = np.float32(right_m)
    return image, depth


def make_box(
    width: int, height: int, plane_m: float = 1.0, box_m: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    image, depth = _base(width, height, plane_m)
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    image[y0:y1, x0:x1] = (20, 180, 40)
    depth[y0:y1, x0:x1] = np.float32(box_m)
    return image, depth


def make_tilted_plane(
    width: int,
    height: int,
    *,
    base_m: float = 0.6,
    du_m: float = 0.001,
    dv_m: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    image, _ = _base(width, height, base_m)
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    depth = np.asarray(base_m + du_m * u + dv_m * v, dtype=np.float32)
    return image, depth


def make_hole(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    image, depth = _base(width, height, 1.0)
    depth[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 0.0
    return image, depth


def make_seeded_noise(
    width: int, height: int, *, seed: int, sigma_m: float = 0.002
) -> tuple[np.ndarray, np.ndarray]:
    image, depth = _base(width, height, 1.0)
    rng = np.random.default_rng(seed)
    depth += rng.normal(0.0, sigma_m, depth.shape).astype(np.float32)
    return image, depth


__all__ = [
    "make_box",
    "make_hole",
    "make_plane",
    "make_seeded_noise",
    "make_step",
    "make_tilted_plane",
]
