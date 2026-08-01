import cv2
import numpy as np
import pytest

from vision_platform.vision2d.geometry import measure_geometry
from vision_platform.vision2d.models import PixelScale


def _rectangle_contour(
    center=(100.0, 80.0), size=(60.0, 30.0), angle=30.0
):
    points = cv2.boxPoints((center, size, angle))
    return points.reshape(-1, 1, 2).astype(np.float32)


def _axis_error(actual: float, expected: float) -> float:
    direct = abs(actual - expected) % 180.0
    return min(direct, 180.0 - direct)


def test_geometry_measures_center_edges_area_perimeter_and_axis_angle():
    geometry = measure_geometry(_rectangle_contour(), pixel_scale=None)

    assert geometry.center_px == pytest.approx((100.0, 80.0), abs=1e-3)
    assert geometry.long_side_px == pytest.approx(60.0, abs=1e-3)
    assert geometry.short_side_px == pytest.approx(30.0, abs=1e-3)
    assert _axis_error(geometry.angle_deg, 30.0) <= 1e-3
    assert geometry.area_px2 == pytest.approx(1800.0, rel=1e-3)
    assert geometry.perimeter_px == pytest.approx(180.0, rel=1e-3)
    assert geometry.size_mm is None
    assert geometry.area_mm2 is None
    assert geometry.perimeter_mm is None


def test_geometry_uses_anisotropic_scale_for_edges_area_and_perimeter():
    contour = _rectangle_contour(size=(40.0, 20.0), angle=0.0)
    scale = PixelScale(mm_per_pixel_x=0.2, mm_per_pixel_y=0.5)

    geometry = measure_geometry(contour, pixel_scale=scale)

    assert geometry.size_mm == pytest.approx((8.0, 10.0), abs=1e-3)
    assert geometry.area_mm2 == pytest.approx(80.0, rel=1e-3)
    assert geometry.perimeter_mm == pytest.approx(36.0, rel=1e-3)
