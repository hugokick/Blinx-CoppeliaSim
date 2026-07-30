import numpy as np
import pytest

from vision_platform.calibration.affine import AffineCalibration
from vision_platform.errors import CalibrationError


def test_three_point_affine_maps_pixel_to_world():
    calibration = AffineCalibration.fit(
        pixel_points=[(0, 0), (100, 0), (0, 100)],
        world_points_mm=[(10, 20), (110, 20), (10, 220)],
        image_size=(640, 480),
        plane_z_mm=30,
    )

    assert np.allclose(
        calibration.pixel_to_world((25, 50)),
        (35, 120),
        atol=1e-6,
    )


def test_affine_rejects_collinear_points():
    with pytest.raises(CalibrationError, match="collinear"):
        AffineCalibration.fit(
            pixel_points=[(0, 0), (10, 10), (20, 20)],
            world_points_mm=[(0, 0), (10, 0), (20, 0)],
            image_size=(100, 100),
            plane_z_mm=0,
        )


def test_affine_uses_least_squares_for_more_than_three_points():
    pixels = [(0, 0), (100, 0), (0, 100), (100, 100), (50, 25)]
    worlds = [(5 + 2 * x, -10 + 3 * y) for x, y in pixels]

    calibration = AffineCalibration.fit(
        pixels,
        worlds,
        image_size=(200, 150),
        plane_z_mm=15,
    )

    assert np.allclose(calibration.pixel_to_world((20, 30)), (45, 80))


def test_evaluate_reports_rms_and_max_error():
    calibration = AffineCalibration.fit(
        [(0, 0), (100, 0), (0, 100)],
        [(0, 0), (100, 0), (0, 100)],
        image_size=(100, 100),
        plane_z_mm=20,
    )

    metrics = calibration.evaluate(
        [
            ((10, 10), (13, 14)),
            ((20, 20), (20, 20)),
        ]
    )

    assert metrics.rms_error_mm == pytest.approx(np.sqrt(12.5))
    assert metrics.max_error_mm == pytest.approx(5.0)
    assert metrics.sample_count == 2


def test_calibration_rejects_wrong_frame_size():
    calibration = AffineCalibration.fit(
        [(0, 0), (100, 0), (0, 100)],
        [(0, 0), (100, 0), (0, 100)],
        image_size=(640, 480),
        plane_z_mm=20,
    )

    with pytest.raises(CalibrationError, match="image size"):
        calibration.validate_image_size((800, 600))


def test_fit_rejects_different_point_counts():
    with pytest.raises(CalibrationError, match="same number"):
        AffineCalibration.fit(
            [(0, 0), (100, 0), (0, 100)],
            [(0, 0), (100, 0)],
            image_size=(640, 480),
            plane_z_mm=20,
        )
