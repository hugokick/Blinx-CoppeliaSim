from __future__ import annotations

import math

import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.geometry import deproject_pixel
from vision_platform.rgbd.models import CameraIntrinsics


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(640, 480, 400.0, 500.0, 319.5, 239.5)


def test_principal_point_has_zero_lateral_coordinates() -> None:
    point = deproject_pixel(
        _intrinsics(), u_px=319.5, v_px=239.5, depth_m=2.0
    )
    assert (point.x_m, point.y_m, point.z_m) == pytest.approx((0.0, 0.0, 2.0))


def test_asymmetric_intrinsics_match_analytic_point() -> None:
    point = deproject_pixel(
        _intrinsics(), u_px=419.5, v_px=339.5, depth_m=2.0
    )
    assert (point.x_m, point.y_m, point.z_m) == pytest.approx((0.5, 0.4, 2.0))
    assert all(type(value) is float for value in (point.x_m, point.y_m, point.z_m))


def test_top_left_pixel_uses_integer_center_without_half_offset() -> None:
    point = deproject_pixel(
        CameraIntrinsics(3, 3, 2.0, 2.0, 1.0, 1.0),
        u_px=0.0,
        v_px=0.0,
        depth_m=1.0,
    )
    assert (point.x_m, point.y_m, point.z_m) == pytest.approx((-0.5, -0.5, 1.0))


@pytest.mark.parametrize(
    ("u_px", "v_px", "expected"),
    [
        (0.0, 0.0, (-1.5975, -0.958, 2.0)),
        (639.0, 0.0, (1.5975, -0.958, 2.0)),
        (0.0, 479.0, (-1.5975, 0.958, 2.0)),
        (639.0, 479.0, (1.5975, 0.958, 2.0)),
    ],
)
def test_four_image_corners_match_analytic_deprojection(
    u_px: float, v_px: float, expected: tuple[float, float, float]
) -> None:
    point = deproject_pixel(_intrinsics(), u_px=u_px, v_px=v_px, depth_m=2.0)
    assert (point.x_m, point.y_m, point.z_m) == pytest.approx(expected, abs=1e-7)


def test_deprojection_is_deterministic_for_repeated_call() -> None:
    first = deproject_pixel(_intrinsics(), u_px=17.0, v_px=23.0, depth_m=1.25)
    second = deproject_pixel(_intrinsics(), u_px=17.0, v_px=23.0, depth_m=1.25)
    assert (first.x_m, first.y_m, first.z_m) == (
        second.x_m,
        second.y_m,
        second.z_m,
    )


@pytest.mark.parametrize(
    ("u_px", "v_px", "depth_m"),
    [
        (-0.1, 0.0, 1.0),
        (640.0, 0.0, 1.0),
        (0.0, 480.0, 1.0),
        (True, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, math.inf),
        (0.0, 0.0, math.nan),
    ],
)
def test_deprojection_rejects_invalid_values(
    u_px: object, v_px: object, depth_m: object
) -> None:
    with pytest.raises(RgbdContractError) as captured:
        deproject_pixel(_intrinsics(), u_px=u_px, v_px=v_px, depth_m=depth_m)
    assert captured.value.code == "RGBD_DEPROJECTION_INVALID"
