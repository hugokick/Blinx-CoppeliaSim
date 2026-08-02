from __future__ import annotations

import math

import pytest

from vision_platform.rgbd_sim.errors import RgbdSimContractError
from vision_platform.rgbd_sim.intrinsics import derive_intrinsics, expected_intrinsics
from vision_platform.rgbd_sim.models import RgbdSensorMetadata


def _metadata(width: int, height: int, angle: float) -> RgbdSensorMetadata:
    return RgbdSensorMetadata(
        sensor_path="/RgbdLab/CameraRig/RgbdSensor",
        resolution=(width, height),
        near_clip_m=0.01,
        far_clip_m=5.0,
        perspective_angle_rad=angle,
        scene_path="simulation/rgbd_lab/BL23_rgbd_lab.ttt",
        scene_sha256="a" * 64,
        sequence_id=0,
        timestamp_s=1700000000.0,
        expected_source_depth_model="optical_z",
    )


@pytest.mark.parametrize("width,height", [(640, 480), (480, 640), (512, 512)])
def test_expected_intrinsics_follow_perspective_contract(width: int, height: int) -> None:
    angle = math.radians(60.0)
    fx, fy = expected_intrinsics(width, height, angle)
    if width >= height:
        expected = width / (2.0 * math.tan(angle / 2.0))
    else:
        expected = height / (2.0 * math.tan(angle / 2.0))
    assert fx == pytest.approx(expected)
    assert fy == pytest.approx(expected)
    intrinsics = derive_intrinsics(_metadata(width, height, angle))
    assert intrinsics.fx_px == pytest.approx(fx)
    assert intrinsics.fy_px == pytest.approx(fy)
    assert intrinsics.cx_px == pytest.approx((width - 1) / 2.0)
    assert intrinsics.cy_px == pytest.approx((height - 1) / 2.0)


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 480, 1.0),
        (640, 0, 1.0),
        (640, 480, 0.0),
        (640, 480, math.pi),
        (640, 480, math.nan),
        (640, 480, math.inf),
        (640.0, 480, 1.0),
        (640, True, 1.0),
    ],
)
def test_expected_intrinsics_reject_invalid_values(arguments: tuple[object, ...]) -> None:
    with pytest.raises(RgbdSimContractError) as captured:
        expected_intrinsics(*arguments)  # type: ignore[arg-type]
    assert captured.value.code == "RGBD_SIM_INTRINSICS_INVALID"
