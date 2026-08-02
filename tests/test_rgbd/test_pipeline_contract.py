from __future__ import annotations

import json

import numpy as np
import pytest

from tests.test_rgbd.synthetic_factory import (
    make_box,
    make_hole,
    make_plane,
    make_seeded_noise,
    make_step,
    make_tilted_plane,
)
from vision_platform.rgbd.measurement import measure_pixel
from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame
from vision_platform.rgbd.serialization import measurement_to_dict


def test_seeded_factory_is_process_contract_deterministic() -> None:
    first_image, first_depth = make_seeded_noise(16, 12, seed=1707)
    second_image, second_depth = make_seeded_noise(16, 12, seed=1707)
    assert np.array_equal(first_image, second_image)
    assert np.array_equal(first_depth, second_depth)


def test_tilted_plane_measurement_matches_independent_equation() -> None:
    image, depth = make_tilted_plane(16, 12, base_m=0.6, du_m=0.001, dv_m=0.002)
    frame = RgbdFrame(image, depth)
    intrinsics = CameraIntrinsics(16, 12, 100.0, 120.0, 7.5, 5.5)
    result = measure_pixel(frame, intrinsics, u_px=10, v_px=8, window_size=3)
    payload = measurement_to_dict(result)
    expected_z = 0.6 + 0.001 * 10 + 0.002 * 8
    assert payload["depth_m"] == pytest.approx(expected_z, abs=1e-5)
    assert payload["point_camera_m"] == pytest.approx(
        [(10 - 7.5) * expected_z / 100.0, (8 - 5.5) * expected_z / 120.0, expected_z],
        abs=1e-5,
    )
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_all_synthetic_surfaces_feed_the_same_measurement_contract() -> None:
    intrinsics = CameraIntrinsics(12, 10, 100.0, 120.0, 5.5, 4.5)
    for factory in (make_plane, make_step, make_box, make_tilted_plane):
        image, depth = factory(12, 10)
        payload = measurement_to_dict(
            measure_pixel(RgbdFrame(image, depth), intrinsics, u_px=1, v_px=1)
        )
        assert payload["status"] == "PASS"
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_hole_and_boundaries_use_defined_failure_and_success_contracts() -> None:
    image, depth = make_hole(9, 9)
    frame = RgbdFrame(image, depth)
    intrinsics = CameraIntrinsics(9, 9, 100.0, 100.0, 4.0, 4.0)
    center = measurement_to_dict(measure_pixel(frame, intrinsics, u_px=4, v_px=4))
    assert center["status"] == "NO_VALID_DEPTH"
    for u_px, v_px in ((0, 0), (8, 0), (0, 8), (8, 8)):
        boundary = measurement_to_dict(
            measure_pixel(frame, intrinsics, u_px=u_px, v_px=v_px, window_size=3)
        )
        assert boundary["status"] == "PASS"
        assert boundary["depth_m"] == pytest.approx(1.0)
