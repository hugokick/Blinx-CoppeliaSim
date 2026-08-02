from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.measurement import measure_pixel
from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame
from vision_platform.rgbd.serialization import measurement_to_dict
from vision_platform.rgbd.transforms import RigidTransform


def _inputs(depth_value: float) -> tuple[RgbdFrame, CameraIntrinsics]:
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    depth = np.full((3, 3), depth_value, dtype=np.float32)
    return RgbdFrame(image, depth), CameraIntrinsics(3, 3, 2.0, 2.0, 1.0, 1.0)


def test_measurement_serializes_camera_and_target_points() -> None:
    frame, intrinsics = _inputs(2.0)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    result = measure_pixel(
        frame,
        intrinsics,
        u_px=2,
        v_px=1,
        camera_frame_id="camera_optical",
        target_frame_id="world",
        camera_to_target=RigidTransform.from_matrix(matrix),
    )
    payload = measurement_to_dict(result)
    assert payload == {
        "schema_version": 1,
        "status": "PASS",
        "pixel_px": [2, 1],
        "window_size": 1,
        "valid_count": 1,
        "depth_m": 2.0,
        "camera_frame_id": "camera_optical",
        "target_frame_id": "world",
        "point_camera_m": [1.0, 0.0, 2.0],
        "point_target_m": [2.0, 2.0, 5.0],
        "failure_code": None,
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    payload["point_camera_m"][0] = 99.0  # type: ignore[index]
    assert measurement_to_dict(result)["point_camera_m"][0] == 1.0  # type: ignore[index]


def test_missing_depth_serializes_null_points() -> None:
    frame, intrinsics = _inputs(0.0)
    payload = measurement_to_dict(
        measure_pixel(frame, intrinsics, u_px=1, v_px=1)
    )
    assert payload["status"] == "NO_VALID_DEPTH"
    assert payload["depth_m"] is None
    assert payload["point_camera_m"] is None
    assert payload["point_target_m"] is None
    assert payload["failure_code"] == "NO_VALID_DEPTH"


def test_target_frame_and_transform_must_be_paired() -> None:
    frame, intrinsics = _inputs(1.0)
    with pytest.raises(RgbdContractError) as captured:
        measure_pixel(frame, intrinsics, u_px=1, v_px=1, target_frame_id="world")
    assert captured.value.code == "RGBD_MEASUREMENT_INVALID"


def test_measurement_rejects_frame_intrinsics_size_mismatch() -> None:
    frame, _ = _inputs(1.0)
    mismatched = CameraIntrinsics(4, 3, 2.0, 2.0, 1.5, 1.0)
    with pytest.raises(RgbdContractError) as captured:
        measure_pixel(frame, mismatched, u_px=1, v_px=1)
    assert captured.value.code == "RGBD_MEASUREMENT_INVALID"


def test_invalid_transform_is_rejected_even_when_depth_is_missing() -> None:
    frame, intrinsics = _inputs(0.0)
    with pytest.raises(RgbdContractError) as captured:
        measure_pixel(
            frame, intrinsics, u_px=1, v_px=1,
            target_frame_id="world", camera_to_target=object(),
        )
    assert captured.value.code == "RGBD_MEASUREMENT_INVALID"


@pytest.mark.parametrize(
    "invalid",
    [None, np.float32(1.0), np.zeros(1), Path("frame.json"), b"bytes", {}],
)
def test_serializer_rejects_arbitrary_objects(invalid: object) -> None:
    with pytest.raises(RgbdContractError) as captured:
        measurement_to_dict(invalid)
    assert captured.value.code == "RGBD_SERIALIZATION_INVALID"
