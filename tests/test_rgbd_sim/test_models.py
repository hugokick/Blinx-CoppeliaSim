from __future__ import annotations

import json

import numpy as np
import pytest

from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame
from vision_platform.rgbd_sim.errors import RgbdSimContractError
from vision_platform.rgbd_sim.models import (
    RgbdSensorMetadata,
    RgbdSimCapture,
    RgbdSourceCapture,
    capture_to_dict,
    source_capture_to_dict,
)


def _metadata(**overrides: object) -> RgbdSensorMetadata:
    values: dict[str, object] = {
        "sensor_path": "/RgbdLab/CameraRig/RgbdSensor",
        "resolution": (4, 3),
        "near_clip_m": 0.01,
        "far_clip_m": 5.0,
        "perspective_angle_rad": 1.0471975512,
        "scene_path": "simulation/rgbd_lab/BL23_rgbd_lab.ttt",
        "scene_sha256": "a" * 64,
        "sequence_id": 0,
        "timestamp_s": 1700000000.0,
        "expected_source_depth_model": "optical_z",
    }
    values.update(overrides)
    return RgbdSensorMetadata(**values)


def _source(**overrides: object) -> RgbdSourceCapture:
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    depth = np.full((3, 4), 1.25, dtype=np.float32)
    return RgbdSourceCapture(
        metadata=_metadata(**overrides), image_bgr=image, source_depth_m=depth
    )


def test_source_capture_copies_and_seals_raw_arrays() -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.full((3, 4), 1.25, dtype=np.float32)
    source = RgbdSourceCapture(
        metadata=_metadata(), image_bgr=image, source_depth_m=depth
    )
    image[:] = 99
    depth[:] = 2.0
    assert int(source.image_bgr[0, 0, 0]) == 0
    assert float(source.source_depth_m[0, 0]) == pytest.approx(1.25)
    assert source.image_bgr.flags.c_contiguous
    assert source.source_depth_m.flags.c_contiguous
    assert not source.image_bgr.flags.writeable
    assert not source.source_depth_m.flags.writeable
    with pytest.raises(ValueError):
        source.image_bgr.setflags(write=True)
    with pytest.raises(ValueError):
        source.source_depth_m.setflags(write=True)


def test_sim_capture_requires_observed_model_matching_expectation() -> None:
    source = _source()
    frame = RgbdFrame(source.image_bgr, source.source_depth_m)
    intrinsics = CameraIntrinsics(4, 3, 3.464101615, 3.464101615, 1.5, 1.0)
    capture = RgbdSimCapture(
        source=source,
        frame=frame,
        intrinsics=intrinsics,
        observed_source_depth_model="optical_z",
    )
    assert capture.output_depth_model == "optical_z"
    assert capture.observed_source_depth_model == "optical_z"
    with pytest.raises(RgbdSimContractError) as captured:
        RgbdSimCapture(
            source=_source(expected_source_depth_model="ray_range"),
            frame=frame,
            intrinsics=intrinsics,
            observed_source_depth_model="optical_z",
        )
    assert captured.value.code == "RGBD_SIM_CAPTURE_INVALID"


def test_capture_serialization_is_json_native_and_excludes_raw_arrays() -> None:
    source = _source()
    frame = RgbdFrame(source.image_bgr, source.source_depth_m)
    capture = RgbdSimCapture(
        source=source,
        frame=frame,
        intrinsics=CameraIntrinsics(4, 3, 3.464101615, 3.464101615, 1.5, 1.0),
        observed_source_depth_model="optical_z",
    )
    source_payload = source_capture_to_dict(source)
    payload = capture_to_dict(capture)
    assert "image_bgr" not in source_payload
    assert "source_depth_m" not in source_payload
    assert "image_bgr" not in payload
    assert "depth_m" not in payload
    assert json.loads(json.dumps(source_payload, allow_nan=False)) == source_payload
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sensor_path", ""),
        ("resolution", (0, 3)),
        ("resolution", (4.0, 3)),
        ("near_clip_m", 0.0),
        ("far_clip_m", 0.01),
        ("perspective_angle_rad", 0.0),
        ("perspective_angle_rad", float("inf")),
        ("scene_sha256", "A" * 64),
        ("scene_sha256", "a" * 63),
        ("sequence_id", -1),
        ("sequence_id", True),
        ("timestamp_s", float("nan")),
        ("expected_source_depth_model", "range"),
    ],
)
def test_sensor_metadata_rejects_invalid_contract(field: str, value: object) -> None:
    values = {
        "sensor_path": "/RgbdLab/CameraRig/RgbdSensor",
        "resolution": (4, 3),
        "near_clip_m": 0.01,
        "far_clip_m": 5.0,
        "perspective_angle_rad": 1.0,
        "scene_path": "simulation/rgbd_lab/BL23_rgbd_lab.ttt",
        "scene_sha256": "a" * 64,
        "sequence_id": 0,
        "timestamp_s": 1700000000.0,
        "expected_source_depth_model": "optical_z",
    }
    values[field] = value
    with pytest.raises(RgbdSimContractError) as captured:
        RgbdSensorMetadata(**values)
    assert captured.value.code == "RGBD_SIM_METADATA_INVALID"


@pytest.mark.parametrize(
    "depth",
    [
        np.full((3, 4), np.nan, dtype=np.float32),
        np.full((3, 4), np.inf, dtype=np.float32),
        np.full((3, 4), -1.0, dtype=np.float32),
    ],
)
def test_source_capture_rejects_invalid_depth(depth: np.ndarray) -> None:
    with pytest.raises(RgbdSimContractError) as captured:
        RgbdSourceCapture(
            metadata=_metadata(),
            image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
            source_depth_m=depth,
        )
    assert captured.value.code == "RGBD_SIM_SOURCE_INVALID"
