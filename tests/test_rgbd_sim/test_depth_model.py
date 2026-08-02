from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vision_platform.rgbd_sim.depth_model import (
    DepthAnchor,
    SourceDepthModelObservation,
    normalize_source_capture,
    observe_source_depth_model,
)
from vision_platform.rgbd_sim.errors import RgbdSimContractError
from vision_platform.rgbd_sim.models import RgbdSensorMetadata, RgbdSourceCapture


def _source(depth: np.ndarray, *, expected: str = "optical_z") -> RgbdSourceCapture:
    metadata = RgbdSensorMetadata(
        sensor_path="/RgbdLab/CameraRig/RgbdSensor",
        resolution=(depth.shape[1], depth.shape[0]),
        near_clip_m=0.01,
        far_clip_m=5.0,
        perspective_angle_rad=1.0471975512,
        scene_path="simulation/rgbd_lab/BL23_rgbd_lab.ttt",
        scene_sha256="a" * 64,
        sequence_id=0,
        timestamp_s=1700000000.0,
        expected_source_depth_model=expected,
    )
    return RgbdSourceCapture(
        metadata=metadata,
        image_bgr=np.zeros((*depth.shape, 3), dtype=np.uint8),
        source_depth_m=np.asarray(depth, dtype=np.float32),
    )


def test_observation_classifies_optical_z_from_center_and_off_axis_anchors() -> None:
    source = _source(np.full((5, 5), 1.0, dtype=np.float32))
    anchors = [
        DepthAnchor(2, 2, optical_z_m=1.0, ray_range_m=1.0, tolerance_m=1e-4),
        DepthAnchor(0, 2, optical_z_m=1.0, ray_range_m=1.1, tolerance_m=1e-4),
        DepthAnchor(4, 2, optical_z_m=1.0, ray_range_m=1.1, tolerance_m=1e-4),
    ]
    observation = observe_source_depth_model(source, anchors)
    assert isinstance(observation, SourceDepthModelObservation)
    assert observation.model == "optical_z"


def test_observation_classifies_ray_range_and_normalizes_to_optical_z() -> None:
    source_depth = np.full((5, 5), 1.0, dtype=np.float32)
    source_depth[2, 0] = 1.1
    source_depth[2, 4] = 1.1
    source = _source(source_depth, expected="ray_range")
    anchors = [
        DepthAnchor(2, 2, optical_z_m=1.0, ray_range_m=1.0, tolerance_m=1e-4),
        DepthAnchor(0, 2, optical_z_m=1.0, ray_range_m=1.1, tolerance_m=1e-4),
        DepthAnchor(4, 2, optical_z_m=1.0, ray_range_m=1.1, tolerance_m=1e-4),
    ]
    observation = observe_source_depth_model(source, anchors)
    capture = normalize_source_capture(source, observation)
    assert observation.model == "ray_range"
    assert capture.output_depth_model == "optical_z"
    assert capture.frame.depth_m[2, 0] == pytest.approx(1.0, abs=2e-3)
    assert capture.frame.depth_m[2, 4] == pytest.approx(1.0, abs=2e-3)
    np.testing.assert_array_equal(source.source_depth_m, source_depth)


def test_observation_rejects_ambiguous_or_unmatched_geometry() -> None:
    source = _source(np.full((3, 3), 1.0, dtype=np.float32))
    ambiguous = [DepthAnchor(1, 1, 1.0, 1.0, 1e-4)]
    with pytest.raises(RgbdSimContractError) as captured:
        observe_source_depth_model(source, ambiguous)
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_INVALID"
    unmatched = [DepthAnchor(1, 1, 1.5, 2.0, 1e-4)]
    with pytest.raises(RgbdSimContractError) as captured:
        observe_source_depth_model(source, unmatched)
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_INVALID"


def test_normalization_requires_observation_to_match_scene_expectation() -> None:
    source = _source(np.ones((3, 3), dtype=np.float32), expected="ray_range")
    with pytest.raises(RgbdSimContractError) as captured:
        normalize_source_capture(source, "optical_z")
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_INVALID"
