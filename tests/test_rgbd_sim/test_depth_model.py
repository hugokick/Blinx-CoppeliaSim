from __future__ import annotations

import copy
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
    capture = normalize_source_capture(source, observation, anchors)
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
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_UNOBSERVED"


def test_normalization_rejects_bare_model_string_without_observation_proof() -> None:
    source = _source(np.ones((3, 3), dtype=np.float32), expected="optical_z")
    with pytest.raises(RgbdSimContractError) as captured:
        normalize_source_capture(source, "optical_z")
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_UNOBSERVED"


def test_observation_is_bound_to_scene_source_sequence_and_anchor_contract() -> None:
    source = _source(np.full((5, 5), 1.0, dtype=np.float32))
    anchors = [
        DepthAnchor(2, 2, 1.0, 1.0, 1e-4),
        DepthAnchor(0, 2, 1.0, 1.1, 1e-4),
        DepthAnchor(4, 2, 1.0, 1.1, 1e-4),
    ]
    observation = observe_source_depth_model(source, anchors)

    assert observation.scene_sha256 == source.metadata.scene_sha256
    assert observation.sequence_id == source.metadata.sequence_id
    assert len(observation.source_digest) == 64
    assert len(observation.anchor_digest) == 64
    normalize_source_capture(source, observation, anchors)

    changed_metadata = source.metadata.__class__(
        **{
            **source.metadata.__dict__,
            "sequence_id": source.metadata.sequence_id + 1,
        }
    )
    changed_frame = _source(np.full((5, 5), 1.0, dtype=np.float32), expected="optical_z")
    changed_frame = type(changed_frame)(changed_metadata, changed_frame.image_bgr, changed_frame.source_depth_m)
    with pytest.raises(RgbdSimContractError) as captured:
        normalize_source_capture(changed_frame, observation, anchors)
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_BINDING_INVALID"

    changed_depth = np.full((5, 5), 1.0, dtype=np.float32)
    changed_depth[0, 0] = 1.25
    changed_frame = _source(changed_depth, expected="optical_z")
    with pytest.raises(RgbdSimContractError) as captured:
        normalize_source_capture(changed_frame, observation, anchors)
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_BINDING_INVALID"

    with pytest.raises(RgbdSimContractError) as captured:
        normalize_source_capture(source, observe_source_depth_model(source, anchors[:-1]), anchors)
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_BINDING_INVALID"


def test_observation_constructor_and_copies_cannot_forge_a_valid_proof() -> None:
    with pytest.raises(RgbdSimContractError) as captured:
        SourceDepthModelObservation("optical_z", 0.0, 0.0)
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_UNOBSERVED"
    with pytest.raises(RgbdSimContractError) as captured:
        SourceDepthModelObservation._from_measurement(
            model="optical_z",
            optical_error_m=0.0,
            ray_range_error_m=0.0,
            scene_sha256="a" * 64,
            source_digest="b" * 64,
            sequence_id=0,
            anchor_digest="c" * 64,
        )
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_UNOBSERVED"

    source = _source(np.full((5, 5), 1.0, dtype=np.float32))
    anchors = [
        DepthAnchor(2, 2, 1.0, 1.0, 1e-4),
        DepthAnchor(0, 2, 1.0, 1.1, 1e-4),
        DepthAnchor(4, 2, 1.0, 1.1, 1e-4),
    ]
    observation = observe_source_depth_model(source, anchors)
    forged = copy.copy(observation)
    with pytest.raises(RgbdSimContractError) as captured:
        normalize_source_capture(source, forged)
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_UNOBSERVED"


def test_observation_rejects_non_finite_error_before_normalization() -> None:
    source = _source(np.full((5, 5), 1.0, dtype=np.float32))
    anchors = [
        DepthAnchor(2, 2, 1.0, 1.0, 1e-4),
        DepthAnchor(0, 2, 1.0, 1.1, 1e-4),
        DepthAnchor(4, 2, 1.0, 1.1, 1e-4),
    ]
    observation = observe_source_depth_model(source, anchors)
    object.__setattr__(observation, "optical_error_m", float("nan"))
    with pytest.raises(RgbdSimContractError) as captured:
        normalize_source_capture(source, observation, anchors)
    assert captured.value.code == "RGBD_SIM_DEPTH_MODEL_BINDING_INVALID"
