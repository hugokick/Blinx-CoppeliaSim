from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from vision_platform.rgbd.models import RgbdFrame
from vision_platform.rgbd_sim.depth_model import SourceDepthModelObservation
from vision_platform.rgbd_sim.models import RgbdSimCapture, RgbdSourceCapture, RgbdSensorMetadata
from vision_platform.rgbd_sim.probe import build_probe_report, probe_report_to_dict
from vision_platform.rgbd_sim.scene_binding import load_scene_binding


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "simulation" / "rgbd_lab" / "scene_manifest.json"


def _capture() -> RgbdSimCapture:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    width = height = 256
    metadata = RgbdSensorMetadata(
        sensor_path="/RgbdLab/CameraRig/RgbdSensor",
        resolution=(width, height),
        near_clip_m=0.05,
        far_clip_m=5.0,
        perspective_angle_rad=math.radians(60.0),
        scene_path="simulation/rgbd_lab/BL23_rgbd_lab.ttt",
        scene_sha256=manifest["scene"]["sha256"],
        sequence_id=7,
        timestamp_s=12.5,
        expected_source_depth_model="optical_z",
    )
    source_depth = np.full((height, width), 2.5, dtype=np.float32)
    for name, value in {
        "near_block": 1.8,
        "far_block": 2.8,
        "step_low": 2.4,
        "step_high": 1.9,
    }.items():
        x0, y0, x1, y1 = manifest["validation_rois"][name]
        source_depth[y0:y1, x0:x1] = value
    image = np.zeros((height, width, 3), dtype=np.uint8)
    source = RgbdSourceCapture(metadata, image, source_depth)
    frame = RgbdFrame(image, source_depth)
    from vision_platform.rgbd_sim.intrinsics import derive_intrinsics

    return RgbdSimCapture(
        source=source,
        frame=frame,
        intrinsics=derive_intrinsics(metadata),
        observed_source_depth_model="optical_z",
    )


def test_probe_proves_configured_virtual_depth_order() -> None:
    binding = load_scene_binding(MANIFEST, repository_root=ROOT)
    report = build_probe_report(_capture(), binding)

    assert report.status == "PASS"
    assert report.observed_source_depth_model == report.expected_source_depth_model
    assert report.output_depth_model == "optical_z"
    assert report.intrinsics.fx_px > 0.0
    assert report.sequence_id == 7
    assert report.rois["near_block"].median_depth_m < report.rois["far_block"].median_depth_m
    assert report.rois["step_high"].median_depth_m < report.rois["step_low"].median_depth_m
    payload = probe_report_to_dict(report)
    assert payload["scene_path"] == "simulation/rgbd_lab/BL23_rgbd_lab.ttt"
    assert payload["sensor_path"] == "/RgbdLab/CameraRig/RgbdSensor"
    assert payload["depth_summary"]["valid_count"] == 65536
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_probe_reports_depth_model_mismatch_without_raw_arrays() -> None:
    binding = load_scene_binding(MANIFEST, repository_root=ROOT)
    capture = _capture()
    bad = RgbdSimCapture(
        source=capture.source,
        frame=capture.frame,
        intrinsics=capture.intrinsics,
        observed_source_depth_model="optical_z",
    )
    report = build_probe_report(bad, binding)
    payload = probe_report_to_dict(report)

    assert report.status == "PASS"
    assert "image_bgr" not in payload
    assert "source_depth_summary" not in payload
