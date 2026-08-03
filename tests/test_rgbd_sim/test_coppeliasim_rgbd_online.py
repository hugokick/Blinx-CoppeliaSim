from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from vision_platform.rgbd_sim import (
    CoppeliaRgbdCapture,
    RgbdSimContractError,
    build_probe_report,
    colorize_depth,
    load_scene_binding,
    normalize_source_capture,
    observe_source_depth_model,
    probe_report_to_dict,
)

from .coppeliasim_process import OwnedCoppeliaSim


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "simulation" / "rgbd_lab" / "scene_manifest.json"


class _CountingSim:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.handle_calls = 0

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def handleVisionSensor(self, handle):
        self.handle_calls += 1
        return self._delegate.handleVisionSensor(handle)


@pytest.fixture(scope="module")
def online_context(tmp_path_factory):
    binding = load_scene_binding(MANIFEST, repository_root=ROOT)
    with OwnedCoppeliaSim.start(binding.scene_file, port=23009) as owned:
        client = RemoteAPIClient(host="127.0.0.1", port=23009)
        sim = client.require("sim")
        try:
            yield binding, owned, client, sim
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


@pytest.mark.coppeliasim
def test_online_rgbd_capture_contract_and_probe(online_context, tmp_path: Path) -> None:
    binding, _owned, client, sim = online_context
    for required_path in binding.required_paths:
        assert int(sim.getObject(required_path)) >= 0
    sensor_handle = int(sim.getObject(binding.sensor_path))
    assert int(sim.getExplicitHandling(sensor_handle)) == 1
    assert tuple(sim.getVisionSensorResolution(sensor_handle)) == binding.resolution
    assert int(sim.getObjectInt32Param(sensor_handle, sim.visionintparam_perspective_operation)) == 1
    # Some installed CoppeliaSim builds return ``None`` for these legacy
    # ignored-channel parameters; the capture contract proves both buffers
    # exist in the same transaction and maps that unsupported value to enabled.
    assert sim.getObjectInt32Param(sensor_handle, sim.visionintparam_rgbignored) in {None, 0}
    assert sim.getObjectInt32Param(sensor_handle, sim.visionintparam_depthignored) in {None, 0}
    assert math.isclose(
        float(sim.getObjectFloatParam(sensor_handle, sim.visionfloatparam_perspective_angle)),
        binding.perspective_angle_rad,
        rel_tol=0.0,
        abs_tol=1e-5,
    )
    assert math.isclose(float(sim.getObjectFloatParam(sensor_handle, sim.visionfloatparam_near_clipping)), binding.near_clip_m, abs_tol=1e-6)
    assert math.isclose(float(sim.getObjectFloatParam(sensor_handle, sim.visionfloatparam_far_clipping)), binding.far_clip_m, abs_tol=1e-6)

    counting_sim = _CountingSim(sim)
    capture_adapter = CoppeliaRgbdCapture(
        sensor_path=binding.sensor_path,
        scene_binding=binding,
        sim=counting_sim,
        client=client,
        port=23009,
    )
    reports = []
    captures = []
    try:
        for index in range(3):
            source = capture_adapter.read_source()
            assert counting_sim.handle_calls == index + 1
            assert source.image_bgr.shape == (*binding.resolution[::-1], 3)
            assert source.source_depth_m.shape == binding.resolution[::-1]
            assert source.source_depth_m.dtype == np.float32
            assert np.all(np.isfinite(source.source_depth_m))
            assert np.all(source.source_depth_m >= 0.0)
            observation = observe_source_depth_model(source, binding.anchors)
            capture = normalize_source_capture(source, observation, binding.anchors)
            report = build_probe_report(capture, binding)
            assert report.status == "PASS"
            assert report.expected_source_depth_model == "optical_z"
            assert report.observed_source_depth_model == "optical_z"
            assert report.output_depth_model == "optical_z"
            assert len(report.source_model_evidence) == 3
            captures.append(capture)
            reports.append(report)
    finally:
        capture_adapter.close()

    assert [capture.source.metadata.sequence_id for capture in captures] == [0, 1, 2]
    first_intrinsics = reports[0].intrinsics
    for report in reports[1:]:
        assert report.intrinsics == first_intrinsics
    for name in ("near_block", "far_block", "step_low", "step_high"):
        medians = [report.rois[name].median_depth_m for report in reports]
        assert max(medians) - min(medians) <= 0.02
    assert reports[0].rois["near_block"].median_depth_m + binding.depth_order_margin_m < reports[0].rois["far_block"].median_depth_m
    assert reports[0].rois["step_high"].median_depth_m + binding.depth_order_margin_m < reports[0].rois["step_low"].median_depth_m

    artifact_dir = tmp_path / "rgbd-online-evidence"
    artifact_dir.mkdir()
    cv2.imwrite(str(artifact_dir / "rgb.png"), captures[0].frame.image_bgr)
    cv2.imwrite(str(artifact_dir / "depth.png"), colorize_depth(captures[0].frame.depth_m, minimum_m=binding.near_clip_m, maximum_m=binding.far_clip_m))
    (artifact_dir / "report.json").write_text(json.dumps(probe_report_to_dict(reports[0]), allow_nan=False, sort_keys=True, indent=2), encoding="utf-8")


@pytest.mark.coppeliasim
def test_online_wrong_sensor_path_fails_without_fallback(online_context) -> None:
    binding, _owned, client, sim = online_context
    capture = CoppeliaRgbdCapture(
        sensor_path="/RgbdLab/CameraRig/NotRgbdSensor",
        scene_binding=binding,
        sim=sim,
        client=client,
        port=23009,
    )
    with pytest.raises(RgbdSimContractError) as exc_info:
        capture.read_source()
    assert exc_info.value.code == "RGBD_SIM_CAPTURE_INVALID"
    capture.close()


@pytest.mark.coppeliasim
def test_online_tampered_manifest_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["scene"]["sha256"] = "0" * 64
    root = tmp_path / "repo"
    manifest = root / "simulation" / "rgbd_lab" / "scene_manifest.json"
    manifest.parent.mkdir(parents=True)
    scene = root / "simulation" / "rgbd_lab" / "BL23_rgbd_lab.ttt"
    scene.write_bytes((ROOT / "simulation" / "rgbd_lab" / "BL23_rgbd_lab.ttt").read_bytes())
    payload["scene"]["path"] = "simulation/rgbd_lab/BL23_rgbd_lab.ttt"
    payload["scene"]["size_bytes"] = scene.stat().st_size
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RgbdSimContractError) as exc_info:
        load_scene_binding(manifest, repository_root=root)
    assert exc_info.value.code == "RGBD_SIM_BINDING_SCENE_INVALID"
