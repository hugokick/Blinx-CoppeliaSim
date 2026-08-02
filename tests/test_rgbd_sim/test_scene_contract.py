from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "simulation" / "rgbd_lab" / "scene_spec.json"
MANIFEST_PATH = ROOT / "simulation" / "rgbd_lab" / "scene_manifest.json"
REQUIRED = {
    "/RgbdLab",
    "/RgbdLab/CameraRig/RgbdSensor",
    "/RgbdLab/ReferencePlane",
    "/RgbdLab/Targets/near_block",
    "/RgbdLab/Targets/far_block",
    "/RgbdLab/Targets/step_low",
    "/RgbdLab/Targets/step_high",
    "/RgbdLab/ProbeAnchors/center",
    "/RgbdLab/ProbeAnchors/off_axis_left",
    "/RgbdLab/ProbeAnchors/off_axis_right",
}


def test_rgbd_scene_spec_declares_fixed_sensor_and_required_paths() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["sensor"]["path"] == "/RgbdLab/CameraRig/RgbdSensor"
    assert spec["sensor"]["resolution"] == [256, 256]
    assert spec["sensor"]["perspective_angle_deg"] == 60.0
    assert spec["sensor"]["explicit_handling"] is True
    assert spec["sensor"]["perspective"] is True
    assert spec["sensor"]["rgb_enabled"] is True
    assert spec["sensor"]["depth_enabled"] is True
    assert set(spec["required_paths"]) == REQUIRED
    assert spec["source_depth_model"] in {"optical_z", "ray_range"}
    assert spec["port"] == 23009


def test_rgbd_scene_manifest_binds_new_scene_and_protected_template() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["scene"]["path"] == "BL23_rgbd_lab.ttt"
    assert len(manifest["scene"]["sha256"]) == 64
    assert manifest["sensor"]["resolution"] == [256, 256]
    assert manifest["sensor"]["explicit_handling"] is True
    assert manifest["protected_assets_unchanged"] is True
    assert len(manifest["template"]["sha256"]) == 64
    scene = ROOT / "simulation" / "rgbd_lab" / "BL23_rgbd_lab.ttt"
    assert scene.is_file()
    digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    assert digest == manifest["scene"]["sha256"]
