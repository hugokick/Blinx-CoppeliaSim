from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "simulation" / "rgbd_lab" / "scene_spec.json"
MANIFEST_PATH = ROOT / "simulation" / "rgbd_lab" / "scene_manifest.json"
PROFILES_PATH = ROOT / "simulation" / "rgbd_lab" / "profiles.json"
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


def test_rgbd_profiles_bind_the_single_standard_sensor_contract() -> None:
    assert PROFILES_PATH.is_file(), "D1-01 profiles.json is required by Task 4"
    raw = PROFILES_PATH.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    profiles = json.loads(raw.decode("utf-8"))
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert set(profiles) == {"schema_version", "baseline_profile_id", "sensor_path", "profiles"}
    assert profiles["schema_version"] == 1
    assert profiles["baseline_profile_id"] == "standard"
    assert profiles["sensor_path"] == spec["sensor"]["path"] == manifest["sensor"]["path"]
    assert isinstance(profiles["profiles"], list)
    assert len(profiles["profiles"]) == 1
    profile = profiles["profiles"][0]
    assert set(profile) == {
        "profile_id",
        "label",
        "resolution",
        "perspective_angle_deg",
        "near_clip_m",
        "far_clip_m",
        "explicit_handling",
        "perspective",
        "rgb_enabled",
        "depth_enabled",
        "expected_source_depth_model",
        "depth_unit",
        "pixel_center_convention",
    }
    assert profile["profile_id"] == "standard"
    assert isinstance(profile["label"], str) and profile["label"]
    sensor_spec = spec["sensor"]
    sensor_manifest = manifest["sensor"]
    assert profile["resolution"] == sensor_spec["resolution"] == sensor_manifest["resolution"] == [256, 256]
    for key in ("perspective_angle_deg", "near_clip_m", "far_clip_m"):
        assert type(profile[key]) in {int, float}
        assert math.isfinite(float(profile[key]))
        assert profile[key] == sensor_spec[key] == sensor_manifest[key]
    for key in ("explicit_handling", "perspective", "rgb_enabled", "depth_enabled"):
        assert type(profile[key]) is bool
        assert profile[key] is sensor_spec[key] is sensor_manifest[key] is True
    assert profile["expected_source_depth_model"] == spec["source_depth_model"] == manifest["source_depth_model"] == "optical_z"
    assert profile["depth_unit"] == "metre"
    assert profile["pixel_center_convention"] == "integer_center_top_left_zero"
