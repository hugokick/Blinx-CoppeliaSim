from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simulation.training_scenes import build_scene


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "simulation" / "vision_ocr_sorting_lab"
SPEC_PATH = LAB / "scene_spec.json"
MANIFEST_PATH = LAB / "scene_manifest.json"
SCENE_PATH = LAB / "BL23_vision_ocr_sorting_lab.ttt"
PROFILE_PATH = LAB / "profiles.json"
ASSET_MANIFEST_PATH = LAB / "ocr_assets_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_08_scene_is_registered_and_independent() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    path, formal = build_scene._formal_spec_path(SPEC_PATH)
    assert path == SPEC_PATH.resolve()
    assert formal.scene_id == "vision-ocr-sorting-lab"
    assert formal.root_path == "/VisionOcrSortingLab"
    assert formal.output_relative == (
        "simulation/vision_ocr_sorting_lab/BL23_vision_ocr_sorting_lab.ttt"
    )
    assert spec["root_path"] == formal.root_path
    assert spec["output"] == formal.output_relative
    assert spec["port"] == 23008
    assert spec["output"] != spec["template"]


def test_v1_08_scene_release_hashes_and_asset_binding_are_consistent() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert spec["scene_id"] == manifest["scene_id"] == "vision-ocr-sorting-lab"
    assert manifest["scene"]["path"] == SCENE_PATH.relative_to(ROOT).as_posix()
    assert manifest["scene"]["sha256"] == _sha256(SCENE_PATH)
    assert manifest["scene"]["size_bytes"] == SCENE_PATH.stat().st_size
    assert manifest["template"]["path"] == spec["template"]
    assert manifest["template"]["sha256"] == _sha256(ROOT / spec["template"])
    assert manifest["ocr_assets_manifest"] == {
        "path": ASSET_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "sha256": _sha256(ASSET_MANIFEST_PATH),
    }
    assert manifest["protected_assets_unchanged"] is True
    assert manifest["required_paths"] == spec["required_paths"]
    assert manifest["reset_contract"] == {
        "strategy": "scene_reload",
        "tool_off": True,
        "robot_home": True,
    }
    assert manifest["ocr_sorting"]["calibration_matrix"] == spec[
        "calibration_matrix"
    ]
    assert manifest["ocr_sorting"]["initial_positions_mm"] == {
        item["alias"]: item["position_mm"] for item in spec["parts"]
    }
    assert manifest["ocr_sorting"]["route_slots_mm"] == {
        route["alias"]: [slot["position_mm"] for slot in route["slots"]]
        for route in spec["routes"]
    }


def test_v1_08_scene_objects_routes_and_fixed_profile() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["root_path"] == "/VisionOcrSortingLab"
    assert [part["alias"] for part in spec["parts"]] == [
        "part_a",
        "part_b",
        "part_c",
        "part_d",
    ]
    assert [part["identifier"] for part in spec["parts"]] == [
        "A1",
        "A2",
        "B1",
        "B2",
    ]
    assert [part["route"] for part in spec["parts"]] == [
        "route_alpha",
        "route_alpha",
        "route_beta",
        "route_beta",
    ]
    assert [route["alias"] for route in spec["routes"]] == [
        "route_alpha",
        "route_beta",
    ]
    assert all(len(route["slots"]) == 2 for route in spec["routes"])
    assert spec["profiles"] == "simulation/vision_ocr_sorting_lab/profiles.json"
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert profiles["baseline_profile_id"] == "standard"
    assert profiles["profiles"] == [
        {
            "profile_id": "standard",
            "label": "OCR 分拣固定视图",
            "resolution": [1024, 1024],
            "perspective_angle_deg": 20,
            "camera_rig_z_m": 0.5,
            "key_diffuse_rgb": [0.8, 0.8, 0.8],
            "fill_diffuse_rgb": [0.35, 0.35, 0.35],
        }
    ]
    assert spec["camera"]["path"] == "/VisionOcrSortingLab/CameraRig/Camera"
    assert spec["lighting"] == {
        "key_path": "/VisionOcrSortingLab/Lighting/KeyLight",
        "fill_path": "/VisionOcrSortingLab/Lighting/FillLight",
    }
    required = set(spec["required_paths"])
    for alias in ("part_a", "part_b", "part_c", "part_d"):
        assert f"/VisionOcrSortingLab/Parts/{alias}" in required
        assert f"/VisionOcrSortingLab/Parts/{alias}/CodeFace" in required
    for route in ("route_alpha", "route_beta"):
        assert f"/VisionOcrSortingLab/Routes/{route}" in required
        for index in (1, 2):
            assert f"/VisionOcrSortingLab/Routes/{route}/slot_{index}" in required


def test_v1_08_scene_json_is_checkout_stable_lf() -> None:
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    rules = {
        line.strip()
        for line in attrs.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for relative in (
        "simulation/vision_ocr_sorting_lab/scene_spec.json",
        "simulation/vision_ocr_sorting_lab/scene_manifest.json",
        "simulation/vision_ocr_sorting_lab/profiles.json",
    ):
        assert f"{relative} text eol=lf" in rules
        assert b"\r\n" not in (ROOT / relative).read_bytes()
