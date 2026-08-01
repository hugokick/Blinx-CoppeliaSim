from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import simulation.training_scenes.build_scene as scene_builder
from simulation.training_scenes.scene_contract import validate_scene_contract
from vision_platform.vision_quality import load_profile_catalog


ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT / "simulation" / "vision_quality_lab"
SPEC_PATH = SCENE_DIR / "scene_spec.json"
PROFILES_PATH = SCENE_DIR / "profiles.json"
REQUIRED_PATHS = [
    "/VisionQualityLab",
    "/VisionQualityLab/InspectionBoard",
    "/VisionQualityLab/Samples",
    "/VisionQualityLab/Samples/ReferenceRectangle",
    "/VisionQualityLab/Samples/ReferenceCircle",
    "/VisionQualityLab/Samples/ReferenceTriangle",
    "/VisionQualityLab/Samples/ResolutionTarget",
    "/VisionQualityLab/CameraRig",
    "/VisionQualityLab/CameraRig/Camera",
    "/VisionQualityLab/Lighting",
    "/VisionQualityLab/Lighting/KeyLight",
    "/VisionQualityLab/Lighting/FillLight",
]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_vision_quality_spec_is_the_exact_formal_declaration():
    spec = _json(SPEC_PATH)

    assert set(spec) == {
        "schema_version", "scene_id", "template", "output", "remove_paths",
        "root_path", "profiles", "workspace", "camera", "samples", "required_paths",
    }
    assert spec["schema_version"] == 1
    assert spec["scene_id"] == "vision-quality-lab"
    assert spec["template"] == "simulation/vision_lab/BL23_vision_lab.ttt"
    assert spec["output"] == "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt"
    assert spec["template"] != spec["output"]
    assert spec["root_path"] == "/VisionQualityLab"
    assert spec["profiles"] == "simulation/vision_quality_lab/profiles.json"
    assert spec["remove_paths"] == ["/VisionLab"]
    assert spec["workspace"] == {
        "center_m": [0.35, 0.0, 0.02],
        "size_m": [0.36, 0.28, 0.02],
    }
    assert spec["camera"] == {
        "rig_position_m": [0.35, 0.0, 0.7],
        "orientation_deg": [180, 0, 0],
    }
    assert spec["samples"] == {
        "ReferenceRectangle": {"shape": "cuboid", "position_m": [0.29, -0.06, 0.045], "size_m": [0.08, 0.04, 0.03], "color_rgb": [0.85, 0.1, 0.1]},
        "ReferenceCircle": {"shape": "cylinder", "position_m": [0.41, -0.06, 0.045], "size_m": [0.05, 0.05, 0.03], "color_rgb": [0.1, 0.3, 0.9]},
        "ReferenceTriangle": {"shape": "triangle", "position_m": [0.29, 0.07, 0.045], "size_m": [0.07, 0.06, 0.03], "color_rgb": [0.1, 0.75, 0.2]},
        "ResolutionTarget": {"shape": "resolution_target", "position_m": [0.41, 0.07, 0.031], "size_m": [0.08, 0.06, 0.002], "stripe_count": 12},
    }
    assert spec["required_paths"] == REQUIRED_PATHS


def test_profile_catalog_declares_the_three_fixed_profiles():
    catalog = load_profile_catalog(PROFILES_PATH)

    assert catalog.baseline_profile_id == "standard"
    assert catalog.profile_ids == ("standard", "wide_dim", "detail_bright")
    assert (catalog.sensor_path, catalog.camera_rig_path, catalog.key_light_path, catalog.fill_light_path) == (
        "/VisionQualityLab/CameraRig/Camera", "/VisionQualityLab/CameraRig",
        "/VisionQualityLab/Lighting/KeyLight", "/VisionQualityLab/Lighting/FillLight",
    )
    assert (catalog.near_clip_m, catalog.far_clip_m) == (0.05, 2.0)
    assert [(item.label, item.resolution, item.perspective_angle_deg, item.camera_rig_z_m, item.key_diffuse_rgb, item.fill_diffuse_rgb) for item in (catalog.require("standard"), catalog.require("wide_dim"), catalog.require("detail_bright"))] == [
        ("标准视图", (512, 512), 60, 0.7, (0.8, 0.8, 0.8), (0.35, 0.35, 0.35)),
        ("宽视场弱光", (256, 256), 75, 0.8, (0.35, 0.35, 0.35), (0.15, 0.15, 0.15)),
        ("细节强光", (768, 768), 40, 0.6, (1.0, 1.0, 1.0), (0.55, 0.55, 0.55)),
    ]


def test_formal_spec_path_only_accepts_the_canonical_vision_quality_spec():
    path, formal = scene_builder._formal_spec_path(SPEC_PATH)

    assert path == SPEC_PATH
    assert formal.scene_id == "vision-quality-lab"
    with pytest.raises(ValueError, match="approved formal spec"):
        scene_builder._formal_spec_path(PROFILES_PATH)


def test_profile_catalog_hash_is_a_nonzero_sha256():
    digest = hashlib.sha256(PROFILES_PATH.read_bytes()).hexdigest()

    assert len(digest) == 64
    assert int(digest, 16) != 0


@pytest.mark.parametrize(
    ("alias", "field", "value", "message"),
    [
        ("ReferenceRectangle", "shape", "cylinder", "ReferenceRectangle"),
        ("ReferenceCircle", "color_rgb", [True, 0.3, 0.9], "color_rgb"),
        ("ReferenceTriangle", "position_m", [float("inf"), 0.07, 0.045], "position_m"),
        ("ResolutionTarget", "stripe_count", True, "stripe_count"),
    ],
)
def test_vision_quality_static_validation_rejects_unsafe_samples(alias, field, value, message):
    spec = _json(SPEC_PATH)
    spec["samples"][alias][field] = value

    with pytest.raises(ValueError, match=message):
        scene_builder._validate_spec(spec, scene_builder._FORMAL_SCENES["simulation/vision_quality_lab/scene_spec.json"])


def test_vision_quality_contract_requires_the_real_profile_catalog_hash(tmp_path):
    profiles = tmp_path / "simulation" / "vision_quality_lab" / "profiles.json"
    profiles.parent.mkdir(parents=True)
    profiles.write_bytes(PROFILES_PATH.read_bytes())
    template = tmp_path / "simulation" / "vision_lab" / "BL23_vision_lab.ttt"
    template.parent.mkdir(parents=True)
    template.write_bytes(b"template")
    output = tmp_path / "simulation" / "vision_quality_lab" / "BL23_vision_quality_lab.ttt"
    output.write_bytes(b"output")
    spec = _json(SPEC_PATH)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "scene_id": "vision-quality-lab",
        "template": {"path": spec["template"], "sha256": hashlib.sha256(template.read_bytes()).hexdigest()},
        "scene": {"path": spec["output"], "sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
        "required_paths": REQUIRED_PATHS, "protected_assets_unchanged": True,
        "profile_catalog": {"path": spec["profiles"], "sha256": hashlib.sha256(profiles.read_bytes()).hexdigest()},
    }), encoding="utf-8")

    report = validate_scene_contract(spec_path, manifest_path, project_root=tmp_path)

    assert report["profile_sha256"] == hashlib.sha256(profiles.read_bytes()).hexdigest()


def test_vision_quality_static_validation_rejects_out_of_bounds_sample_geometry():
    spec = _json(SPEC_PATH)
    spec["samples"]["ReferenceRectangle"]["size_m"] = [2.0, 0.04, 0.03]

    with pytest.raises(ValueError, match="size_m"):
        scene_builder._validate_spec(spec, scene_builder._FORMAL_SCENES["simulation/vision_quality_lab/scene_spec.json"])


def test_vision_quality_build_refuses_before_simulator_connection_or_construction(monkeypatch):
    touched = []

    def forbidden(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("simulator or primitive construction was touched")

    monkeypatch.setattr(scene_builder, "RemoteAPIClient", forbidden)
    monkeypatch.setattr(scene_builder, "_build_workspace", forbidden)

    with pytest.raises(RuntimeError, match="Task 10"):
        scene_builder.build_scene(spec_path=SPEC_PATH, host="127.0.0.1", port=23000)

    assert touched == []
