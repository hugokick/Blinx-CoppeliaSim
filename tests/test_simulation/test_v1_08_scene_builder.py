from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import simulation.training_scenes.build_scene as scene_builder
from tools.vision_lab import build_v1_08_scene


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "simulation" / "vision_ocr_sorting_lab" / "scene_spec.json"


def test_python_wrapper_passes_only_canonical_v1_08_spec_and_port(monkeypatch):
    calls = []

    def fake_build_scene(**kwargs):
        calls.append(kwargs)
        return {"status": "PENDING_COPPELIASIM"}

    monkeypatch.setattr(build_v1_08_scene, "build_scene", fake_build_scene)
    assert build_v1_08_scene.main() == 0
    assert calls == [
        {
            "spec_path": SPEC,
            "host": "127.0.0.1",
            "port": 23008,
        }
    ]


def test_python_wrapper_rejects_port_override(monkeypatch):
    source = (ROOT / "tools" / "vision_lab" / "build_v1_08_scene.py").read_text(
        encoding="utf-8"
    )
    assert "23008" in source
    assert "sys.argv" not in source
    assert "--port" not in source


def test_powershell_wrapper_composes_owned_launch_and_cleanup_contract():
    path = ROOT / "tools" / "vision_lab" / "build_v1_08_scene.ps1"
    source = path.read_text(encoding="utf-8")
    assert "launch_coppeliasim.ps1" in source
    assert "process_ownership.ps1" in source
    assert "build_v1_08_scene.py" in source
    assert "23008" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "finally" in source
    assert "-Code" not in source
    assert "Invoke-Expression" not in source
    assert "-Output" not in source


def test_v1_08_spec_is_strictly_validated_before_remote_connection(monkeypatch):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    formal = scene_builder._FORMAL_SCENES[
        "simulation/vision_ocr_sorting_lab/scene_spec.json"
    ]
    scene_builder._validate_spec(spec, formal)

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("remote connection must follow spec validation")

    monkeypatch.setattr(scene_builder, "RemoteAPIClient", ForbiddenClient)
    with pytest.raises(AssertionError, match="remote connection"):
        scene_builder.build_scene(spec_path=SPEC, host="127.0.0.1", port=23008)


def test_existing_v1_08_release_recovery_does_not_require_code_assets_manifest():
    manifest_path = SPEC.parent / "scene_manifest.json"
    scene_path = SPEC.parent / "BL23_vision_ocr_sorting_lab.ttt"
    template_path = ROOT / "simulation" / "vision_lab" / "BL23_vision_lab.ttt"
    profile_path = SPEC.parent / "profiles.json"
    assets_path = SPEC.parent / "ocr_assets_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formal = scene_builder._FORMAL_SCENES[
        "simulation/vision_ocr_sorting_lab/scene_spec.json"
    ]
    release = scene_builder._recoverable_release(
        scene_path,
        manifest_path,
        formal,
        hashlib.sha256(template_path.read_bytes()).hexdigest(),
        (
            "simulation/vision_ocr_sorting_lab/profiles.json",
            profile_path,
            hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        ),
        None,
        (
            "simulation/vision_ocr_sorting_lab/ocr_assets_manifest.json",
            assets_path,
            hashlib.sha256(assets_path.read_bytes()).hexdigest(),
        ),
    )
    assert release is not None
    assert release.scene_sha256 == manifest["scene"]["sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_profile_id", "not_standard"),
        ("profile.perspective_angle_deg", float("inf")),
        ("profile.resolution", [1024.0, 1024]),
        ("profile.key_diffuse_rgb", [1.5, 0.8, 0.8]),
        ("profile.camera_rig_z_m", "0.5"),
    ],
)
def test_ocr_profile_catalog_rejects_invalid_fixed_profile_values(
    tmp_path, field, value
):
    payload = json.loads(
        (ROOT / "simulation" / "vision_ocr_sorting_lab" / "profiles.json").read_text(
            encoding="utf-8"
        )
    )
    if field.startswith("profile."):
        payload["profiles"][0][field.split(".", 1)[1]] = value
    else:
        payload[field] = value
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError):
        scene_builder._load_ocr_profile_catalog(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("near_clip_m", 10**1000),
        lambda payload: payload["profiles"][0].__setitem__(
            "perspective_angle_deg", 10**1000
        ),
        lambda payload: payload["profiles"][0].__setitem__(
            "key_diffuse_rgb", [10**1000, 0.8, 0.8]
        ),
    ],
)
def test_ocr_profile_catalog_rejects_unbounded_integer_values(tmp_path, mutate):
    payload = json.loads(
        (ROOT / "simulation" / "vision_ocr_sorting_lab" / "profiles.json").read_text(
            encoding="utf-8"
        )
    )
    mutate(payload)
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        scene_builder._load_ocr_profile_catalog(path)


@pytest.mark.parametrize("tamper", ["hash", "path"])
def test_ocr_assets_manifest_rejects_tampered_label_binding(tmp_path, tamper):
    payload = json.loads(
        (ROOT / "simulation" / "vision_ocr_sorting_lab" / "ocr_assets_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if tamper == "hash":
        payload["labels"][0]["sha256"] = "0" * 64
    else:
        payload["labels"][0]["path"] = "../labels/A1.png"
    path = tmp_path / "ocr_assets_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        scene_builder._validate_ocr_assets_manifest(path)
