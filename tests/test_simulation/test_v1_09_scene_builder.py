from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from simulation.training_scenes import build_scene as builder


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "simulation" / "vision_defect_sorting_lab" / "scene_spec.json"


def _write_v1_09_recovery_fixture(tmp_path: Path) -> dict[str, object]:
    formal = builder._FORMAL_SCENES["simulation/vision_defect_sorting_lab/scene_spec.json"]
    output = tmp_path / "BL23_vision_defect_sorting_lab.ttt"
    output.write_bytes(b"scene-bytes")
    scene_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    profile_path = tmp_path / "profiles.json"
    profile_path.write_bytes(b"profile-bytes")
    profile_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    asset_path = tmp_path / "defect_assets_manifest.json"
    asset_path.write_bytes(b"asset-manifest-bytes")
    asset_sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "scene_id": formal.scene_id,
        "protected_assets_unchanged": True,
        "required_paths": list(formal.required_paths),
        "template": {"path": builder.TEMPLATE_RELATIVE, "sha256": "a" * 64},
        "scene": {
            "path": formal.output_relative,
            "sha256": scene_sha256,
            "size_bytes": len(output.read_bytes()),
        },
        "profile_catalog": {
            "path": "simulation/vision_defect_sorting_lab/profiles.json",
            "sha256": profile_sha256,
        },
        "defect_assets_manifest": {
            "path": "simulation/vision_defect_sorting_lab/defect_assets_manifest.json",
            "sha256": asset_sha256,
        },
    }
    manifest_path = tmp_path / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "formal": formal,
        "output": output,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "template_hash": "a" * 64,
        "profile_path": profile_path,
        "profile_sha256": profile_sha256,
        "asset_path": asset_path,
        "asset_sha256": asset_sha256,
        "profile_binding": (
            "simulation/vision_defect_sorting_lab/profiles.json",
            profile_path,
            profile_sha256,
        ),
        "asset_binding": (
            "simulation/vision_defect_sorting_lab/defect_assets_manifest.json",
            asset_path,
            asset_sha256,
        ),
    }


def test_v1_09_is_registered_with_the_generic_builder() -> None:
    spec_path, formal = builder._formal_spec_path(SPEC_PATH)
    assert spec_path == SPEC_PATH.resolve()
    assert formal.scene_id == "vision-defect-sorting-lab"
    assert formal.root_path == "/VisionDefectSortingLab"
    assert formal.output_relative == "simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt"
    assert "/VisionDefectSortingLab/CameraRig/Camera" in formal.required_paths


def test_v1_09_spec_validation_is_strict_about_assets_and_geometry() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    _spec_path, formal = builder._formal_spec_path(SPEC_PATH)
    builder._validate_spec(spec, formal)
    assert spec["port"] == 23010


def test_v1_09_builder_entrypoint_is_fixed_port_and_does_not_read_ground_truth() -> None:
    source = (ROOT / "tools" / "vision_lab" / "build_v1_09_scene.py").read_text(encoding="utf-8")
    assert "port=23010" in source
    assert "build_scene" in source
    assert "acceptance_ground_truth" not in source


def test_v1_09_asset_manifest_is_not_available_before_task_3() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    manifest_path = ROOT / spec["defect_assets_manifest"]
    if manifest_path.exists():
        pytest.skip("Task 3 asset manifest has been integrated")
    assert not manifest_path.exists()


def test_v1_09_complete_release_recovery_requires_profile_and_asset_bindings(tmp_path: Path) -> None:
    fixture = _write_v1_09_recovery_fixture(tmp_path)
    release = builder._recoverable_release(
        fixture["output"],
        fixture["manifest_path"],
        fixture["formal"],
        fixture["template_hash"],
        fixture["profile_binding"],
        None,
        None,
        fixture["asset_binding"],
    )
    assert release is not None
    assert release.scene_sha256 == fixture["manifest"]["scene"]["sha256"]


@pytest.mark.parametrize(
    "tamper",
    (
        "profile_missing",
        "profile_binding_mismatch",
        "defect_assets_missing",
        "defect_assets_binding_mismatch",
        "profile_bytes_changed",
        "defect_assets_bytes_changed",
    ),
)
def test_v1_09_release_recovery_rejects_incomplete_or_tampered_bindings(tmp_path: Path, tamper: str) -> None:
    fixture = _write_v1_09_recovery_fixture(tmp_path)
    manifest = fixture["manifest"]
    if tamper == "profile_missing":
        manifest.pop("profile_catalog")
    elif tamper == "profile_binding_mismatch":
        manifest["profile_catalog"]["sha256"] = "b" * 64
    elif tamper == "defect_assets_missing":
        manifest.pop("defect_assets_manifest")
    elif tamper == "defect_assets_binding_mismatch":
        manifest["defect_assets_manifest"]["sha256"] = "c" * 64
    elif tamper == "profile_bytes_changed":
        fixture["profile_path"].write_bytes(b"profile-bytes-tampered")
    elif tamper == "defect_assets_bytes_changed":
        fixture["asset_path"].write_bytes(b"asset-manifest-bytes-tampered")
    else:
        raise AssertionError(f"unknown tamper case: {tamper}")
    fixture["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

    release = builder._recoverable_release(
        fixture["output"],
        fixture["manifest_path"],
        fixture["formal"],
        fixture["template_hash"],
        fixture["profile_binding"],
        None,
        None,
        fixture["asset_binding"],
    )
    assert release is None


def test_v1_09_manifest_accepts_task3_canonical_schema(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    assets = []
    for index, asset_id in enumerate(("reference", "entry_a", "entry_b", "entry_c", "entry_d", "entry_e", "entry_f")):
        asset_path = assets_root / f"{asset_id}.png"
        assert cv2.imwrite(str(asset_path), np.full((256, 256), 160 + index, dtype=np.uint8))
        assets.append(
            {
                "asset_id": asset_id,
                "generator": "tools.vision_lab.generate_v1_09_defect_assets",
                "generator_version": "1.0.0",
                "path": f"assets/{asset_id}.png",
                "purpose": "test",
                "sha256": builder._sha256(asset_path),
                "size_px": [256, 256],
                "source": "project-original-generated",
            }
        )
    manifest = tmp_path / "defect_assets_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": assets,
                "generator": "tools.vision_lab.generate_v1_09_defect_assets",
                "generator_version": "1.0.0",
                "scene_id": "V1-09",
                "schema_version": 1,
                "seed": 20260803,
                "source": "project-original-generated",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validated = builder._validate_defect_assets_manifest(manifest)
    assert len(validated["assets"]) == 7


def test_v1_09_reference_inspection_face_is_direct_child(tmp_path: Path) -> None:
    asset_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(asset_path), np.full((256, 256), 255, dtype=np.uint8))

    class FakeSim:
        primitiveshape_cuboid = 1
        colorcomponent_ambient_diffuse = 2
        shapeintparam_static = 3
        shapeintparam_respondable = 4

        def __init__(self) -> None:
            self.next_handle = 1
            self.aliases: dict[int, str] = {}
            self.parents: dict[int, int] = {}
            self.group_handle = 0

        def createPrimitiveShape(self, *_args):
            handle = self.next_handle
            self.next_handle += 1
            return handle

        def setObjectAlias(self, handle, name):
            self.aliases[int(handle)] = str(name)

        def setShapeColor(self, *_args):
            return None

        def setObjectInt32Param(self, *_args):
            return None

        def setObjectParent(self, handle, parent, _keep_in_place):
            self.parents[int(handle)] = int(parent)

        def setObjectPosition(self, *_args):
            return None

        def groupShapes(self, pieces, _merge):
            self.group_handle = 1000
            return self.group_handle

        def createDummy(self, *_args):
            handle = self.next_handle
            self.next_handle += 1
            return handle

    fake = FakeSim()
    builder._defect_surface_part(
        fake,
        {"alias": "reference", "position_mm": [60, 54, 18], "size_mm": [28, 28, 16]},
        asset_path,
        100,
        inspection_face_direct=True,
        respondable=False,
    )
    assert fake.aliases[fake.group_handle] == "InspectionFace"
    assert fake.parents[fake.group_handle] == 100
    assert "ReferencePart" not in fake.aliases.values()
