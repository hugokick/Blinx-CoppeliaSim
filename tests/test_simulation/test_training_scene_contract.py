from __future__ import annotations

import hashlib
import json

import pytest

from simulation.training_scenes.scene_contract import validate_scene_contract


def _write_contract(tmp_path):
    template = tmp_path / "template.ttt"
    template.write_bytes(b"protected-template")
    scene = tmp_path / "new-scene.ttt"
    scene.write_bytes(b"independent-scene")
    spec = tmp_path / "scene_spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_id": "robot-basics",
                "template": "template.ttt",
                "output": "new-scene.ttt",
                "root_path": "/RobotBasics",
                "required_paths": [
                    "/BLX_base_link",
                    "/RobotBasics/Camera",
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "scene_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_id": "robot-basics",
                "template": {
                    "path": "template.ttt",
                    "sha256": hashlib.sha256(
                        template.read_bytes()
                    ).hexdigest(),
                },
                "scene": {
                    "path": "new-scene.ttt",
                    "sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                },
                "required_paths": [
                    "/BLX_base_link",
                    "/RobotBasics/Camera",
                ],
                "protected_assets_unchanged": True,
            }
        ),
        encoding="utf-8",
    )
    return spec, manifest, template


def test_scene_contract_verifies_template_scene_and_paths(tmp_path):
    spec, manifest, _ = _write_contract(tmp_path)

    report = validate_scene_contract(
        spec,
        manifest,
        project_root=tmp_path,
    )

    assert report["status"] == "PASS"
    assert report["scene_id"] == "robot-basics"
    assert report["required_path_count"] == 2


def test_scene_contract_rejects_modified_template(tmp_path):
    spec, manifest, template = _write_contract(tmp_path)
    template.write_bytes(b"changed")

    with pytest.raises(ValueError, match="template sha256"):
        validate_scene_contract(
            spec,
            manifest,
            project_root=tmp_path,
        )


def test_scene_contract_rejects_output_equal_to_template(tmp_path):
    spec, manifest, _ = _write_contract(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["output"] = payload["template"]
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite template"):
        validate_scene_contract(
            spec,
            manifest,
            project_root=tmp_path,
        )
