from __future__ import annotations

import hashlib
import json
import os

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


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_scene_contract_rejects_resolved_output_equal_to_template(tmp_path):
    spec, manifest, template = _write_contract(tmp_path)
    (tmp_path / "alias").mkdir()
    spec_payload = _read_json(spec)
    spec_payload["output"] = "alias/../template.ttt"
    _write_json(spec, spec_payload)
    manifest_payload = _read_json(manifest)
    manifest_payload["scene"] = {
        "path": spec_payload["output"],
        "sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
    }
    _write_json(manifest, manifest_payload)

    with pytest.raises(ValueError, match="must not overwrite template"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


def test_scene_contract_rejects_hardlinked_output(tmp_path):
    spec, manifest, template = _write_contract(tmp_path)
    scene = tmp_path / "new-scene.ttt"
    scene.unlink()
    os.link(template, scene)
    assert template != scene
    assert template.samefile(scene)
    manifest_payload = _read_json(manifest)
    manifest_payload["scene"]["sha256"] = hashlib.sha256(
        scene.read_bytes()
    ).hexdigest()
    _write_json(manifest, manifest_payload)

    with pytest.raises(ValueError, match="must not overwrite template"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize("field", ["template", "output"])
def test_scene_contract_rejects_file_path_escape(tmp_path, field):
    project = tmp_path / "project"
    project.mkdir()
    spec, manifest, _ = _write_contract(project)
    outside = tmp_path / f"outside-{field}.ttt"
    outside.write_bytes(b"outside")
    raw = f"../{outside.name}"
    spec_payload = _read_json(spec)
    spec_payload[field] = raw
    _write_json(spec, spec_payload)
    manifest_payload = _read_json(manifest)
    manifest_field = "template" if field == "template" else "scene"
    manifest_payload[manifest_field] = {
        "path": raw,
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    _write_json(manifest, manifest_payload)

    with pytest.raises(ValueError, match="path must stay inside project"):
        validate_scene_contract(spec, manifest, project_root=project)


@pytest.mark.parametrize("field", ["template", "output"])
def test_scene_contract_rejects_absolute_file_paths(tmp_path, field):
    spec, manifest, template = _write_contract(tmp_path)
    absolute = template if field == "template" else tmp_path / "new-scene.ttt"
    spec_payload = _read_json(spec)
    spec_payload[field] = str(absolute)
    _write_json(spec, spec_payload)
    manifest_payload = _read_json(manifest)
    manifest_field = "template" if field == "template" else "scene"
    manifest_payload[manifest_field]["path"] = str(absolute)
    _write_json(manifest, manifest_payload)

    with pytest.raises(ValueError, match="project-relative path"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize(
    ("document", "value"),
    [
        ("spec", True),
        ("spec", 1.0),
        ("manifest", True),
        ("manifest", 1.0),
    ],
)
def test_scene_contract_rejects_non_integer_schema_one(
    tmp_path,
    document,
    value,
):
    spec, manifest, _ = _write_contract(tmp_path)
    path = spec if document == "spec" else manifest
    payload = _read_json(path)
    payload["schema_version"] = value
    _write_json(path, payload)

    with pytest.raises(ValueError, match="schema_version must be integer 1"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize("value", ["", " ", 7, None, True])
def test_scene_contract_rejects_invalid_scene_id(tmp_path, value):
    spec, manifest, _ = _write_contract(tmp_path)
    for path in (spec, manifest):
        payload = _read_json(path)
        payload["scene_id"] = value
        _write_json(path, payload)

    with pytest.raises(ValueError, match="scene_id must be a non-empty string"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


def test_scene_contract_rejects_missing_scene_id_as_value_error(tmp_path):
    spec, manifest, _ = _write_contract(tmp_path)
    payload = _read_json(spec)
    payload.pop("scene_id")
    _write_json(spec, payload)

    with pytest.raises(ValueError, match="scene_id must be a non-empty string"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("template", None), ("scene", []), ("template", "template.ttt")],
)
def test_scene_contract_rejects_invalid_manifest_file_objects(
    tmp_path,
    field,
    value,
):
    spec, manifest, _ = _write_contract(tmp_path)
    payload = _read_json(manifest)
    payload[field] = value
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match=f"manifest {field} must be an object"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize("field", ["template", "output"])
def test_scene_contract_rejects_blank_spec_file_paths(tmp_path, field):
    spec, manifest, _ = _write_contract(tmp_path)
    payload = _read_json(spec)
    payload[field] = " "
    _write_json(spec, payload)

    with pytest.raises(
        ValueError,
        match=f"spec {field} must be a non-empty project-relative path",
    ):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize("field", ["template", "scene"])
def test_scene_contract_rejects_missing_manifest_file_path(tmp_path, field):
    spec, manifest, _ = _write_contract(tmp_path)
    payload = _read_json(manifest)
    payload[field].pop("path")
    _write_json(manifest, payload)

    with pytest.raises(
        ValueError,
        match=f"manifest {field} path must be a non-empty project-relative path",
    ):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize("field", ["template", "scene"])
@pytest.mark.parametrize("bad_sha", ["A" * 64, "abc", None, True])
def test_scene_contract_rejects_invalid_manifest_sha256(
    tmp_path,
    field,
    bad_sha,
):
    spec, manifest, _ = _write_contract(tmp_path)
    payload = _read_json(manifest)
    payload[field]["sha256"] = bad_sha
    _write_json(manifest, payload)

    with pytest.raises(
        ValueError,
        match=(
            f"manifest {field} sha256 must be 64 lowercase hexadecimal "
            "characters"
        ),
    ):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize(
    "required_paths",
    [
        [],
        "/RobotBasics/Camera",
        [123],
        ["RobotBasics/Camera"],
        ["/RobotBasics/Camera", "/RobotBasics/Camera"],
        [" "],
    ],
)
@pytest.mark.parametrize("document", ["spec", "manifest"])
def test_scene_contract_rejects_invalid_required_paths(
    tmp_path,
    required_paths,
    document,
):
    spec, manifest, _ = _write_contract(tmp_path)
    path = spec if document == "spec" else manifest
    payload = _read_json(path)
    payload["required_paths"] = required_paths
    _write_json(path, payload)

    with pytest.raises(ValueError, match="required_paths"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


def test_scene_contract_rejects_required_path_mismatch(tmp_path):
    spec, manifest, _ = _write_contract(tmp_path)
    payload = _read_json(manifest)
    payload["required_paths"] = ["/Different"]
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="required_paths mismatch"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_scene_contract_requires_literal_protected_assets_true(tmp_path, value):
    spec, manifest, _ = _write_contract(tmp_path)
    payload = _read_json(manifest)
    payload["protected_assets_unchanged"] = value
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="protected assets were not verified"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "spec_field", "wrong_path", "message"),
    [
        ("template", "template", "other-template.ttt", "template path mismatch"),
        ("scene", "output", "other-scene.ttt", "scene path mismatch"),
    ],
)
def test_scene_contract_requires_manifest_paths_to_match_spec(
    tmp_path,
    field,
    spec_field,
    wrong_path,
    message,
):
    spec, manifest, _ = _write_contract(tmp_path)
    payload = _read_json(manifest)
    payload[field]["path"] = wrong_path
    _write_json(manifest, payload)

    assert _read_json(spec)[spec_field] != wrong_path
    with pytest.raises(ValueError, match=message):
        validate_scene_contract(spec, manifest, project_root=tmp_path)
