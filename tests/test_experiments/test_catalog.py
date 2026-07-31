from __future__ import annotations

import json

import pytest

from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.experiments.models import (
    CapabilityReport,
    ExperimentAcceptance,
    ExperimentDefinition,
    ExperimentRunContext,
)


def _write_experiment(root, *, experiment_id="R1-01", scene="scene.ttt"):
    experiments = root / "config" / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    guide = root / "docs" / "experiments" / f"{experiment_id}.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("# guide\n", encoding="utf-8")
    template = root / "student_programs" / "templates" / f"{experiment_id}.py"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("def main(ctx):\n    ctx.robot.home()\n", encoding="utf-8")
    scene_path = root / scene
    scene_path.write_bytes(b"scene")
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "pack_id": "R1",
        "title": "机械臂认知和基础操作",
        "version": "2.2.0",
        "scene": scene,
        "scene_manifest": "scene_manifest.json",
        "student_template": template.relative_to(root).as_posix(),
        "guide": guide.relative_to(root).as_posix(),
        "capabilities": ["robot.home", "robot.pose"],
        "workspace": {
            "x_mm": [20, 140],
            "y_mm": [-90, 90],
            "z_mm": [10, 140],
            "safe_z_mm": 100,
        },
        "public_parameters": {"observation_pose_mm": [100, 0, 120]},
        "acceptance": {
            "probe_kind": "motion_observation",
            "automated_checks": ["robot_paths", "command_trace"],
            "human_checks": ["学生能够解释六个关节"],
        },
        "hardware_status": "PENDING_HARDWARE",
    }
    path = experiments / f"{experiment_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (root / "scene_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene": {"path": scene, "sha256": "0" * 64},
                "required_paths": ["/BLX_base_link"],
            }
        ),
        encoding="utf-8",
    )
    (experiments / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": [f"{experiment_id}.json"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_catalog_loads_and_resolves_project_paths(tmp_path):
    _write_experiment(tmp_path)

    catalog = ExperimentCatalog.load(
        tmp_path / "config" / "experiments" / "catalog.json",
        project_root=tmp_path,
    )
    experiment = catalog.require("R1-01")

    assert experiment.experiment_id == "R1-01"
    assert experiment.scene == (tmp_path / "scene.ttt").resolve()
    assert experiment.student_template.name == "R1-01.py"
    assert experiment.hardware_status == "PENDING_HARDWARE"
    assert catalog.ids == ("R1-01",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("experiment_id", "r1-01", "experiment_id"),
        ("hardware_status", "PASS", "PENDING_HARDWARE"),
        ("capabilities", ["robot.home", "robot.home"], "duplicate"),
    ],
)
def test_catalog_rejects_invalid_contract(tmp_path, field, value, message):
    path = _write_experiment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ExperimentCatalog.load(
            tmp_path / "config" / "experiments" / "catalog.json",
            project_root=tmp_path,
        )


def test_catalog_rejects_paths_outside_project(tmp_path):
    path = _write_experiment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scene"] = "../outside.ttt"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inside project"):
        ExperimentCatalog.load(
            tmp_path / "config" / "experiments" / "catalog.json",
            project_root=tmp_path,
        )


def test_loaded_experiment_mappings_are_recursively_immutable(tmp_path):
    path = _write_experiment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["public_parameters"]["nested"] = {"values": [1, 2]}
    path.write_text(json.dumps(payload), encoding="utf-8")

    experiment = ExperimentCatalog.load(
        tmp_path / "config" / "experiments" / "catalog.json",
        project_root=tmp_path,
    ).require("R1-01")

    assert experiment.workspace["x_mm"] == (20, 140)
    assert experiment.public_parameters["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        experiment.workspace["x_mm"] = (-999, 999)
    with pytest.raises(TypeError):
        experiment.public_parameters["nested"]["values"][0] = 9


def test_directly_constructed_models_copy_and_freeze_nested_mappings(tmp_path):
    workspace = {"x_mm": [20, 140], "nested": {"values": [1, 2]}}
    public_parameters = {"nested": {"values": [3, 4]}}
    reasons = {"camera.rgb": {"details": ["offline"]}}
    acceptance = ExperimentAcceptance("probe", ("check",), ("human",))

    definition = ExperimentDefinition(
        experiment_id="R1-01",
        pack_id="R1",
        title="title",
        version="2.2.0",
        scene=tmp_path / "scene.ttt",
        scene_manifest=tmp_path / "manifest.json",
        student_template=tmp_path / "template.py",
        guide=tmp_path / "guide.md",
        capabilities=("robot.home",),
        workspace=workspace,
        public_parameters=public_parameters,
        acceptance=acceptance,
        hardware_status="PENDING_HARDWARE",
    )
    report = CapabilityReport((), ("camera.rgb",), reasons)
    context = ExperimentRunContext(
        experiment_id="R1-01",
        experiment_version="2.2.0",
        scene_path=tmp_path / "scene.ttt",
        scene_sha256="0" * 64,
        scene_manifest_path=tmp_path / "manifest.json",
        public_parameters=public_parameters,
    )

    workspace["x_mm"][0] = -999
    public_parameters["nested"]["values"].append(5)
    reasons["camera.rgb"]["details"].append("mutated")

    assert definition.workspace["x_mm"] == (20, 140)
    assert definition.workspace["nested"]["values"] == (1, 2)
    assert definition.public_parameters["nested"]["values"] == (3, 4)
    assert report.reasons["camera.rgb"]["details"] == ("offline",)
    assert context.public_parameters["nested"]["values"] == (3, 4)
    with pytest.raises(TypeError):
        report.reasons["camera.rgb"] = "changed"
    with pytest.raises(TypeError):
        context.public_parameters["nested"] = {}


def test_catalog_rejects_catalog_file_outside_project(tmp_path):
    project = tmp_path / "project"
    experiment_path = _write_experiment(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_experiment = outside / experiment_path.name
    outside_experiment.write_text(
        experiment_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    outside_catalog = outside / "catalog.json"
    outside_catalog.write_text(
        json.dumps({"schema_version": 1, "experiments": [experiment_path.name]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="catalog.*inside project"):
        ExperimentCatalog.load(outside_catalog, project_root=project)


def test_catalog_rejects_experiment_file_outside_project(tmp_path):
    project = tmp_path / "project"
    experiment_path = _write_experiment(project)
    outside_experiment = tmp_path / "outside.json"
    outside_experiment.write_text(
        experiment_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    catalog_path = project / "config" / "experiments" / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": [str(outside_experiment.resolve())],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="experiment.*inside project"):
        ExperimentCatalog.load(catalog_path, project_root=project)


@pytest.mark.parametrize(
    "catalog_payload",
    [
        {"schema_version": 1, "experiments": ["R1-01.json"], "extra": True},
        {"schema_version": 1, "experiments": []},
    ],
)
def test_catalog_rejects_non_strict_catalog_schema(tmp_path, catalog_payload):
    _write_experiment(tmp_path)
    catalog_path = tmp_path / "config" / "experiments" / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

    with pytest.raises(ValueError):
        ExperimentCatalog.load(catalog_path, project_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pack_id", 123),
        ("title", ""),
        ("version", ["2.2.0"]),
        ("scene", 123),
        ("guide", "   "),
    ],
)
def test_catalog_rejects_non_string_or_blank_string_fields(tmp_path, field, value):
    path = _write_experiment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        ExperimentCatalog.load(
            tmp_path / "config" / "experiments" / "catalog.json",
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["acceptance"].update({"extra": True}), "acceptance"),
        (lambda payload: payload["acceptance"].update({"probe_kind": 7}), "probe_kind"),
        (lambda payload: payload.update({"capabilities": []}), "capabilities"),
        (lambda payload: payload.update({"capabilities": ["   "]}), "capabilities"),
        (
            lambda payload: payload["acceptance"].update({"human_checks": []}),
            "human_checks",
        ),
        (lambda payload: payload.update({"public_parameters": []}), "public_parameters"),
        (lambda payload: payload.update({"workspace": []}), "workspace"),
    ],
)
def test_catalog_rejects_invalid_nested_schema(tmp_path, mutate, message):
    path = _write_experiment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ExperimentCatalog.load(
            tmp_path / "config" / "experiments" / "catalog.json",
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda workspace: workspace.update({"unexpected": 1}), "workspace"),
        (lambda workspace: workspace.pop("safe_z_mm"), "safe_z_mm"),
        (lambda workspace: workspace.update({"x_mm": [20]}), "x_mm"),
        (lambda workspace: workspace.update({"y_mm": ["low", 90]}), "y_mm"),
        (lambda workspace: workspace.update({"z_mm": [140, 10]}), "z_mm"),
        (lambda workspace: workspace.update({"safe_z_mm": True}), "safe_z_mm"),
        (lambda workspace: workspace.update({"safe_z_mm": 999}), "safe_z_mm"),
    ],
)
def test_catalog_rejects_invalid_workspace_contract(tmp_path, mutate, message):
    path = _write_experiment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload["workspace"])
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ExperimentCatalog.load(
            tmp_path / "config" / "experiments" / "catalog.json",
            project_root=tmp_path,
        )
