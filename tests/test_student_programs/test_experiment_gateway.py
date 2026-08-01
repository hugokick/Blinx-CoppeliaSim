from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision_platform.experiments.models import (
    ExperimentAcceptance,
    ExperimentDefinition,
    ExperimentRunContext,
)
from vision_platform.models import Frame
from vision_platform.student.experiment_gateway import StudentExperimentGateway


class FakeCamera:
    def read(self, timeout_s):
        assert timeout_s == 2.0
        return Frame(
            image_bgr=np.full((6, 8, 3), 127, dtype=np.uint8),
            width=8,
            height=6,
            timestamp_s=12.5,
            source="coppeliasim",
            sequence_id=4,
        )


class FakeEvidence:
    def __init__(self):
        self.calls = []
        self.json_calls = []

    def record_snapshot(self, **payload):
        self.calls.append(payload)
        return {"path": f"frames/{payload['snapshot_id']}.png"}

    def record_json_artifact(self, name, payload):
        self.json_calls.append((name, payload))
        return name


def _bundle(
    tmp_path,
    *,
    experiment_id="R1-01",
    probe_kind="motion_observation",
    public_parameters=None,
    task_contracts=None,
):
    scene = tmp_path / "scene.ttt"
    scene.write_bytes(b"test-scene")
    scene_sha256 = hashlib.sha256(scene.read_bytes()).hexdigest()
    manifest_path = tmp_path / "scene_manifest.json"
    manifest = {
        "schema_version": 1,
        "scene": {
            "path": str(scene),
            "sha256": scene_sha256,
        },
        "task_contracts": dict(task_contracts or {}),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    parameters = dict(public_parameters or {})
    definition = ExperimentDefinition(
        experiment_id=experiment_id,
        pack_id="R1",
        title="Test experiment",
        version="2.2.0",
        scene=scene,
        scene_manifest=manifest_path,
        student_template=tmp_path / "student.py",
        guide=tmp_path / "guide.md",
        capabilities=("scene.probe",),
        workspace={
            "x_mm": (20, 140),
            "y_mm": (-90, 90),
            "z_mm": (10, 140),
            "safe_z_mm": 100,
        },
        public_parameters=parameters,
        acceptance=ExperimentAcceptance(
            probe_kind=probe_kind,
            automated_checks=("scene_probe",),
            human_checks=("teacher_review",),
        ),
        hardware_status="PENDING_HARDWARE",
    )
    context = ExperimentRunContext(
        experiment_id=experiment_id,
        experiment_version="2.2.0",
        scene_path=scene,
        scene_sha256=scene_sha256,
        scene_manifest_path=manifest_path,
        public_parameters=parameters,
    )
    return context, definition, manifest


def _gateway(tmp_path, *, application, evidence):
    context, definition, manifest = _bundle(tmp_path)
    return StudentExperimentGateway(
        application=application,
        evidence=evidence,
        context=context,
        definition=definition,
        scene_manifest=manifest,
    )


def test_file_bytes_seal_requires_exact_absolute_metadata(tmp_path):
    from vision_platform.student.experiment_gateway import FileBytesSeal

    path = (tmp_path / "manifest.json").resolve()
    content = b'{"schema_version":1}'
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    seal = FileBytesSeal.capture(path)

    assert seal.path == path
    assert seal.size == len(content)
    assert seal.sha256 == digest
    assert seal.content == content
    with pytest.raises(TypeError, match="path"):
        FileBytesSeal(
            path=str(path),
            size=len(content),
            sha256=digest,
            content=content,
        )
    with pytest.raises(ValueError, match="absolute"):
        FileBytesSeal(
            path=path.relative_to(tmp_path),
            size=len(content),
            sha256=digest,
            content=content,
        )
    with pytest.raises(ValueError, match="sha256"):
        FileBytesSeal(
            path=path,
            size=len(content),
            sha256="f" * 64,
            content=content,
        )


def test_gateway_captures_png_and_records_same_bytes(tmp_path):
    evidence = FakeEvidence()
    gateway = _gateway(
        tmp_path,
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=evidence,
    )

    value = gateway.dispatch("camera.capture", {})

    assert value["snapshot_id"] == "frame-000001"
    assert value["width"] == 8
    assert value["height"] == 6
    assert value["source"] == "coppeliasim"
    assert value["sequence_id"] == 4
    assert value["png_bytes"].startswith(b"\x89PNG\r\n\x1a\n")
    assert evidence.calls[0]["png_bytes"] is value["png_bytes"]
    assert evidence.calls[0]["metadata"] == {
        "width": 8,
        "height": 6,
        "timestamp_s": 12.5,
        "source": "coppeliasim",
        "sequence_id": 4,
    }
    decoded = cv2.imdecode(
        np.frombuffer(value["png_bytes"], dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    assert decoded.shape == (6, 8, 3)
    assert value["evidence_path"] == "frames/frame-000001.png"


def test_gateway_returns_public_experiment_data_only_and_json_native(tmp_path):
    context, definition, manifest = _bundle(
        tmp_path,
        public_parameters={"safe_z_mm": 100, "labels": ("A", "B")},
    )
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(),
        context=context,
        definition=definition,
        scene_manifest=manifest,
    )

    value = gateway.dispatch("experiment.info", {})

    assert value == {
        "experiment_id": "R1-01",
        "experiment_version": "2.2.0",
        "scene_sha256": context.scene_sha256,
        "public_parameters": {"safe_z_mm": 100, "labels": ["A", "B"]},
        "hardware_status": "PENDING_HARDWARE",
    }
    assert json.loads(json.dumps(value, allow_nan=False)) == value
    assert "scene_path" not in value
    assert "scene_manifest_path" not in value
    assert "handle" not in value
    assert "sim" not in value


@pytest.mark.parametrize("name", ["camera.capture", "experiment.info"])
def test_gateway_read_only_commands_reject_all_arguments(tmp_path, name):
    gateway = _gateway(
        tmp_path,
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(),
    )

    with pytest.raises(ValueError, match="does not accept arguments"):
        gateway.dispatch(name, {"unexpected": True})


def test_gateway_rejects_unknown_command_without_dynamic_dispatch(tmp_path):
    gateway = _gateway(
        tmp_path,
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(),
    )

    with pytest.raises(ValueError, match="COMMAND_NOT_ALLOWED"):
        gateway.dispatch("sim.getObject", {})


class ProbeSim:
    handle_world = -1

    def __init__(self, positions):
        self.positions = dict(positions)

    def getObject(self, path):
        if path not in self.positions:
            raise RuntimeError(f"missing: {path}")
        return path

    def getObjectPosition(self, handle, relative_to):
        assert relative_to == self.handle_world
        return self.positions[handle]


def test_gateway_copies_manifest_before_recording_initial_probe(tmp_path):
    objects = [
        {
            "alias": f"stack_{index:02d}",
            "position_mm": [40 + index * 10, -60, 20],
        }
        for index in range(1, 7)
    ]
    parameters = {
        "scene_group_path": "/LogisticsLab/Tasks/Stack",
        "stack_slots_mm": [[118, -45, 20]] * 6,
    }
    context, definition, manifest = _bundle(
        tmp_path,
        experiment_id="R1-05",
        probe_kind="stack_2x3",
        public_parameters=parameters,
        task_contracts={"Stack": {"objects": objects}},
    )
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/{item['alias']}": [
            value / 1000 for value in item["position_mm"]
        ]
        for item in objects
    }
    evidence = FakeEvidence()
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(sim=ProbeSim(positions)),
        evidence=evidence,
        context=context,
        definition=definition,
        scene_manifest=manifest,
    )
    manifest["task_contracts"]["Stack"]["objects"][0][
        "position_mm"
    ] = [999, 999, 999]

    report = gateway.record_probe("initial")

    assert report["status"] == "PASS"
    assert evidence.json_calls == [("scene-initial.json", report)]


def test_gateway_records_error_report_when_probe_raises(tmp_path):
    context, definition, manifest = _bundle(tmp_path)
    evidence = FakeEvidence()
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(sim=ProbeSim({})),
        evidence=evidence,
        context=context,
        definition=definition,
        scene_manifest=manifest,
    )

    report = gateway.record_probe("final")

    assert report["status"] == "ERROR"
    assert report["phase"] == "final"
    assert report["hardware_status"] == "PENDING_HARDWARE"
    assert report["error"]["code"] == "SCENE_PROBE_FAILED"
    assert evidence.json_calls == [("scene-final.json", report)]
    assert "grade" not in report
    assert "score" not in report
