from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision_platform.experiments.models import ExperimentRunContext
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

    def record_snapshot(self, **payload):
        self.calls.append(payload)
        return {"path": f"frames/{payload['snapshot_id']}.png"}


def _context(tmp_path, **overrides):
    values = {
        "experiment_id": "R1-05",
        "experiment_version": "2.2.0",
        "scene_path": tmp_path / "scene.ttt",
        "scene_sha256": "a" * 64,
        "scene_manifest_path": tmp_path / "scene_manifest.json",
        "public_parameters": {"safe_z_mm": 100},
    }
    values.update(overrides)
    return ExperimentRunContext(**values)


def test_gateway_captures_png_and_records_same_bytes(tmp_path):
    evidence = FakeEvidence()
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=evidence,
        context=_context(tmp_path),
        capture_timeout_s=2.0,
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
    context = _context(
        tmp_path,
        public_parameters={"safe_z_mm": 100, "labels": ("A", "B")},
        hardware_status="PASS",
    )
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(),
        context=context,
    )

    value = gateway.dispatch("experiment.info", {})

    assert value == {
        "experiment_id": "R1-05",
        "experiment_version": "2.2.0",
        "scene_sha256": "a" * 64,
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
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(),
        context=_context(tmp_path),
    )

    with pytest.raises(ValueError, match="does not accept arguments"):
        gateway.dispatch(name, {"unexpected": True})


def test_gateway_rejects_unknown_command_without_dynamic_dispatch(tmp_path):
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(),
        context=_context(tmp_path),
    )

    with pytest.raises(ValueError, match="COMMAND_NOT_ALLOWED"):
        gateway.dispatch("sim.getObject", {})
