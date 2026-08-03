from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.models import ExperimentAcceptance, ExperimentDefinition, ExperimentRunContext
from vision_platform.student.experiment_gateway import StudentExperimentGateway


class Camera:
    def __init__(self) -> None:
        self.read_calls = 0

    def read(self, timeout_s: float):
        self.read_calls += 1
        return SimpleNamespace(
            image_bgr=np.zeros((1024, 1024, 3), dtype=np.uint8),
            width=1024,
            height=1024,
            timestamp_s=1.0,
            source="coppeliasim",
            sequence_id=self.read_calls,
        )


class Calls:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


class Evidence:
    def __init__(self, root: Path) -> None:
        self.directory = root
        self.directory.mkdir(parents=True)
        self.json_calls: list[tuple[str, dict]] = []

    def record_snapshot(self, **payload: object) -> dict:
        path = self.directory / "frames" / f"{payload['snapshot_id']}.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload["png_bytes"])
        return {"path": path.relative_to(self.directory).as_posix(), "snapshot_id": payload["snapshot_id"], **payload["metadata"]}

    def record_json_artifact(self, name: str, payload: dict) -> str:
        self.json_calls.append((name, payload))
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return name


class ProfileController:
    def current(self):
        from vision_platform.vision_quality.models import AppliedVisionProfile
        return AppliedVisionProfile("standard", (1024, 1024), 60.0, 0.7, (0.8, 0.8, 0.8), (0.35, 0.35, 0.35))


def _gateway(tmp_path: Path, *, activate=None):
    scene = tmp_path / "BL23_vision_defect_sorting_lab.ttt"
    scene.write_bytes(b"v1-09-scene")
    assets = tmp_path / "defect_assets_manifest.json"
    assets.write_bytes(b"{}")
    manifest = {
        "schema_version": 1,
        "scene": {"path": scene.name, "sha256": hashlib.sha256(scene.read_bytes()).hexdigest()},
        "task_contracts": {},
        "defect_assets_manifest": {"path": assets.name, "sha256": hashlib.sha256(assets.read_bytes()).hexdigest()},
    }
    manifest_path = tmp_path / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    public_parameters = {"defect_sorting": {"profile_id": "standard", "image_size_px": [1024, 1024]}}
    definition = ExperimentDefinition(
        experiment_id="V1-09", pack_id="V1", title="defect sorting", version="2.2.0",
        scene=scene, scene_manifest=manifest_path, student_template=tmp_path / "student.py", guide=tmp_path / "guide.md",
        capabilities=("camera.rgb", "camera.profile", "lighting.profile", "vision2d.surface_defects"),
        workspace={"x_mm": [20.0, 155.0], "y_mm": [-95.0, 95.0], "z_mm": [10.0, 140.0], "safe_z_mm": 110.0},
        public_parameters=public_parameters,
        acceptance=ExperimentAcceptance("defect_sort", (), ("teacher_review",)),
        hardware_status="PENDING_HARDWARE",
    )
    context = ExperimentRunContext("V1-09", "2.2.0", scene, manifest["scene"]["sha256"], manifest_path, public_parameters)
    camera = Camera()
    app = SimpleNamespace(config=SimpleNamespace(camera_backend="sim", robot_backend="sim"), sim=object(), camera=camera, robot=Calls(), tool=Calls())
    evidence = Evidence(tmp_path / "evidence")
    gateway = StudentExperimentGateway(
        application=app, evidence=evidence, context=context, definition=definition,
        scene_manifest=manifest, profile_controller=ProfileController(), defect_guard_activator=activate,
    )
    return gateway, camera, app, evidence


def test_v1_09_gateway_requires_narrow_capabilities_and_activates_complete_plan(tmp_path, monkeypatch):
    activated = []
    gateway, camera, app, evidence = _gateway(tmp_path, activate=activated.append)
    fake = SimpleNamespace(
        plan=SimpleNamespace(
            plan_id="a" * 64, run_id="run-1", frame_id="frame-1", scene_sha256="b" * 64,
            config_sha256="c" * 64, asset_manifest_sha256="d" * 64, image_size=(1024, 1024),
            safe_z_mm=110.0, speed_mm_s=15.0, entries=(), status="PASS", schema_version=1,
        ),
        results=(), observations=(), annotated_frame=np.zeros((1024, 1024, 3), dtype=np.uint8),
        reference_crop=np.zeros((192, 192, 3), dtype=np.uint8), candidate_crops={}, finding_masks={},
        manifest_sha256="d" * 64, scene_id="V1-09", config=SimpleNamespace(), run_id="run-1", frame_id="frame-1",
    )
    monkeypatch.setattr("vision_platform.student.experiment_gateway.DefectSortingService.from_manifest", lambda *args, **kwargs: SimpleNamespace(analyze=lambda *a, **k: fake))
    with pytest.raises(VisionPlatformError):
        gateway.dispatch("vision2d.surface_defects", {})
    assert camera.read_calls == 1
    assert app.robot.calls == []
    assert app.tool.calls == []
    assert activated == []
    assert evidence.json_calls == []


def test_v1_09_gateway_rejects_arguments_and_direct_entry_dispatch(tmp_path):
    gateway, camera, app, _ = _gateway(tmp_path, activate=lambda plan: None)
    with pytest.raises(ValueError, match="does not accept arguments"):
        gateway.dispatch("vision2d.surface_defects", {"roi": [0, 0, 1, 1]})
    with pytest.raises(VisionPlatformError) as captured:
        gateway.dispatch("vision2d.defect_sort_entry", {"entry_id": "entry_a"})
    assert captured.value.code == "DEFECT_SORT_ENTRY_RUNNER_ONLY"
    assert camera.read_calls == 0
    assert app.robot.calls == []
    assert app.tool.calls == []


def test_v1_09_runtime_files_do_not_reference_ground_truth(tmp_path):
    del tmp_path
    source = Path("vision_platform/experiments/defect_service.py")
    assert "acceptance_ground_truth" not in source.read_text(encoding="utf-8")
