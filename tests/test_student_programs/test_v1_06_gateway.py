from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.vision_lab.generate_v1_06_template import generate
from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.models import (
    ExperimentAcceptance,
    ExperimentDefinition,
    ExperimentRunContext,
)
from vision_platform.models import Frame
from vision_platform.student.experiment_gateway import StudentExperimentGateway
from vision_platform.experiments.capabilities import check_capabilities
from vision_platform.vision_quality.models import AppliedVisionProfile


class Camera:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.read_calls = 0

    def read(self, timeout_s: float) -> Frame:
        assert timeout_s == 2.0
        self.read_calls += 1
        height, width = self.image.shape[:2]
        return Frame(
            image_bgr=self.image.copy(),
            width=width,
            height=height,
            timestamp_s=1.0,
            source="coppeliasim",
            sequence_id=self.read_calls,
        )


class ProfileController:
    def current(self) -> AppliedVisionProfile:
        return AppliedVisionProfile(
            profile_id="standard",
            resolution=(512, 512),
            perspective_angle_deg=60,
            camera_rig_z_m=0.7,
            key_diffuse_rgb=(0.8, 0.8, 0.8),
            fill_diffuse_rgb=(0.35, 0.35, 0.35),
        )


class Evidence:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.calls: list[dict] = []
        self.json_calls: list[tuple[str, dict]] = []

    def record_snapshot(self, **payload: object) -> dict:
        self.calls.append(payload)
        path = self.directory / "frames" / f"{payload['snapshot_id']}.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload["png_bytes"])
        return {
            "snapshot_id": payload["snapshot_id"],
            "path": f"frames/{payload['snapshot_id']}.png",
            "sha256": hashlib.sha256(payload["png_bytes"]).hexdigest(),
            **payload["metadata"],
        }

    def record_json_artifact(self, name: str, payload: dict) -> str:
        self.json_calls.append((name, payload))
        (self.directory / name).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return name


def _gateway(tmp_path: Path):
    scene = tmp_path / "BL23_vision_quality_lab.ttt"
    scene.write_bytes(b"scene")
    template_dir = tmp_path / "templates"
    generated = generate(template_dir)
    template_manifest = dict(generated.manifest)
    template_manifest["asset_path"] = "templates/v1_06_reference.png"
    generated.manifest_path.write_text(
        json.dumps(template_manifest, ensure_ascii=False), encoding="utf-8"
    )
    scene_manifest = {
        "schema_version": 1,
        "scene": {
            "path": str(scene),
            "sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        },
        "task_contracts": {},
        "template_catalog": {
            "path": "templates/manifest.json",
            "sha256": hashlib.sha256(generated.manifest_path.read_bytes()).hexdigest(),
        },
    }
    scene_manifest_path = tmp_path / "scene_manifest.json"
    scene_manifest_path.write_text(
        json.dumps(scene_manifest, ensure_ascii=False), encoding="utf-8"
    )
    parameters = {"profile_id": "standard"}
    definition = ExperimentDefinition(
        experiment_id="V1-06",
        pack_id="V1",
        title="Template matching",
        version="2.2.0",
        scene=scene,
        scene_manifest=scene_manifest_path,
        student_template=tmp_path / "student.py",
        guide=tmp_path / "guide.md",
        capabilities=(
            "camera.rgb",
            "camera.profile",
            "lighting.profile",
            "vision2d.template_matching",
            "scene.probe",
        ),
        workspace={"safe_z_mm": 100},
        public_parameters=parameters,
        acceptance=ExperimentAcceptance(
            probe_kind="none",
            automated_checks=("template_match",),
            human_checks=("teacher_review",),
        ),
        hardware_status="PENDING_HARDWARE",
    )
    context = ExperimentRunContext(
        experiment_id="V1-06",
        experiment_version="2.2.0",
        scene_path=scene,
        scene_sha256=scene_manifest["scene"]["sha256"],
        scene_manifest_path=scene_manifest_path,
        public_parameters=parameters,
    )
    template = cv2.imread(str(template_dir / "v1_06_reference.png"), cv2.IMREAD_COLOR)
    assert template is not None
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[100 : 100 + template.shape[0], 100 : 100 + template.shape[1]] = template
    camera = Camera(image)
    evidence = Evidence(tmp_path / "evidence")
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=camera),
        evidence=evidence,
        context=context,
        definition=definition,
        scene_manifest=scene_manifest,
        profile_controller=ProfileController(),
    )
    return gateway, camera, evidence


def test_gateway_runs_fixed_template_match_and_records_three_layers(tmp_path: Path) -> None:
    gateway, camera, evidence = _gateway(tmp_path)

    value = gateway.dispatch("vision2d.template_match", {})

    assert value["template_id"] == "v1_06_red_rectangle"
    assert value["template_version"] == "1.0.0"
    assert value["matched"] is True
    assert value["bbox_px"] == [100, 100, 67, 40]
    assert value["center_px"] == [133.5, 120.0]
    assert value["search_roi_px"] == [0, 0, 512, 512]
    assert camera.read_calls == 1
    assert len(evidence.calls) == 3
    assert len(evidence.json_calls) == 1
    bundle = evidence.json_calls[0][1]
    assert [layer["layer_id"] for layer in bundle["layers"]] == [
        "raw",
        "template",
        "annotated",
    ]
    assert bundle["hardware_status"] == "PENDING_HARDWARE"
    assert "score" in bundle["result"]


def test_gateway_template_match_requires_capability_and_rejects_arguments(tmp_path: Path) -> None:
    gateway, camera, evidence = _gateway(tmp_path)
    with pytest.raises(ValueError, match="does not accept arguments"):
        gateway.dispatch("vision2d.template_match", {"threshold": 0.1})

    gateway._definition = gateway._definition.__class__(
        **{**gateway._definition.__dict__, "capabilities": ("camera.rgb",)}
    )
    with pytest.raises(VisionPlatformError) as captured:
        gateway.dispatch("vision2d.template_match", {})
    assert captured.value.code == "VISION_TEMPLATE_CONTEXT_REQUIRED"
    assert camera.read_calls == 0
    assert evidence.calls == []


def test_template_matching_capability_requires_sim_camera_and_profile_pair() -> None:
    application = SimpleNamespace(
        config=SimpleNamespace(camera_backend="sim"),
        sim=object(),
        camera=object(),
    )
    requested = (
        "camera.rgb",
        "camera.profile",
        "lighting.profile",
        "vision2d.template_matching",
    )
    assert check_capabilities(application, requested).ready

    incomplete = check_capabilities(
        application,
        ("camera.rgb", "vision2d.template_matching"),
    )
    assert incomplete.missing == ("vision2d.template_matching",)
    assert "同时声明" in incomplete.reasons["vision2d.template_matching"]

    replay = check_capabilities(
        SimpleNamespace(
            config=SimpleNamespace(camera_backend="replay"),
            sim=object(),
            camera=object(),
        ),
        requested,
    )
    assert "vision2d.template_matching" in replay.missing
    assert "CoppeliaSim" in replay.reasons["vision2d.template_matching"]
