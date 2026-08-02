from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import vision_platform.vision_quality.controller as controller_module

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.models import (
    ExperimentAcceptance,
    ExperimentDefinition,
    ExperimentRunContext,
)
from vision_platform.models import Frame
from vision_platform.student.experiment_gateway import StudentExperimentGateway
from vision_platform.vision_quality.catalog import load_profile_catalog_bytes
from vision_platform.vision_quality.models import AppliedVisionProfile
from vision_platform.vision_quality.models import VisionResultEvidenceError


_UNSET = object()


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


class FakeVision2DCamera:
    def __init__(self):
        self.read_calls = 0
        self.image = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(self.image, (150, 160), 22, (0, 0, 255), -1)
        cv2.rectangle(self.image, (220, 140), (285, 180), (0, 255, 0), -1)
        cv2.fillConvexPoly(
            self.image,
            np.array([[315, 185], [340, 135], [365, 185]], dtype=np.int32),
            (255, 0, 0),
        )
        cv2.circle(self.image, (470, 470), 25, (0, 255, 255), -1)

    def read(self, timeout_s):
        assert timeout_s == 2.0
        self.read_calls += 1
        return Frame(
            image_bgr=self.image.copy(),
            width=512,
            height=512,
            timestamp_s=13.5,
            source="coppeliasim",
            sequence_id=8,
        )


class FakeEvidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.json_calls = []

    def record_snapshot(self, **payload):
        self.calls.append(payload)
        frames = self.directory / "frames"
        frames.mkdir(exist_ok=True)
        path = frames / f"{payload['snapshot_id']}.png"
        path.write_bytes(payload["png_bytes"])
        metadata = payload["metadata"]
        return {
            "snapshot_id": payload["snapshot_id"],
            "path": f"frames/{payload['snapshot_id']}.png",
            "sha256": hashlib.sha256(payload["png_bytes"]).hexdigest(),
            **metadata,
        }

    def record_json_artifact(self, name, payload):
        self.json_calls.append((name, payload))
        (self.directory / name).write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )
        return name


class FakeProfileController:
    def __init__(self, *, resolution=(8, 6), public_override=None):
        self.profile_id = "standard"
        self.resolution = resolution
        self.public_override = public_override
        self.current_calls = 0
        self.apply_calls = []
        self.reset_calls = 0

    def _state(self):
        if self.public_override is not None:
            return SimpleNamespace(to_public_dict=lambda: self.public_override)
        return AppliedVisionProfile(
            profile_id=self.profile_id,
            resolution=self.resolution,
            perspective_angle_deg=60,
            camera_rig_z_m=0.7,
            key_diffuse_rgb=(0.8, 0.8, 0.8),
            fill_diffuse_rgb=(0.35, 0.35, 0.35),
        )

    def current(self):
        self.current_calls += 1
        return self._state()

    def apply(self, profile_id):
        self.apply_calls.append(profile_id)
        if profile_id not in {"standard", "wide_dim", "detail_bright"}:
            raise ValueError("VISION_PROFILE_NOT_ALLOWED")
        self.profile_id = profile_id
        return self._state()

    def reset(self):
        self.reset_calls += 1
        self.profile_id = "standard"
        return self._state()


def _bundle(
    tmp_path,
    *,
    experiment_id="R1-01",
    probe_kind="motion_observation",
    public_parameters=None,
    task_contracts=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def _profile_gateway(
    tmp_path,
    *,
    controller=None,
    camera=None,
    evidence=None,
):
    context, definition, manifest = _bundle(
        tmp_path,
        experiment_id="V1-01",
        public_parameters={
            "baseline_profile_id": "standard",
            "allowed_profile_ids": [
                "standard",
                "wide_dim",
                "detail_bright",
            ],
        },
    )
    definition = replace(
        definition,
        capabilities=(
            "camera.rgb",
            "camera.profile",
            "lighting.profile",
            "scene.probe",
        ),
    )
    selected_controller = (
        FakeProfileController() if controller is None else controller
    )
    selected_evidence = evidence or FakeEvidence(tmp_path / "evidence")
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=camera or FakeCamera()),
        evidence=selected_evidence,
        context=context,
        definition=definition,
        scene_manifest=manifest,
        profile_controller=selected_controller,
    )
    return gateway, selected_controller, selected_evidence


def _vision2d_parameters(**vision_overrides):
    vision = {
        "profile_id": "standard",
        "roi_px": [100, 100, 400, 220],
        "min_area_ratio": 0.002,
        "max_area_ratio": 0.05,
        "saturation_min": 60,
        "value_min": 40,
        "pixel_scale_mm": [1.4, 1.4],
    }
    vision.update(vision_overrides)
    return {"vision2d": vision, "analysis_focus": "size"}


def _vision2d_gateway(
    tmp_path,
    *,
    parameters=None,
    capabilities=None,
    controller=None,
    camera=None,
    evidence=None,
):
    context, definition, manifest = _bundle(
        tmp_path,
        experiment_id="V1-02",
        public_parameters=parameters or _vision2d_parameters(),
    )
    definition = replace(
        definition,
        capabilities=(
            capabilities
            or (
                "camera.rgb",
                "camera.profile",
                "lighting.profile",
                "vision2d.analysis",
                "scene.probe",
            )
        ),
    )
    selected_camera = camera or FakeVision2DCamera()
    selected_controller = controller or FakeProfileController(
        resolution=(512, 512)
    )
    selected_evidence = evidence or FakeEvidence(tmp_path / "evidence")
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=selected_camera),
        evidence=selected_evidence,
        context=context,
        definition=definition,
        scene_manifest=manifest,
        profile_controller=selected_controller,
    )
    return gateway, selected_camera, selected_controller, selected_evidence


def _profile_catalog_payload():
    return {
        "schema_version": 1,
        "baseline_profile_id": "standard",
        "sensor_path": "/VisionQualityLab/CameraRig/Camera",
        "camera_rig_path": "/VisionQualityLab/CameraRig",
        "key_light_path": "/VisionQualityLab/Lighting/KeyLight",
        "fill_light_path": "/VisionQualityLab/Lighting/FillLight",
        "near_clip_m": 0.05,
        "far_clip_m": 2.0,
        "profiles": [
            {
                "profile_id": "standard",
                "label": "标准视图",
                "resolution": [512, 512],
                "perspective_angle_deg": 60,
                "camera_rig_z_m": 0.7,
                "key_diffuse_rgb": [0.8, 0.8, 0.8],
                "fill_diffuse_rgb": [0.35, 0.35, 0.35],
            },
            {
                "profile_id": "wide_dim",
                "label": "宽视场弱光",
                "resolution": [256, 256],
                "perspective_angle_deg": 75,
                "camera_rig_z_m": 0.8,
                "key_diffuse_rgb": [0.35, 0.35, 0.35],
                "fill_diffuse_rgb": [0.15, 0.15, 0.15],
            },
        ],
    }


def _factory_bundle(
    tmp_path,
    *,
    allowed_profile_ids=_UNSET,
    baseline_profile_id=_UNSET,
):
    project = tmp_path / "project"
    (project / "config" / "experiments").mkdir(parents=True)
    scene_dir = project / "simulation" / "vision_quality_lab"
    scene_dir.mkdir(parents=True)
    scene = scene_dir / "BL23_vision_quality_lab.ttt"
    scene.write_bytes(b"vision-quality-scene")
    profile_path = scene_dir / "profiles.json"
    profile_path.write_text(
        json.dumps(_profile_catalog_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    selected_allowed = (
        ["standard", "wide_dim"]
        if allowed_profile_ids is _UNSET
        else allowed_profile_ids
    )
    selected_baseline = (
        "standard"
        if baseline_profile_id is _UNSET
        else baseline_profile_id
    )
    _, definition, _ = _bundle(
        tmp_path,
        experiment_id="V1-01",
        public_parameters={
            "baseline_profile_id": selected_baseline,
            "allowed_profile_ids": selected_allowed,
            "camera_path": "/VisionQualityLab/CameraRig/Camera",
        },
    )
    definition = replace(
        definition,
        scene=scene,
        capabilities=("camera.profile", "lighting.profile"),
    )
    manifest = {
        "profile_catalog": {
            "path": "simulation/vision_quality_lab/profiles.json",
            "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        }
    }
    return definition, manifest, profile_path


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
    evidence = FakeEvidence(tmp_path / "evidence")
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
        evidence=FakeEvidence(tmp_path / "evidence"),
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
        evidence=FakeEvidence(tmp_path / "evidence"),
    )

    with pytest.raises(ValueError, match="does not accept arguments"):
        gateway.dispatch(name, {"unexpected": True})


def test_gateway_rejects_unknown_command_without_dynamic_dispatch(tmp_path):
    gateway = _gateway(
        tmp_path,
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(tmp_path / "evidence"),
    )

    with pytest.raises(ValueError, match="COMMAND_NOT_ALLOWED"):
        gateway.dispatch("sim.getObject", {})


def test_gateway_dispatches_profile_commands_and_reuses_raw_capture(tmp_path):
    gateway, controller, evidence = _profile_gateway(tmp_path)

    applied = gateway.dispatch(
        "camera.profile.apply",
        {"profile_id": "wide_dim"},
    )
    current = gateway.dispatch("camera.profile.get", {})
    captured = gateway.dispatch("camera.capture", {})
    reset = gateway.dispatch("camera.profile.reset", {})

    assert applied["profile_id"] == "wide_dim"
    assert current == applied
    assert reset["profile_id"] == "standard"
    assert controller.apply_calls == ["wide_dim"]
    assert controller.reset_calls == 1
    assert captured["vision_bundle_path"].startswith("vision-bundle-")
    assert len(evidence.calls) == 1
    assert evidence.calls[0]["metadata"]["vision_profile"] == applied
    bundle_name, bundle = evidence.json_calls[-1]
    assert bundle_name == captured["vision_bundle_path"]
    assert bundle["status"] == "PASS"
    assert bundle["hardware_status"] == "PENDING_HARDWARE"
    assert bundle["profile"] == applied
    assert bundle["layers"] == [
        {
            "layer_id": "raw",
            "title": "原图",
            "path": "frames/frame-000001.png",
            "sha256": hashlib.sha256(captured["png_bytes"]).hexdigest(),
            "width": 8,
            "height": 6,
        }
    ]


def test_gateway_runs_controlled_vision2d_analysis_and_records_five_layers(
    tmp_path,
):
    gateway, camera, _, evidence = _vision2d_gateway(tmp_path)

    value = gateway.dispatch("vision2d.analyze", {})

    assert set(value) == {
        "snapshot_id",
        "vision_bundle_path",
        "profile_id",
        "status",
        "image_size",
        "targets",
        "rejected_targets",
    }
    assert value["snapshot_id"] == "frame-000001"
    assert value["profile_id"] == "standard"
    assert value["status"] == "PASS"
    assert value["image_size"] == [512, 512]
    assert len(value["targets"]) == 3
    assert max(target["center_px"][0] for target in value["targets"]) < 400
    assert camera.read_calls == 1

    assert len(evidence.calls) == 5
    assert len(evidence.json_calls) == 1
    artifact_name, bundle = evidence.json_calls[0]
    assert artifact_name == value["vision_bundle_path"]
    assert bundle["status"] == "PASS"
    assert bundle["hardware_status"] == "PENDING_HARDWARE"
    assert bundle["result"]["schema_version"] == 1
    assert bundle["profile"]["profile_id"] == "standard"
    assert bundle["profile"]["vision2d"] == _vision2d_parameters()[
        "vision2d"
    ]
    assert [layer["layer_id"] for layer in bundle["layers"]] == [
        "raw",
        "roi-input",
        "foreground-mask",
        "cleaned-mask",
        "annotated",
    ]
    assert bundle["layers"][0]["path"] == "frames/frame-000001.png"
    assert len({layer["path"] for layer in bundle["layers"]}) == 5
    assert all(
        [layer["width"], layer["height"]] == [512, 512]
        for layer in bundle["layers"]
    )


def test_gateway_vision2d_analysis_rejects_arguments_and_missing_capability(
    tmp_path,
):
    gateway, _, _, _ = _vision2d_gateway(tmp_path / "arguments")
    with pytest.raises(ValueError, match="does not accept arguments"):
        gateway.dispatch("vision2d.analyze", {"roi_px": [0, 0, 1, 1]})

    absent, camera, _, evidence = _vision2d_gateway(
        tmp_path / "absent",
        capabilities=("camera.rgb", "camera.profile", "lighting.profile"),
    )
    with pytest.raises(VisionPlatformError) as captured:
        absent.dispatch("vision2d.analyze", {})
    assert captured.value.code == "VISION2D_CONTEXT_REQUIRED"
    assert camera.read_calls == 0
    assert evidence.calls == []


def test_gateway_vision2d_analysis_fails_closed_for_config_and_profile(
    tmp_path,
):
    invalid, camera, _, evidence = _vision2d_gateway(
        tmp_path / "invalid",
        parameters=_vision2d_parameters(extra=True),
    )
    with pytest.raises(VisionPlatformError) as invalid_error:
        invalid.dispatch("vision2d.analyze", {})
    assert invalid_error.value.code == "VISION2D_CONFIG_INVALID"
    assert camera.read_calls == 0
    assert evidence.calls == []

    controller = FakeProfileController(resolution=(512, 512))
    controller.profile_id = "wide_dim"
    mismatch, camera, _, evidence = _vision2d_gateway(
        tmp_path / "profile",
        controller=controller,
    )
    with pytest.raises(VisionPlatformError) as profile_error:
        mismatch.dispatch("vision2d.analyze", {})
    assert profile_error.value.code == "VISION2D_PROFILE_MISMATCH"
    assert camera.read_calls == 0
    assert evidence.calls == []


def test_profile_capture_keeps_raw_snapshot_when_bundle_json_write_fails(
    tmp_path,
):
    class FailingBundleEvidence(FakeEvidence):
        def record_json_artifact(self, name, payload):
            self.json_calls.append((name, payload))
            raise OSError("bundle-json-write-failed")

    evidence = FailingBundleEvidence(tmp_path / "evidence")
    gateway, _, _ = _profile_gateway(tmp_path, evidence=evidence)

    with pytest.raises(VisionResultEvidenceError, match="bundle-json-write-failed"):
        gateway.dispatch("camera.capture", {})

    assert len(evidence.calls) == 1
    raw_record = evidence.calls[0]
    raw_path = evidence.directory / "frames" / "frame-000001.png"
    assert raw_path.read_bytes() == raw_record["png_bytes"]
    assert raw_record["metadata"]["vision_profile"]["profile_id"] == (
        "standard"
    )
    assert len(evidence.json_calls) == 1
    assert list(evidence.directory.glob("vision-bundle-*.json")) == []


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("camera.profile.get", {"unexpected": True}),
        ("camera.profile.apply", {}),
        (
            "camera.profile.apply",
            {"profile_id": "wide_dim", "unexpected": True},
        ),
        ("camera.profile.reset", {"unexpected": True}),
    ],
)
def test_gateway_profile_commands_require_exact_arguments(
    tmp_path,
    name,
    args,
):
    gateway, _, _ = _profile_gateway(tmp_path)

    with pytest.raises(ValueError, match="arguments"):
        gateway.dispatch(name, args)


def test_gateway_rejects_unallowed_profile_id(tmp_path):
    gateway, _, _ = _profile_gateway(tmp_path)

    with pytest.raises(ValueError, match="VISION_PROFILE_NOT_ALLOWED"):
        gateway.dispatch(
            "camera.profile.apply",
            {"profile_id": "teacher_only"},
        )


def test_gateway_rejects_profile_commands_for_r1_even_if_injected(tmp_path):
    context, definition, manifest = _bundle(tmp_path)
    controller = FakeProfileController()
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(tmp_path / "evidence"),
        context=context,
        definition=definition,
        scene_manifest=manifest,
        profile_controller=controller,
    )

    with pytest.raises(VisionPlatformError) as captured:
        gateway.dispatch("camera.profile.get", {})
    assert captured.value.code == "VISION_PROFILE_CONTEXT_REQUIRED"
    assert controller.current_calls == 0


def test_gateway_rejects_profile_commands_when_controller_is_absent(tmp_path):
    context, definition, manifest = _bundle(
        tmp_path,
        experiment_id="V1-01",
    )
    definition = replace(
        definition,
        capabilities=("camera.profile", "lighting.profile"),
    )
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(tmp_path / "evidence"),
        context=context,
        definition=definition,
        scene_manifest=manifest,
        profile_controller=None,
    )

    with pytest.raises(VisionPlatformError) as captured:
        gateway.dispatch("camera.profile.reset", {})
    assert captured.value.code == "VISION_PROFILE_CONTEXT_REQUIRED"


def test_gateway_rejects_non_json_profile_controller_output(tmp_path):
    controller = FakeProfileController(
        public_override={"profile_id": "standard", "handle": object()},
    )
    gateway, _, _ = _profile_gateway(tmp_path, controller=controller)

    with pytest.raises(TypeError, match="JSON-native"):
        gateway.dispatch("camera.profile.get", {})


def test_profile_capture_rejects_resolution_mismatch_before_recording(tmp_path):
    evidence = FakeEvidence(tmp_path / "evidence")
    controller = FakeProfileController(resolution=(9, 6))
    gateway, _, _ = _profile_gateway(
        tmp_path,
        controller=controller,
        evidence=evidence,
    )

    with pytest.raises(RuntimeError, match="VISION_PROFILE_RESOLUTION_MISMATCH"):
        gateway.dispatch("camera.capture", {})
    assert evidence.calls == []
    assert evidence.json_calls == []


def test_gateway_reset_environment_is_public_and_optional(tmp_path):
    gateway, controller, _ = _profile_gateway(tmp_path)

    value = gateway.reset_environment()

    assert value["profile_id"] == "standard"
    assert controller.reset_calls == 1

    context, definition, manifest = _bundle(tmp_path / "without-profile")
    absent = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(tmp_path / "without-profile" / "evidence"),
        context=context,
        definition=definition,
        scene_manifest=manifest,
        profile_controller=None,
    )
    assert absent.reset_environment() is None


def test_controller_factory_skips_definition_without_both_capabilities(tmp_path):
    _, definition, manifest = _bundle(tmp_path)

    result = controller_module.controller_for_experiment(
        object(),
        definition,
        manifest,
    )

    assert result is None


@pytest.mark.parametrize("capability", ["camera.profile", "lighting.profile"])
def test_controller_factory_rejects_partial_profile_capability_pair(
    tmp_path,
    capability,
):
    _, definition, manifest = _bundle(tmp_path)
    definition = replace(definition, capabilities=(capability,))

    with pytest.raises(VisionPlatformError) as captured:
        controller_module.controller_for_experiment(
            object(),
            definition,
            manifest,
        )

    assert isinstance(captured.value, ValueError)
    assert captured.value.code == "VISION_PROFILE_CONTEXT_REQUIRED"


@pytest.mark.parametrize(
    "application",
    [
        SimpleNamespace(
            config=SimpleNamespace(camera_backend="replay"),
            sim=object(),
            camera=object(),
        ),
        SimpleNamespace(
            config=SimpleNamespace(camera_backend="sim"),
            sim=None,
            camera=object(),
        ),
        SimpleNamespace(
            config=SimpleNamespace(camera_backend="sim"),
            sim=object(),
            camera=None,
        ),
    ],
)
def test_controller_factory_has_stable_backend_unavailable_error(
    tmp_path,
    application,
):
    definition, manifest, _ = _factory_bundle(tmp_path)

    with pytest.raises(VisionPlatformError) as captured:
        controller_module.controller_for_experiment(
            application,
            definition,
            manifest,
        )
    assert isinstance(captured.value, RuntimeError)
    assert captured.value.code == "VISION_PROFILE_BACKEND_UNAVAILABLE"


def test_controller_validation_errors_keep_legacy_types_and_stable_codes():
    with pytest.raises(VisionPlatformError) as invalid_catalog:
        controller_module.VisionProfileController(
            sim=object(),
            camera=object(),
            catalog=object(),
            allowed_profile_ids=("standard",),
        )
    assert isinstance(invalid_catalog.value, TypeError)
    assert invalid_catalog.value.code == "VISION_PROFILE_CATALOG_INVALID"

    catalog = load_profile_catalog_bytes(
        json.dumps(
            _profile_catalog_payload(),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    with pytest.raises(VisionPlatformError) as invalid_allowed:
        controller_module.VisionProfileController(
            sim=object(),
            camera=object(),
            catalog=catalog,
            allowed_profile_ids=(),
        )
    assert isinstance(invalid_allowed.value, ValueError)
    assert invalid_allowed.value.code == "VISION_PROFILE_NOT_ALLOWED"

    controller = object.__new__(controller_module.VisionProfileController)
    controller._catalog = catalog
    controller._allowed_profile_ids = ("standard",)
    for profile_id, code in (
        (1, "VISION_PROFILE_ID_INVALID"),
        ("wide_dim", "VISION_PROFILE_NOT_ALLOWED"),
    ):
        with pytest.raises(VisionPlatformError) as captured:
            controller.apply(profile_id)
        assert isinstance(captured.value, ValueError)
        assert captured.value.code == code


def test_controller_factory_validates_root_and_constructs_exact_context(
    tmp_path,
    monkeypatch,
):
    definition, manifest, profile_path = _factory_bundle(tmp_path)
    application = SimpleNamespace(
        config=SimpleNamespace(camera_backend="sim"),
        sim=object(),
        camera=object(),
    )
    captured = {}
    sentinel = object()

    def construct(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(controller_module, "VisionProfileController", construct)

    result = controller_module.controller_for_experiment(
        application,
        definition,
        manifest,
    )

    assert result is sentinel
    assert captured["sim"] is application.sim
    assert captured["camera"] is application.camera
    assert captured["allowed_profile_ids"] == ("standard", "wide_dim")
    assert captured["catalog"].baseline_profile_id == "standard"
    assert profile_path.is_file()


def test_controller_factory_hashes_and_parses_the_same_profile_bytes(
    tmp_path,
    monkeypatch,
):
    definition, manifest, profile_path = _factory_bundle(tmp_path)
    application = SimpleNamespace(
        config=SimpleNamespace(camera_backend="sim"),
        sim=object(),
        camera=object(),
    )
    original = profile_path.read_bytes()
    swapped = b"{}"
    sentinel = object()
    original_read_bytes = Path.read_bytes

    def swap_after_read(self):
        content = original_read_bytes(self)
        if self.resolve() == profile_path.resolve():
            profile_path.write_bytes(swapped)
        return content

    monkeypatch.setattr(Path, "read_bytes", swap_after_read)
    monkeypatch.setattr(
        controller_module,
        "VisionProfileController",
        lambda **kwargs: sentinel,
    )

    assert controller_module.controller_for_experiment(
        application,
        definition,
        manifest,
    ) is sentinel
    assert profile_path.read_bytes() == swapped


@pytest.mark.parametrize(
    "baseline_profile_id",
    [None, "wide_dim", 1],
)
def test_controller_factory_requires_exact_catalog_baseline_parameter(
    tmp_path,
    baseline_profile_id,
):
    definition, manifest, _ = _factory_bundle(
        tmp_path,
        baseline_profile_id=baseline_profile_id,
    )

    with pytest.raises(VisionPlatformError) as captured:
        controller_module.controller_for_experiment(
            SimpleNamespace(
                config=SimpleNamespace(camera_backend="sim"),
                sim=object(),
                camera=object(),
            ),
            definition,
            manifest,
        )
    assert isinstance(captured.value, ValueError)
    assert captured.value.code == "VISION_PROFILE_CONTEXT_REQUIRED"


def test_controller_factory_rejects_unverified_project_root(tmp_path):
    definition, manifest, profile_path = _factory_bundle(tmp_path)
    outside = tmp_path / "outside" / "vision_quality_lab"
    outside.mkdir(parents=True)
    scene = outside / "scene.ttt"
    scene.write_bytes(b"outside-scene")
    (outside / "profiles.json").write_bytes(profile_path.read_bytes())
    definition = replace(definition, scene=scene)

    with pytest.raises(ValueError, match="VISION_PROFILE_CONTEXT_REQUIRED"):
        controller_module.controller_for_experiment(
            SimpleNamespace(
                config=SimpleNamespace(camera_backend="sim"),
                sim=object(),
                camera=object(),
            ),
            definition,
            manifest,
        )


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {
            "path": "simulation/vision_quality_lab/wrong.json",
            "sha256": "0" * 64,
        },
        {
            "path": "simulation/vision_quality_lab/profiles.json",
            "sha256": "0" * 64,
        },
        {
            "path": "simulation/vision_quality_lab/profiles.json",
            "sha256": "0" * 64,
            "extra": True,
        },
    ],
)
def test_controller_factory_rejects_invalid_catalog_binding(tmp_path, entry):
    definition, manifest, profile_path = _factory_bundle(tmp_path)
    if isinstance(entry, dict) and "extra" in entry:
        entry = dict(entry)
        entry["sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    manifest["profile_catalog"] = entry

    with pytest.raises(ValueError, match="VISION_PROFILE_CONTEXT_REQUIRED"):
        controller_module.controller_for_experiment(
            SimpleNamespace(
                config=SimpleNamespace(camera_backend="sim"),
                sim=object(),
                camera=object(),
            ),
            definition,
            manifest,
        )


@pytest.mark.parametrize(
    "allowed_profile_ids",
    [None, "standard", [], ["standard", 1]],
)
def test_controller_factory_rejects_invalid_allowed_profile_field(
    tmp_path,
    allowed_profile_ids,
):
    definition, manifest, _ = _factory_bundle(
        tmp_path,
        allowed_profile_ids=allowed_profile_ids,
    )

    with pytest.raises(ValueError, match="VISION_PROFILE_CONTEXT_REQUIRED"):
        controller_module.controller_for_experiment(
            SimpleNamespace(
                config=SimpleNamespace(camera_backend="sim"),
                sim=object(),
                camera=object(),
            ),
            definition,
            manifest,
        )


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
    evidence = FakeEvidence(tmp_path / "evidence")
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
    evidence = FakeEvidence(tmp_path / "evidence")
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
