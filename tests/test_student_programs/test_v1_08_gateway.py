from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.models import ExperimentAcceptance, ExperimentDefinition, ExperimentRunContext
from vision_platform.experiments.ocr_sorting import ApprovedOcrSortEntry, OcrSortPlan
from vision_platform.student.experiment_gateway import StudentExperimentGateway
from vision_platform.student.ocr_sort_guard import OcrSortGuard
from vision_platform.student.protocol import CommandMessage, RunState
from vision_platform.student.runner import StudentProgramController
from vision_platform.student.safety import StudentExecutionPolicy
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.vision2d.ocr import OCRCharacter, OCRResult, TrainingReport
from vision_platform.vision_quality.models import AppliedVisionProfile


class Camera:
    def __init__(self) -> None:
        self.read_calls = 0

    def read(self, timeout_s: float):
        self.read_calls += 1
        image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        return SimpleNamespace(
            image_bgr=image,
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
        return {
            "snapshot_id": payload["snapshot_id"],
            "path": path.relative_to(self.directory).as_posix(),
            "sha256": hashlib.sha256(payload["png_bytes"]).hexdigest(),
            **payload["metadata"],
        }

    def record_json_artifact(self, name: str, payload: dict) -> str:
        self.json_calls.append((name, payload))
        (self.directory / name).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return name


class ProfileController:
    def __init__(self) -> None:
        self.calls = 0

    def current(self) -> AppliedVisionProfile:
        self.calls += 1
        return AppliedVisionProfile(
            profile_id="standard", resolution=(1024, 1024),
            perspective_angle_deg=60.0, camera_rig_z_m=0.7,
            key_diffuse_rgb=(0.8, 0.8, 0.8), fill_diffuse_rgb=(0.35, 0.35, 0.35),
        )


def _plan() -> OcrSortPlan:
    entries = tuple(
        ApprovedOcrSortEntry(
            entry_id=entry_id,
            part_id=part_id,
            identifier=identifier,
            route_id=route_id,
            roi_px=(100 + index * 100, 200, 96, 128),
            pick_xyz_mm=pick,
            drop_xyz_mm=drop,
            confidence=0.98,
        )
        for index, (entry_id, part_id, identifier, route_id, pick, drop) in enumerate(
            (
                ("entry_a", "part_a", "A1", "route_alpha", (40.0, -45.0, 18.0), (116.0, -60.0, 22.0)),
                ("entry_b", "part_b", "A2", "route_alpha", (80.0, -45.0, 18.0), (128.0, -60.0, 22.0)),
                ("entry_c", "part_c", "B1", "route_beta", (40.0, 5.0, 18.0), (116.0, 60.0, 22.0)),
                ("entry_d", "part_d", "B2", "route_beta", (80.0, 5.0, 18.0), (128.0, 60.0, 22.0)),
            )
        )
    )
    return OcrSortPlan("a" * 64, entries, 110.0, 15.0)


def _service_result() -> SimpleNamespace:
    results = []
    for identifier in ("A1", "A2", "B1", "B2"):
        result = OCRResult(
            status="PASS", text=identifier,
            characters=(
                OCRCharacter(identifier[0], (5, 5, 20, 60), 0.98),
                OCRCharacter(identifier[1], (30, 5, 20, 60), 0.98),
            ), image_size=(96, 128), threshold_method="otsu",
            character_count=2, failure_code=None, processing_ms=1.0,
        )
        results.append(result)
    return SimpleNamespace(
        results=tuple(results),
        observations=(),
        plan=_plan(),
        training_report=TrainingReport(36, 12, 1.0, 20260802),
        annotated_frame=np.zeros((1024, 1024, 3), dtype=np.uint8),
        manifest_sha256="b" * 64,
        scene_id="V1-08",
    )


def _gateway(tmp_path: Path, *, activate=None):
    scene = tmp_path / "BL23_vision_ocr_sorting_lab.ttt"
    scene.write_bytes(b"scene-v1-08")
    assets = tmp_path / "ocr_assets_manifest.json"
    assets.write_text("{}", encoding="utf-8")
    scene_manifest = {
        "schema_version": 1,
        "scene": {
            "path": str(scene),
            "sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        },
        "task_contracts": {},
        "ocr_assets_manifest": {
            "path": assets.name,
            "sha256": hashlib.sha256(assets.read_bytes()).hexdigest(),
        },
        "ocr_sorting": {
            "part_ids": ["part_a", "part_b", "part_c", "part_d"],
            "fixed_rois_px": {
                "A1": [100, 200, 96, 128], "A2": [200, 200, 96, 128],
                "B1": [300, 200, 96, 128], "B2": [400, 200, 96, 128],
            },
        },
    }
    manifest_path = tmp_path / "scene_manifest.json"
    manifest_path.write_text(json.dumps(scene_manifest), encoding="utf-8")
    public_parameters = {
        "ocr_sorting": {
            "fixed_rois_px": scene_manifest["ocr_sorting"]["fixed_rois_px"],
        }
    }
    definition = ExperimentDefinition(
        experiment_id="V1-08", pack_id="V1", title="OCR sorting", version="2.2.0",
        scene=scene, scene_manifest=manifest_path,
        student_template=tmp_path / "student.py", guide=tmp_path / "guide.md",
        capabilities=("camera.rgb", "camera.profile", "lighting.profile", "vision2d.ocr_sorting"),
        workspace={"x_mm": [20.0, 140.0], "y_mm": [-90.0, 90.0], "z_mm": [10.0, 140.0], "safe_z_mm": 110.0},
        public_parameters=public_parameters,
        acceptance=ExperimentAcceptance("motion_observation", (), ("teacher_review",)),
        hardware_status="PENDING_HARDWARE",
    )
    context = ExperimentRunContext(
        "V1-08", "2.2.0", scene, scene_manifest["scene"]["sha256"],
        manifest_path, public_parameters,
    )
    camera = Camera()
    application = SimpleNamespace(
        config=SimpleNamespace(camera_backend="sim", robot_backend="sim"),
        sim=object(), camera=camera, robot=Calls(), tool=Calls(),
    )
    evidence = Evidence(tmp_path / "evidence")
    gateway = StudentExperimentGateway(
        application=application, evidence=evidence, context=context,
        definition=definition, scene_manifest=scene_manifest,
        profile_controller=ProfileController(),
        ocr_guard_activator=activate,
    )
    return gateway, camera, application, evidence


def test_gateway_builds_complete_ocr_plan_before_motion(tmp_path, monkeypatch) -> None:
    activated: list[OcrSortPlan] = []
    gateway, camera, application, evidence = _gateway(tmp_path, activate=activated.append)
    monkeypatch.setattr(
        "vision_platform.student.experiment_gateway.OcrSortingService.from_manifest",
        lambda *args, **kwargs: SimpleNamespace(
            analyze=lambda frame, rois: _service_result(),
        ),
    )

    value = gateway.dispatch("vision2d.ocr_sorting", {})

    assert value["status"] == "PASS"
    assert tuple(item["identifier"] for item in value["results"]) == (
        "A1", "A2", "B1", "B2"
    )
    assert value["results"][0]["characters"][0]["bbox_px"] == [5, 5, 20, 60]
    assert tuple(item["entry_id"] for item in value["entries"]) == (
        "entry_a", "entry_b", "entry_c", "entry_d"
    )
    assert camera.read_calls == 1
    assert activated and activated[0].plan_id == "a" * 64
    assert application.robot.calls == []
    assert application.tool.calls == []
    assert evidence.json_calls


def test_gateway_final_probe_forwards_same_run_context(tmp_path, monkeypatch) -> None:
    gateway, _, _, _ = _gateway(tmp_path, activate=lambda plan: None)
    captured: dict[str, object] = {}

    def fake_probe(sim, definition, **kwargs):
        captured.update(kwargs)
        return {
            "schema_version": 1,
            "experiment_id": "V1-08",
            "phase": "final",
            "status": "PASS",
            "matched": 4,
            "expected": 4,
            "hardware_status": "PENDING_HARDWARE",
        }

    monkeypatch.setattr(
        "vision_platform.student.experiment_gateway.probe_experiment",
        fake_probe,
    )
    context = {
        "run_id": "run-v1-08",
        "scene_hash": "a" * 64,
        "snapshot_id": "frame-000001",
        "entry_evidence": (),
        "consumed_entry_ids": (),
        "robot_home": True,
        "tool_on": False,
    }
    report = gateway.collect_probe("final", run_context=context)

    assert report["status"] == "PASS"
    # Gateway copies contexts into its bounded JSON-native representation;
    # tuples are intentionally normalized to lists before reaching probes.
    expected_context = {
        **context,
        "entry_evidence": [],
        "consumed_entry_ids": [],
    }
    assert captured["run_context"] == expected_context


def test_runner_captures_ocr_context_before_gateway_cleanup(tmp_path: Path) -> None:
    controller, _, _, evidence = _controller_for_private_ocr(tmp_path)
    controller._experiment_context = SimpleNamespace(scene_sha256="c" * 64)
    controller._experiment_gateway._ocr_evidence["snapshot_id"] = "frame-000001"
    controller._command_ocr_sort_entry({"entry_id": "entry_a"})

    context = controller._ocr_probe_context()

    assert context["run_id"] == evidence.run_id
    assert context["scene_hash"] == "c" * 64
    assert context["snapshot_id"] == "frame-000001"
    assert context["consumed_entry_ids"] == ("entry_a",)
    assert context["entry_evidence"][0]["entry_id"] == "entry_a"


@pytest.mark.parametrize("fail_home", [False, True])
def test_runner_home_fallback_requires_successful_cleanup(
    tmp_path: Path,
    fail_home: bool,
) -> None:
    controller, robot, _, _ = _controller_for_private_ocr(tmp_path)

    def move_home() -> None:
        robot.moves.append(("home",))
        if fail_home:
            raise RuntimeError("home-cleanup-failed")

    controller._application.robot.move_home = move_home
    controller._ocr_home_confirmed = False

    controller._cleanup()

    assert controller._ocr_robot_home_state() is (not fail_home)


def test_gateway_rejects_result_geometry_before_guard_activation(tmp_path, monkeypatch) -> None:
    activated: list[OcrSortPlan] = []
    gateway, camera, application, _ = _gateway(tmp_path, activate=activated.append)

    def malformed_result() -> SimpleNamespace:
        output = _service_result()
        result = output.results[0]
        bad_character = OCRCharacter("A", (90, 5, 20, 60), 0.98)
        output.results = (
            OCRResult(
                status=result.status,
                text=result.text,
                characters=(bad_character, result.characters[1]),
                image_size=result.image_size,
                threshold_method=result.threshold_method,
                character_count=result.character_count,
                failure_code=result.failure_code,
                processing_ms=result.processing_ms,
            ),
            *output.results[1:],
        )
        return output

    monkeypatch.setattr(
        "vision_platform.student.experiment_gateway.OcrSortingService.from_manifest",
        lambda *args, **kwargs: SimpleNamespace(
            analyze=lambda frame, rois: malformed_result(),
        ),
    )
    with pytest.raises(VisionPlatformError) as captured:
        gateway.dispatch("vision2d.ocr_sorting", {})
    assert captured.value.code == "OCR_SORT_RESULT_INVALID"
    assert activated == []
    assert camera.read_calls == 1
    assert application.robot.calls == []
    assert application.tool.calls == []


@pytest.mark.parametrize("args", [{"roi": [0, 0, 10, 10]}, {"path": "x.png"}, {"entry_id": "entry_a"}])
def test_gateway_rejects_ocr_arguments_before_capture(tmp_path, args) -> None:
    gateway, camera, application, _ = _gateway(tmp_path, activate=lambda plan: None)
    with pytest.raises(ValueError, match="does not accept arguments"):
        gateway.dispatch("vision2d.ocr_sorting", args)
    assert camera.read_calls == 0
    assert application.robot.calls == []


def test_gateway_requires_narrow_guard_activation_callback(tmp_path, monkeypatch) -> None:
    gateway, camera, application, _ = _gateway(tmp_path, activate=None)
    monkeypatch.setattr(
        "vision_platform.student.experiment_gateway.OcrSortingService.from_manifest",
        lambda *args, **kwargs: SimpleNamespace(analyze=lambda frame, rois: _service_result()),
    )
    with pytest.raises(VisionPlatformError) as captured:
        gateway.dispatch("vision2d.ocr_sorting", {})
    assert captured.value.code == "OCR_SORT_EXECUTOR_REQUIRED"
    assert camera.read_calls == 0
    assert application.robot.calls == []


def _controller_for_private_ocr(tmp_path: Path):
    class Robot:
        def __init__(self) -> None:
            self.pose = [100.0, 0.0, 110.0]
            self.moves: list[tuple] = []

        def current_world_pose(self):
            return tuple(self.pose)

        def move_world(self, x, y, z, *, speed):
            self.moves.append((x, y, z, speed))
            self.pose[:] = [x, y, z]

        def move_home(self):
            self.moves.append(("home",))

    class Tool:
        def __init__(self) -> None:
            self.events: list[str] = []

        def on(self):
            self.events.append("on")

        def off(self):
            self.events.append("off")

    class Evidence:
        run_id = "run-1"

        def __init__(self) -> None:
            self.events: list[tuple] = []

        def record_event(self, *args, **kwargs):
            self.events.append((args, kwargs))

    robot = Robot()
    tool = Tool()
    app = SimpleNamespace(
        config=SimpleNamespace(robot_backend="sim", camera_backend="sim"),
        workspace=WorkspacePolicy(x_mm=(20, 140), y_mm=(-90, 90), z_mm=(10, 140), safe_z_mm=110),
        robot=robot,
        tool=tool,
        camera=None,
        sim=None,
    )
    session = SimpleNamespace(application=app)
    policy = StudentExecutionPolicy(
        min_speed=1, max_speed=30, max_runtime_s=60, max_commands=200,
        command_timeout_s=10, max_sleep_s=5, tool_on_max_z_mm=35,
    )
    controller = StudentProgramController(
        session=session, execution_policy=policy, output_root=tmp_path / "runs"
    )
    plan = _plan()
    controller._experiment_definition = SimpleNamespace(
        capabilities=("vision2d.ocr_sorting",),
    )
    controller._ocr_guard.activate(plan)
    evidence = Evidence()

    class Gateway:
        _ocr_evidence = {"plan": plan}

        def reset_environment(self):
            return None

        def collect_ocr_entry_probe(self, entry_id: str, *, run_id: str):
            return {
                "run_id": run_id,
                "scene_hash": "c" * 64,
                "part_id": next(item.part_id for item in plan.entries if item.entry_id == entry_id),
                "route_id": next(item.route_id for item in plan.entries if item.entry_id == entry_id),
                "slot_id": "slot_1" if entry_id in {"entry_a", "entry_c"} else "slot_2",
                "evidence_id": f"ocr-entry-{entry_id}-001",
            }

    controller._experiment_gateway = Gateway()
    controller._evidence = evidence
    controller._state = RunState.RUNNING
    return controller, robot, tool, evidence


def test_private_ocr_runner_executes_only_guard_actions_and_probe(tmp_path: Path) -> None:
    controller, robot, tool, evidence = _controller_for_private_ocr(tmp_path)
    receipt = controller._command_ocr_sort_entry({"entry_id": "entry_a"})
    assert receipt["status"] == "COMPLETED"
    assert len(robot.moves) == 6
    assert tool.events == ["on", "off"]
    assert controller._ocr_guard.consumed_entry_ids == ("entry_a",)
    assert len(evidence.events) == 8


def test_unknown_or_duplicate_entry_fails_before_any_cleanup_device_call(tmp_path: Path) -> None:
    controller, robot, tool, _ = _controller_for_private_ocr(tmp_path)
    with pytest.raises(VisionPlatformError) as captured:
        controller._command_ocr_sort_entry({"entry_id": "unknown"})
    assert captured.value.code == "OCR_SORT_ENTRY_INVALID"
    assert robot.moves == []
    assert tool.events == []

    # A successfully completed entry is consumed; selecting it again is also
    # a pre-motion rejection and must not trigger tool-off/home cleanup.
    controller, robot, tool, _ = _controller_for_private_ocr(tmp_path / "second")
    controller._command_ocr_sort_entry({"entry_id": "entry_a"})
    robot.moves.clear()
    tool.events.clear()
    with pytest.raises(VisionPlatformError) as captured:
        controller._command_ocr_sort_entry({"entry_id": "entry_a"})
    assert captured.value.code == "OCR_SORT_ENTRY_INVALID"
    assert robot.moves == []
    assert tool.events == []


def test_stop_invalidates_ocr_guard_before_later_entry_can_move(tmp_path: Path) -> None:
    controller, robot, tool, _ = _controller_for_private_ocr(tmp_path)
    controller._request_stop("CANCELLED", {"code": "CANCELLED", "message": "stop"})

    assert controller._ocr_guard.state == "INVALIDATED"
    assert controller._ocr_guard.plan_id is None
    with pytest.raises(VisionPlatformError) as captured:
        controller._command_ocr_sort_entry({"entry_id": "entry_a"})
    assert captured.value.code in {"OCR_SORT_PLAN_INVALIDATED", "STUDENT_STOPPED"}
    assert robot.moves == []
    assert tool.events == []


def test_successful_terminalization_preserves_completed_ocr_guard(tmp_path: Path) -> None:
    controller, robot, tool, evidence = _controller_for_private_ocr(tmp_path)
    for entry_id in ("entry_a", "entry_b", "entry_c", "entry_d"):
        controller._command_ocr_sort_entry({"entry_id": entry_id})
    assert controller._ocr_guard.state == "COMPLETED"

    # Isolate terminal-state bookkeeping from the device/evidence adapters;
    # the guard state must remain available for final evidence after PASS.
    controller._experiment_gateway = None
    controller._evidence = None
    controller._reap_child = lambda *, force: True
    controller._close_ipc = lambda: None
    controller._emit_snapshot = lambda: None
    controller._complete_run("PASS", None)

    assert controller._ocr_guard.state == "COMPLETED"
    assert controller._ocr_guard.consumed_entry_ids == (
        "entry_a", "entry_b", "entry_c", "entry_d"
    )


@pytest.mark.parametrize("state", [RunState.EMPTY, RunState.RUNNING, RunState.PAUSED, RunState.PASSED, RunState.FAILED])
@pytest.mark.parametrize("name", ["robot.move_world", "robot.home", "robot.pose", "tool.on", "tool.off"])
def test_v1_08_dispatch_rejects_raw_device_commands_in_every_state(tmp_path: Path, state: RunState, name: str) -> None:
    controller, robot, tool, _ = _controller_for_private_ocr(tmp_path)
    controller._state = state
    args = {
        "robot.move_world": {"x_mm": 40, "y_mm": 0, "z_mm": 110, "speed": 15},
        "robot.home": {}, "robot.pose": {}, "tool.on": {}, "tool.off": {},
    }[name]
    with pytest.raises(VisionPlatformError) as captured:
        controller._dispatch(CommandMessage("000001", name, args))
    assert captured.value.code == "COMMAND_NOT_ALLOWED"
    assert robot.moves == []
    assert tool.events == []
