from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.models import ExperimentAcceptance, ExperimentDefinition, ExperimentRunContext
from vision_platform.models import Frame
from vision_platform.student.experiment_gateway import StudentExperimentGateway
from vision_platform.vision2d.code_recognition import CodeReading, CodeRecognitionResult
from vision_platform.vision_quality.models import AppliedVisionProfile


class Camera:
    def __init__(self) -> None:
        self.read_calls = 0

    def read(self, timeout_s: float) -> Frame:
        assert timeout_s == 2.0
        self.read_calls += 1
        return Frame(
            image_bgr=np.full((1024, 1024, 3), 255, dtype=np.uint8),
            width=1024,
            height=1024,
            timestamp_s=1.0,
            source="coppeliasim",
            sequence_id=self.read_calls,
        )


class ProfileController:
    def current(self) -> AppliedVisionProfile:
        return AppliedVisionProfile(
            profile_id="standard", resolution=(1024, 1024),
            perspective_angle_deg=60.0, camera_rig_z_m=0.7,
            key_diffuse_rgb=(0.8, 0.8, 0.8), fill_diffuse_rgb=(0.35, 0.35, 0.35),
        )


class Evidence:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.directory = root
        self.root.mkdir(parents=True)
        self.json_calls: list[tuple[str, dict]] = []

    def record_snapshot(self, **payload: object) -> dict:
        path = self.root / "frames" / f"{payload['snapshot_id']}.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload["png_bytes"])
        return {
            "snapshot_id": payload["snapshot_id"],
            "path": path.relative_to(self.root).as_posix(),
            "sha256": hashlib.sha256(payload["png_bytes"]).hexdigest(),
            **payload["metadata"],
        }

    def record_json_artifact(self, name: str, payload: dict) -> str:
        self.json_calls.append((name, payload))
        (self.root / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return name


class Calls:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


def _reading(code_type: str, payload: str, center: tuple[float, float]) -> CodeReading:
    u, v = center
    return CodeReading(
        code_type, payload,
        ((u - 2, v - 2), (u + 2, v - 2), (u + 2, v + 2), (u - 2, v + 2)),
        (int(u - 2), int(v - 2), 4, 4), center, 0.98, True,
    )


def _recognition(case: str) -> CodeRecognitionResult:
    readings = [
        _reading("qr", "V1-07-A", (40.0, 45.0)),
        _reading("qr", "V1-07-B", (80.0, 45.0)),
        _reading("ean13", "6901234567892", (40.0, 95.0)),
        _reading("ean13", "6901234567809", (80.0, 95.0)),
    ]
    if case == "unknown":
        readings[0] = replace(readings[0], data="UNKNOWN")
    elif case == "duplicate":
        readings[1] = replace(readings[1], data=readings[0].data)
    elif case == "undecoded":
        readings[0] = replace(readings[0], data=None, decoded=False, failure_code="QR_DECODE_FAILED")
    elif case == "wrong_count":
        readings.pop()
    elif case == "low_confidence":
        readings[0] = replace(readings[0], confidence=0.2)
    elif case == "outside_workspace":
        readings[0] = _reading("qr", "V1-07-A", (300.0, 45.0))
    elif case == "invalid_bbox":
        readings[0] = replace(readings[0], bbox_px=(38, 43, 0, 4))
    return CodeRecognitionResult(
        "PASS", tuple(readings), (1024, 1024), None, ("qr", "ean13"), 4.0
    )


def _routing_config() -> dict[str, object]:
    return {
        "schema_version": 1, "expected_count": 4, "confidence_min": 0.90,
        "calibration_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, -90.0]],
        "pick_z_mm": 18.0, "safe_z_mm": 110.0, "speed_mm_s": 15.0,
        "routes": [
            {"entry_id": "entry_a", "part_id": "part_a", "code_type": "qr", "payload": "V1-07-A", "route_id": "route_red", "drop_xyz_mm": [116.0, -60.0, 22.0]},
            {"entry_id": "entry_b", "part_id": "part_b", "code_type": "qr", "payload": "V1-07-B", "route_id": "route_blue", "drop_xyz_mm": [116.0, 60.0, 22.0]},
            {"entry_id": "entry_c", "part_id": "part_c", "code_type": "ean13", "payload": "6901234567892", "route_id": "route_red", "drop_xyz_mm": [128.0, -60.0, 22.0]},
            {"entry_id": "entry_d", "part_id": "part_d", "code_type": "ean13", "payload": "6901234567809", "route_id": "route_blue", "drop_xyz_mm": [128.0, 60.0, 22.0]},
        ],
    }


def _gateway(tmp_path: Path, recognition_case: str = "pass"):
    scene = tmp_path / "BL23_vision_code_routing_lab.ttt"
    scene.write_bytes(b"v1-07-scene")
    asset_manifest = {
        "schema_version": 1,
        "entries": [
            {"part_id": route["part_id"], "code_type": route["code_type"], "payload": route["payload"]}
            for route in _routing_config()["routes"]
        ],
    }
    asset_path = tmp_path / "code_assets_manifest.json"
    asset_path.write_text(json.dumps(asset_manifest), encoding="utf-8")
    part_ids = ["part_a", "part_b", "part_c", "part_d"]
    if recognition_case == "scene_mismatch":
        part_ids.pop()
    initial_positions = {
        "part_a": [40.0, -45.0, 18.0],
        "part_b": [80.0, -45.0, 18.0],
        "part_c": [40.0, 5.0, 18.0],
        "part_d": [80.0, 5.0, 18.0],
    }
    if recognition_case == "calibration_mismatch":
        initial_positions["part_a"][0] += 10.0
    scene_manifest = {
        "schema_version": 1,
        "scene": {"path": str(scene), "sha256": hashlib.sha256(scene.read_bytes()).hexdigest()},
        "task_contracts": {},
        "code_assets_manifest": {"path": asset_path.name, "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest()},
        "code_routing": {
            "part_ids": part_ids,
            "initial_positions_mm": initial_positions,
            "calibration_plane_z_mm": 27.4,
            "calibration_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, -90.0]],
            "route_slots_mm": {
                "route_red": [[116.0, -60.0, 22.0], [128.0, -60.0, 22.0]],
                "route_blue": [[116.0, 60.0, 22.0], [128.0, 60.0, 22.0]],
            },
        },
    }
    scene_manifest_path = tmp_path / "scene_manifest.json"
    scene_manifest_path.write_text(json.dumps(scene_manifest), encoding="utf-8")
    public_parameters = {
        "camera_path": "/VisionCodeRoutingLab/CameraRig/Camera",
        "code_routing": _routing_config(),
    }
    workspace = {"x_mm": [20.0, 140.0], "y_mm": [-90.0, 90.0], "z_mm": [10.0, 140.0], "safe_z_mm": 110.0}
    definition = ExperimentDefinition(
        experiment_id="V1-07", pack_id="V1", title="Code routing", version="2.2.0",
        scene=scene, scene_manifest=scene_manifest_path,
        student_template=tmp_path / "student.py", guide=tmp_path / "guide.md",
        capabilities=("camera.rgb", "camera.profile", "lighting.profile", "vision2d.code_routing", "robot.home", "robot.pose", "robot.move_world", "tool.suction", "scene.probe"),
        workspace=workspace, public_parameters=public_parameters,
        acceptance=ExperimentAcceptance("code_route_occupancy", ("four_routes",), ("teacher_review",)),
        hardware_status="PENDING_HARDWARE",
    )
    context = ExperimentRunContext(
        "V1-07", "2.2.0", scene, scene_manifest["scene"]["sha256"],
        scene_manifest_path, public_parameters,
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
    )
    return gateway, camera, evidence, application, _recognition(recognition_case)


def test_gateway_builds_complete_plan_before_any_motion(tmp_path, monkeypatch) -> None:
    gateway, camera, evidence, application, recognition = _gateway(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        "vision_platform.student.experiment_gateway.recognize_codes",
        lambda image, **kwargs: calls.append("recognize") or recognition,
    )

    value = gateway.dispatch("vision2d.code_routes", {})

    assert value["schema_version"] == 1
    assert value["status"] == "PASS"
    assert len(value["entries"]) == 4
    assert camera.read_calls == 1
    assert calls == ["recognize"]
    assert application.robot.calls == []
    assert application.tool.calls == []
    assert gateway.route_guard.completion()["active"] is True
    bundle = evidence.json_calls[-1][1]
    assert [layer["layer_id"] for layer in bundle["layers"]] == ["raw", "annotated"]
    assert bundle["result"]["plan_id"] == value["plan_id"]


@pytest.mark.parametrize("argument", [{"roi": [0, 0, 10, 10]}, {"payload": "V1-07-A"}, {"path": "code.png"}])
def test_gateway_rejects_all_student_detector_arguments_before_capture(tmp_path, argument) -> None:
    gateway, camera, evidence, application, recognition = _gateway(tmp_path)
    with pytest.raises(ValueError, match="does not accept arguments"):
        gateway.dispatch("vision2d.code_routes", argument)
    assert camera.read_calls == 0
    assert application.robot.calls == []


@pytest.mark.parametrize(
    ("recognition_case", "expected_code"),
    [
        ("unknown", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("duplicate", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("undecoded", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("wrong_count", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("low_confidence", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("outside_workspace", "CODE_ROUTE_PLAN_INVALID"),
        ("scene_mismatch", "CODE_ROUTE_CONFIG_INVALID"),
        ("calibration_mismatch", "CODE_ROUTE_SCENE_POSITION_MISMATCH"),
        ("invalid_bbox", "CODE_ROUTE_RECOGNITION_INVALID"),
    ],
)
def test_gateway_plan_failures_never_move(tmp_path, monkeypatch, recognition_case, expected_code) -> None:
    gateway, camera, evidence, application, recognition = _gateway(tmp_path, recognition_case=recognition_case)
    monkeypatch.setattr(
        "vision_platform.student.experiment_gateway.recognize_codes",
        lambda image, **kwargs: recognition,
    )
    with pytest.raises(VisionPlatformError) as captured:
        gateway.dispatch("vision2d.code_routes", {})
    assert captured.value.code == expected_code
    assert application.robot.calls == []
    assert application.tool.calls == []
    assert gateway.route_guard.completion()["active"] is False
