from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.defect_service import DefectSortingService
from vision_platform.experiments.models import ExperimentAcceptance, ExperimentDefinition, ExperimentRunContext
from vision_platform.student.experiment_gateway import StudentExperimentGateway


ROOT = Path(__file__).resolve().parents[2]
FORMAL_ASSET_MANIFEST = (
    ROOT / "simulation" / "vision_defect_sorting_lab" / "defect_assets_manifest.json"
)
FIXED_ROIS = {
    "reference": (416, 56, 192, 192),
    "entry_a": (80, 320, 192, 192),
    "entry_b": (416, 320, 192, 192),
    "entry_c": (752, 320, 192, 192),
    "entry_d": (80, 648, 192, 192),
    "entry_e": (416, 648, 192, 192),
    "entry_f": (752, 648, 192, 192),
}


class Camera:
    def __init__(self, image_bgr: np.ndarray | None = None) -> None:
        self.read_calls = 0
        self.image_bgr = (
            np.zeros((1024, 1024, 3), dtype=np.uint8)
            if image_bgr is None
            else np.ascontiguousarray(image_bgr).copy()
        )

    def read(self, timeout_s: float):
        self.read_calls += 1
        return SimpleNamespace(
            image_bgr=self.image_bgr.copy(),
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
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return name


class ProfileController:
    def current(self):
        from vision_platform.vision_quality.models import AppliedVisionProfile
        return AppliedVisionProfile("standard", (1024, 1024), 20.0, 0.492, (0.8, 0.8, 0.8), (0.35, 0.35, 0.35))


def _formal_frame() -> np.ndarray:
    manifest = json.loads(FORMAL_ASSET_MANIFEST.read_text(encoding="utf-8"))
    assets: dict[str, np.ndarray] = {}
    for record in manifest["assets"]:
        image = cv2.imread(
            str(FORMAL_ASSET_MANIFEST.parent / record["path"]),
            cv2.IMREAD_COLOR,
        )
        assert image is not None
        assets[record["asset_id"]] = cv2.resize(
            image,
            (192, 192),
            interpolation=cv2.INTER_AREA,
        )

    frame = np.full((1024, 1024, 3), 245, dtype=np.uint8)
    for entry_id, (x, y, width, height) in FIXED_ROIS.items():
        asset_id = "reference" if entry_id == "reference" else entry_id
        frame[y : y + height, x : x + width] = assets[asset_id]
    return frame


def _gateway(tmp_path: Path, *, activate=None, image_bgr: np.ndarray | None = None):
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
    camera = Camera(image_bgr)
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


def test_v1_09_gateway_records_complete_two_dimensional_masks_as_bgr_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activated = []
    gateway, camera, app, evidence = _gateway(
        tmp_path,
        activate=activated.append,
        image_bgr=_formal_frame(),
    )
    service = DefectSortingService.from_manifest(FORMAL_ASSET_MANIFEST)
    captured_output = {}
    real_analyze = service.analyze

    def capture_analyze(*args, **kwargs):
        output = real_analyze(*args, **kwargs)
        captured_output["value"] = output
        return output

    monkeypatch.setattr(service, "analyze", capture_analyze)
    monkeypatch.setattr(
        "vision_platform.student.experiment_gateway.DefectSortingService.from_manifest",
        lambda *_args, **_kwargs: service,
    )

    response = gateway.dispatch("vision2d.surface_defects", {})

    assert [entry["decision"] for entry in response["entries"]] == [
        "qualified",
        "missing",
        "hole",
        "foreign",
        "broken",
        "dimension",
    ]
    assert camera.read_calls == 1
    assert app.robot.calls == []
    assert app.tool.calls == []
    assert len(activated) == 1
    assert len(activated[0].entries) == 6
    assert len(captured_output["value"].finding_masks) == 6
    assert all(
        mask.dtype == np.uint8 and mask.shape == (192, 192)
        for mask in captured_output["value"].finding_masks.values()
    )

    artifact_name, bundle = evidence.json_calls[-1]
    assert artifact_name.startswith("vision-bundle-")
    mask_layers = [
        layer for layer in bundle["layers"] if layer["layer_id"].startswith("mask-")
    ]
    assert [layer["layer_id"] for layer in mask_layers] == [
        f"mask-entry_{letter}" for letter in "abcdef"
    ]
    for layer in mask_layers:
        stored = cv2.imread(
            str(evidence.directory / layer["path"]),
            cv2.IMREAD_UNCHANGED,
        )
        assert stored is not None
        assert stored.shape == (192, 192, 3)


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
