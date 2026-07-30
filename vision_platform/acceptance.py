from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_platform.application import VisionLabApplication
from vision_platform.calibration.affine import AffineCalibration
from vision_platform.calibration.store import save_calibration
from vision_platform.errors import CalibrationError
from vision_platform.image_io import write_image
from vision_platform.models import TaskResult


@dataclass(frozen=True)
class CalibrationAcceptanceReport:
    status: str
    rms_error_mm: float
    max_error_mm: float
    fit_marker_count: int
    validation_marker_count: int
    calibration_path: Path
    raw_frame_path: Path
    annotated_frame_path: Path
    report_path: Path


@dataclass(frozen=True)
class ClassificationAcceptanceReport:
    task_result: TaskResult
    events_path: Path
    keyframe_paths: tuple[Path, ...]
    report_path: Path
    pending_hardware: tuple[dict[str, Any], ...]


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_centers(
    image_bgr: np.ndarray,
    *,
    kind: str,
) -> list[tuple[float, float]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if kind == "white":
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, 190], dtype=np.uint8),
            np.array([179, 45, 255], dtype=np.uint8),
        )
    elif kind == "magenta":
        mask = cv2.inRange(
            hsv,
            np.array([135, 100, 100], dtype=np.uint8),
            np.array([175, 255, 255], dtype=np.uint8),
        )
    else:
        raise ValueError(f"Unsupported marker kind: {kind}")
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    centers: list[tuple[float, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 60.0 <= area <= 800.0:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * np.pi * area / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )
        if circularity < 0.65:
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            continue
        centers.append(
            (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        )
    return sorted(centers, key=lambda point: (point[1], point[0]))


def _fit_scene_calibration(
    *,
    image_bgr: np.ndarray,
    scene_spec: dict[str, Any],
) -> tuple[
    AffineCalibration,
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    fit_pixels = _marker_centers(image_bgr, kind="white")
    validation_pixels = _marker_centers(image_bgr, kind="magenta")
    fit_markers = [
        marker
        for marker in scene_spec["calibration_markers"]
        if marker["role"] == "fit"
    ]
    validation_markers = [
        marker
        for marker in scene_spec["calibration_markers"]
        if marker["role"] == "validation"
    ]
    if len(fit_pixels) != len(fit_markers):
        raise CalibrationError(
            "Expected exactly three white fit markers in the simulation frame",
            expected=len(fit_markers),
            actual=len(fit_pixels),
            detected=[list(point) for point in fit_pixels],
        )
    if len(validation_pixels) != len(validation_markers):
        raise CalibrationError(
            "Expected the configured magenta validation markers",
            expected=len(validation_markers),
            actual=len(validation_pixels),
            detected=[list(point) for point in validation_pixels],
        )

    fit_world = [
        tuple(float(value) for value in marker["world_mm"][:2])
        for marker in fit_markers
    ]
    validation_world = [
        tuple(float(value) for value in marker["world_mm"][:2])
        for marker in validation_markers
    ]
    height, width = image_bgr.shape[:2]
    candidates: list[tuple[float, AffineCalibration]] = []
    for ordered_pixels in itertools.permutations(fit_pixels):
        calibration = AffineCalibration.fit(
            ordered_pixels,
            fit_world,
            image_size=(width, height),
            plane_z_mm=float(scene_spec["workspace"]["plane_z_mm"]),
            source="coppeliasim_markers",
            scene_version=str(scene_spec.get("schema_version", 1)),
        )
        metrics = calibration.evaluate(
            zip(validation_pixels, validation_world)
        )
        candidates.append((metrics.max_error_mm, calibration))
    # The four markers form a rectangle.  A validation corner alone cannot
    # distinguish all symmetric permutations: an axis-swapped transform can
    # also predict the fourth corner with a tiny residual.  The saved scene
    # has a fixed top-down camera contract: +world X maps toward -pixel X and
    # +world Y maps toward +pixel Y, with negligible cross-axis terms.
    axis_compatible = [
        item
        for item in candidates
        if item[1].matrix[0, 0] < 0
        and item[1].matrix[1, 1] > 0
        and abs(item[1].matrix[0, 0])
        > 2.0 * abs(item[1].matrix[0, 1])
        and abs(item[1].matrix[1, 1])
        > 2.0 * abs(item[1].matrix[1, 0])
    ]
    if not axis_compatible:
        raise CalibrationError(
            "No marker correspondence satisfies the fixed camera-axis contract"
        )
    _, selected = min(axis_compatible, key=lambda item: item[0])
    return selected, fit_pixels, validation_pixels


def _annotate_calibration(
    image_bgr: np.ndarray,
    *,
    calibration: AffineCalibration,
    fit_pixels: list[tuple[float, float]],
    validation_pixels: list[tuple[float, float]],
) -> np.ndarray:
    annotated = image_bgr.copy()
    for label, points, color in (
        ("FIT", fit_pixels, (255, 255, 255)),
        ("CHECK", validation_pixels, (255, 0, 255)),
    ):
        for index, point in enumerate(points, start=1):
            center = (int(round(point[0])), int(round(point[1])))
            world = calibration.pixel_to_world(point)
            cv2.circle(annotated, center, 9, color, 2)
            cv2.putText(
                annotated,
                f"{label}{index} ({world[0]:.1f},{world[1]:.1f})",
                (center[0] + 10, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated


def run_calibration_acceptance(
    scene: VisionLabApplication,
    *,
    output_dir: str | Path,
) -> CalibrationAcceptanceReport:
    application = scene
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    application.open()
    frame = application.camera.read(timeout_s=5.0)
    raw_path = output / "calibration_raw.png"
    annotated_path = output / "calibration_annotated.png"
    calibration_path = output / "calibration.json"
    report_path = output / "calibration_acceptance.json"
    if not write_image(raw_path, frame.image_bgr):
        raise RuntimeError(f"Could not write calibration frame: {raw_path}")

    calibration, fit_pixels, validation_pixels = _fit_scene_calibration(
        image_bgr=frame.image_bgr,
        scene_spec=application.scene_spec,
    )
    assert calibration.metrics is not None
    metrics = calibration.metrics
    max_rms = float(
        application.config.calibration.get("max_rms_error_mm", 3.0)
    )
    max_error = float(
        application.config.calibration.get("max_error_mm", 5.0)
    )
    status = (
        "PASS"
        if metrics.rms_error_mm <= max_rms
        and metrics.max_error_mm <= max_error
        else "FAIL"
    )
    save_calibration(calibration, calibration_path)
    application.set_calibration(calibration)
    annotated = _annotate_calibration(
        frame.image_bgr,
        calibration=calibration,
        fit_pixels=fit_pixels,
        validation_pixels=validation_pixels,
    )
    if not write_image(annotated_path, annotated):
        raise RuntimeError(
            f"Could not write annotated calibration frame: {annotated_path}"
        )
    payload = {
        "schema_version": 1,
        "status": status,
        "source": "coppeliasim",
        "image_size": [frame.width, frame.height],
        "fit_marker_count": len(fit_pixels),
        "validation_marker_count": len(validation_pixels),
        "rms_error_mm": metrics.rms_error_mm,
        "max_error_mm": metrics.max_error_mm,
        "limits_mm": {
            "rms": max_rms,
            "max": max_error,
        },
        "artifacts": {
            "raw_frame": raw_path.name,
            "annotated_frame": annotated_path.name,
            "calibration": calibration_path.name,
        },
    }
    _write_json(report_path, payload)
    if status != "PASS":
        raise CalibrationError(
            "Simulation calibration exceeded the teaching error budget",
            rms_error_mm=metrics.rms_error_mm,
            max_error_mm=metrics.max_error_mm,
            limits=payload["limits_mm"],
            report=str(report_path),
        )
    return CalibrationAcceptanceReport(
        status=status,
        rms_error_mm=metrics.rms_error_mm,
        max_error_mm=metrics.max_error_mm,
        fit_marker_count=len(fit_pixels),
        validation_marker_count=len(validation_pixels),
        calibration_path=calibration_path,
        raw_frame_path=raw_path,
        annotated_frame_path=annotated_path,
        report_path=report_path,
    )


def _pending_hardware_items() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "hikvision_live_camera",
            "status": "PENDING_HARDWARE",
            "reason": "当前机器未连接海康相机，需在实验室采集真实帧验收",
        },
        {
            "id": "physical_blx_robot",
            "status": "PENDING_HARDWARE",
            "reason": "真实机械臂动作、吸附和重复精度需在实验平台验收",
        },
        {
            "id": "sim_to_real_teaching_effect",
            "status": "PENDING_HUMAN_ACCEPTANCE",
            "reason": "由教师最终确认学生代码从仿真迁移到真机的教学效果",
        },
    )


def run_classification_acceptance(
    *,
    scene: VisionLabApplication,
    object_count: int,
    output_dir: str | Path,
) -> ClassificationAcceptanceReport:
    application = scene
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    application.open()
    if application.calibration is None:
        run_calibration_acceptance(
            application,
            output_dir=output / "calibration",
        )

    current_index = 0
    keyframes: list[Path] = []

    def capture_keyframe(event) -> None:
        nonlocal current_index
        if event.state == "ACQUIRE":
            current_index = int(event.data.get("object_index", current_index + 1))
            return
        phase = {
            "DETECT": "before_pick",
            "VERIFY": "after_release",
        }.get(event.state)
        if phase is None:
            return
        frame = application.camera.read(timeout_s=5.0)
        path = output / "keyframes" / f"{current_index:02d}_{phase}.png"
        if not write_image(path, frame.image_bgr):
            raise RuntimeError(f"Could not write keyframe: {path}")
        keyframes.append(path)

    unsubscribe = application.event_bus.subscribe(capture_keyframe)
    try:
        task = application.create_classification_task()
        result = task.run(
            max_objects=object_count,
            task_id="coppeliasim-six-object-classification",
        )
    finally:
        unsubscribe()

    events_path = output / "events.jsonl"
    events_path.write_text(
        "".join(
            json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
            for event in result.events
        ),
        encoding="utf-8",
    )
    pending = _pending_hardware_items()
    report_path = output / "classification_acceptance.json"
    payload = {
        "schema_version": 1,
        "status": result.status,
        "task": result.to_dict(),
        "keyframes": [
            path.relative_to(output).as_posix() for path in keyframes
        ],
        "events": events_path.name,
        "pending_hardware": list(pending),
    }
    _write_json(report_path, payload)
    return ClassificationAcceptanceReport(
        task_result=result,
        events_path=events_path,
        keyframe_paths=tuple(keyframes),
        report_path=report_path,
        pending_hardware=pending,
    )


def run_full_acceptance(
    application: VisionLabApplication,
    *,
    output_dir: str | Path,
    object_count: int = 6,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration = run_calibration_acceptance(
        application,
        output_dir=output,
    )
    classification = run_classification_acceptance(
        scene=application,
        object_count=object_count,
        output_dir=output,
    )

    config_snapshot = {
        "camera_backend": application.config.camera_backend,
        "robot_backend": application.config.robot_backend,
        "coppelia_host": application.config.coppelia_host,
        "coppelia_port": application.config.coppelia_port,
        "coppelia_scene": str(application.config.coppelia_scene),
        "workspace": {
            "x_mm": list(application.config.workspace.x_mm),
            "y_mm": list(application.config.workspace.y_mm),
            "z_mm": list(application.config.workspace.z_mm),
            "safe_z_mm": application.config.workspace.safe_z_mm,
        },
        "calibration": dict(application.config.calibration),
        "recognition": dict(application.config.recognition),
        "task": dict(application.config.task),
    }
    config_path = _write_json(output / "config_snapshot.json", config_snapshot)
    source_manifest_path = (
        application.config.project_root
        / "simulation"
        / "vision_lab"
        / "source_manifest.json"
    )
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    source_step = source_manifest["source_step"]
    source_archive = source_manifest["source_archive"]
    scene_path = application.config.coppelia_scene
    hashes = {
        "scene": {
            "path": str(scene_path),
            "sha256": _sha256(scene_path),
        },
        "vendor_step": {
            "archive": source_archive["default_path"],
            "archive_entry": source_step["archive_entry"],
            "sha256": source_step["sha256"],
        },
    }
    status = (
        "PASS"
        if calibration.status == "PASS"
        and classification.task_result.status == "PASS"
        else "FAIL"
    )
    acceptance_path = output / "acceptance.json"
    _write_json(
        acceptance_path,
        {
            "schema_version": 1,
            "status": status,
            "scope": "CoppeliaSim automated acceptance",
            "calibration": {
                "status": calibration.status,
                "rms_error_mm": calibration.rms_error_mm,
                "max_error_mm": calibration.max_error_mm,
                "report": calibration.report_path.name,
            },
            "classification": {
                "status": classification.task_result.status,
                "metrics": dict(classification.task_result.metrics),
                "report": classification.report_path.name,
            },
            "artifacts": {
                "events": classification.events_path.name,
                "keyframes": [
                    path.relative_to(output).as_posix()
                    for path in classification.keyframe_paths
                ],
                "config_snapshot": config_path.name,
            },
            "hashes": hashes,
            "pending_hardware": list(classification.pending_hardware),
        },
    )
    return acceptance_path
