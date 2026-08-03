from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation.vision_lab.hashing import asset_sha256
from vision_platform.vision2d.code_recognition import encode_ean13_payload
from vision_platform.vision_quality import load_profile_catalog
from vision_platform.vision_quality.catalog import load_profile_catalog_bytes
from vision_platform.vision_quality.models import VisionProfile, VisionProfileCatalog
from vision_platform.coppelia_scene import stage_scene_for_coppeliasim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = (
    PROJECT_ROOT / "simulation" / "vision_lab" / "source_manifest.json"
)
TEMPLATE_RELATIVE = "simulation/vision_lab/BL23_vision_lab.ttt"
PROTECTED_ROOTS = (
    TEMPLATE_RELATIVE,
    "robot_backends/models/BLX_openr6.ttt",
    "robot_backends/models/openr6_arm_coppeliasim.urdf",
    "robot_backends/models/meshes_blx",
    "simulation/vision_lab/assets/robot",
)
DECLARED_PROTECTED_ROOTS = PROTECTED_ROOTS[1:4]
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}

_BASICS_REQUIRED_PATHS = (
    "/BLX_base_link",
    "/BLX_joint1",
    "/BLX_joint2",
    "/BLX_joint3",
    "/BLX_joint4",
    "/BLX_joint5",
    "/BLX_joint6",
    "/BLX_tool_suction",
    "/RobotBasics",
    "/RobotBasics/Workspace",
    "/RobotBasics/Camera",
    "/RobotBasics/TeachPoints/TeachPoint1",
    "/RobotBasics/TeachPoints/TeachPoint2",
    "/RobotBasics/TeachPoints/TeachPoint3",
    "/RobotBasics/TeachPoints/TeachPoint4",
)
_LOGISTICS_REQUIRED_PATHS = (
    "/BLX_base_link",
    "/BLX_joint1",
    "/BLX_joint2",
    "/BLX_joint3",
    "/BLX_joint4",
    "/BLX_joint5",
    "/BLX_joint6",
    "/BLX_tool_suction",
    "/LogisticsLab",
    "/LogisticsLab/Workspace",
    "/LogisticsLab/Camera",
    "/LogisticsLab/Tasks/Stack",
    "/LogisticsLab/Tasks/Stack/Pickables",
    "/LogisticsLab/Tasks/Stack/Targets",
    "/LogisticsLab/Tasks/Digits",
    "/LogisticsLab/Tasks/Digits/Pickables",
    "/LogisticsLab/Tasks/Digits/Targets",
    "/LogisticsLab/Tasks/Classes",
    "/LogisticsLab/Tasks/Classes/Pickables",
    "/LogisticsLab/Tasks/Classes/Targets",
)
_VISION_QUALITY_REQUIRED_PATHS = (
    "/VisionQualityLab", "/VisionQualityLab/InspectionBoard",
    "/VisionQualityLab/Samples", "/VisionQualityLab/Samples/ReferenceRectangle",
    "/VisionQualityLab/Samples/ReferenceCircle", "/VisionQualityLab/Samples/ReferenceTriangle",
    "/VisionQualityLab/Samples/ResolutionTarget", "/VisionQualityLab/CameraRig",
    "/VisionQualityLab/CameraRig/Camera", "/VisionQualityLab/Lighting",
    "/VisionQualityLab/Lighting/KeyLight", "/VisionQualityLab/Lighting/FillLight",
)
_CODE_ROUTING_REQUIRED_PATHS = (
    "/BLX_base_link", "/BLX_tool_suction", "/VisionCodeRoutingLab",
    "/VisionCodeRoutingLab/Workspace", "/VisionCodeRoutingLab/CameraRig/Camera",
    "/VisionCodeRoutingLab/Lighting", "/VisionCodeRoutingLab/Lighting/KeyLight",
    "/VisionCodeRoutingLab/Lighting/FillLight",
    "/VisionCodeRoutingLab/Parts", "/VisionCodeRoutingLab/Parts/part_a",
    "/VisionCodeRoutingLab/Parts/part_a/CodeFace", "/VisionCodeRoutingLab/Parts/part_b",
    "/VisionCodeRoutingLab/Parts/part_b/CodeFace", "/VisionCodeRoutingLab/Parts/part_c",
    "/VisionCodeRoutingLab/Parts/part_c/CodeFace", "/VisionCodeRoutingLab/Parts/part_d",
    "/VisionCodeRoutingLab/Parts/part_d/CodeFace", "/VisionCodeRoutingLab/Bins/route_red",
    "/VisionCodeRoutingLab/Bins/route_red/red_1", "/VisionCodeRoutingLab/Bins/route_red/red_2",
    "/VisionCodeRoutingLab/Bins/route_blue", "/VisionCodeRoutingLab/Bins/route_blue/blue_1",
    "/VisionCodeRoutingLab/Bins/route_blue/blue_2",
)
_OCR_SORTING_REQUIRED_PATHS = (
    "/BLX_base_link",
    "/BLX_joint1",
    "/BLX_joint2",
    "/BLX_joint3",
    "/BLX_joint4",
    "/BLX_joint5",
    "/BLX_joint6",
    "/BLX_tool_suction",
    "/VisionOcrSortingLab",
    "/VisionOcrSortingLab/Workspace",
    "/VisionOcrSortingLab/CameraRig",
    "/VisionOcrSortingLab/CameraRig/Camera",
    "/VisionOcrSortingLab/Lighting",
    "/VisionOcrSortingLab/Lighting/KeyLight",
    "/VisionOcrSortingLab/Lighting/FillLight",
    "/VisionOcrSortingLab/Parts",
    "/VisionOcrSortingLab/Parts/part_a",
    "/VisionOcrSortingLab/Parts/part_a/CodeFace",
    "/VisionOcrSortingLab/Parts/part_b",
    "/VisionOcrSortingLab/Parts/part_b/CodeFace",
    "/VisionOcrSortingLab/Parts/part_c",
    "/VisionOcrSortingLab/Parts/part_c/CodeFace",
    "/VisionOcrSortingLab/Parts/part_d",
    "/VisionOcrSortingLab/Parts/part_d/CodeFace",
    "/VisionOcrSortingLab/Routes",
    "/VisionOcrSortingLab/Routes/route_alpha",
    "/VisionOcrSortingLab/Routes/route_alpha/slot_1",
    "/VisionOcrSortingLab/Routes/route_alpha/slot_2",
    "/VisionOcrSortingLab/Routes/route_beta",
    "/VisionOcrSortingLab/Routes/route_beta/slot_1",
    "/VisionOcrSortingLab/Routes/route_beta/slot_2",
)
_DEFECT_SORTING_REQUIRED_PATHS = (
    "/BLX_base_link",
    "/BLX_joint1",
    "/BLX_joint2",
    "/BLX_joint3",
    "/BLX_joint4",
    "/BLX_joint5",
    "/BLX_joint6",
    "/BLX_tool_suction",
    "/VisionDefectSortingLab",
    "/VisionDefectSortingLab/Workspace",
    "/VisionDefectSortingLab/CameraRig",
    "/VisionDefectSortingLab/CameraRig/Camera",
    "/VisionDefectSortingLab/Lighting",
    "/VisionDefectSortingLab/Lighting/KeyLight",
    "/VisionDefectSortingLab/Lighting/FillLight",
    "/VisionDefectSortingLab/Reference",
    "/VisionDefectSortingLab/Reference/InspectionFace",
    "/VisionDefectSortingLab/Parts",
    "/VisionDefectSortingLab/Parts/part_a",
    "/VisionDefectSortingLab/Parts/part_a/InspectionFace",
    "/VisionDefectSortingLab/Parts/part_b",
    "/VisionDefectSortingLab/Parts/part_b/InspectionFace",
    "/VisionDefectSortingLab/Parts/part_c",
    "/VisionDefectSortingLab/Parts/part_c/InspectionFace",
    "/VisionDefectSortingLab/Parts/part_d",
    "/VisionDefectSortingLab/Parts/part_d/InspectionFace",
    "/VisionDefectSortingLab/Parts/part_e",
    "/VisionDefectSortingLab/Parts/part_e/InspectionFace",
    "/VisionDefectSortingLab/Parts/part_f",
    "/VisionDefectSortingLab/Parts/part_f/InspectionFace",
    "/VisionDefectSortingLab/Slots",
    "/VisionDefectSortingLab/Slots/slot_qualified",
    "/VisionDefectSortingLab/Slots/slot_missing",
    "/VisionDefectSortingLab/Slots/slot_hole",
    "/VisionDefectSortingLab/Slots/slot_foreign",
    "/VisionDefectSortingLab/Slots/slot_broken",
    "/VisionDefectSortingLab/Slots/slot_dimension",
)


@dataclass(frozen=True)
class _FormalScene:
    spec_relative: str
    scene_id: str
    root_path: str
    output_relative: str
    required_paths: tuple[str, ...]


@dataclass(frozen=True)
class _RecoverableRelease:
    scene_sha256: str
    scene_size: int


class _BuildOutputLock:
    """Fail-fast thread/process lock for one normalized formal output."""

    def __init__(self, output: Path):
        normalized = Path(os.path.abspath(output)).resolve(strict=False)
        self._key = os.path.normcase(str(normalized))
        digest = hashlib.sha256(self._key.encode("utf-8")).hexdigest()
        self._path = (
            Path(tempfile.gettempdir())
            / "robot-sim-scene-build-locks"
            / f"{digest}.lock"
        )
        self._thread_lock: threading.Lock | None = None
        self._stream = None
        self._os_lock_acquired = False

    def acquire(self) -> None:
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(
                self._key,
                threading.Lock(),
            )
        if not thread_lock.acquire(blocking=False):
            raise RuntimeError("build already in progress")
        self._thread_lock = thread_lock
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._path.open("a+b")
            self._stream.seek(0, os.SEEK_END)
            if self._stream.tell() == 0:
                self._stream.write(b"\0")
                self._stream.flush()
                os.fsync(self._stream.fileno())
            self._stream.seek(0)
            try:
                self._acquire_os_lock()
            except OSError as exc:
                raise RuntimeError("build already in progress") from exc
            self._os_lock_acquired = True
        except BaseException as acquire_error:
            acquire_cleanup_errors = []
            if self._stream is not None:
                try:
                    self._stream.close()
                except BaseException as exc:
                    acquire_cleanup_errors.append(
                        _cleanup_diagnostic(
                            "output lock acquire close failed",
                            exc,
                        )
                    )
                self._stream = None
            try:
                thread_lock.release()
            except BaseException as exc:
                acquire_cleanup_errors.append(
                    _cleanup_diagnostic(
                        "thread lock acquire rollback failed",
                        exc,
                    )
                )
            self._thread_lock = None
            _note_cleanup_errors(acquire_error, acquire_cleanup_errors)
            raise

    def _acquire_os_lock(self) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(
                self._stream.fileno(),
                msvcrt.LK_NBLCK,
                1,
            )
            return
        import fcntl

        fcntl.flock(
            self._stream.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )

    def _release_os_lock(self) -> None:
        if os.name == "nt":
            import msvcrt

            self._stream.seek(0)
            msvcrt.locking(
                self._stream.fileno(),
                msvcrt.LK_UNLCK,
                1,
            )
            return
        import fcntl

        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)

    def release(self) -> list[str]:
        errors: list[str] = []
        if self._stream is not None:
            if self._os_lock_acquired:
                try:
                    self._release_os_lock()
                except BaseException as exc:
                    errors.append(
                        "output lock release failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                self._os_lock_acquired = False
            try:
                self._stream.close()
            except BaseException as exc:
                errors.append(
                    "output lock close failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            self._stream = None
        if self._thread_lock is not None:
            try:
                self._thread_lock.release()
            except BaseException as exc:
                errors.append(
                    "thread build lock release failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            self._thread_lock = None
        return errors


_FORMAL_SCENES = {
    "simulation/robot_basics/scene_spec.json": _FormalScene(
        spec_relative="simulation/robot_basics/scene_spec.json",
        scene_id="robot-basics",
        root_path="/RobotBasics",
        output_relative="simulation/robot_basics/BL23_robot_basics.ttt",
        required_paths=_BASICS_REQUIRED_PATHS,
    ),
    "simulation/logistics_lab/scene_spec.json": _FormalScene(
        spec_relative="simulation/logistics_lab/scene_spec.json",
        scene_id="logistics-lab",
        root_path="/LogisticsLab",
        output_relative="simulation/logistics_lab/BL23_logistics_lab.ttt",
        required_paths=_LOGISTICS_REQUIRED_PATHS,
    ),
    "simulation/vision_quality_lab/scene_spec.json": _FormalScene(
        spec_relative="simulation/vision_quality_lab/scene_spec.json",
        scene_id="vision-quality-lab",
        root_path="/VisionQualityLab",
        output_relative="simulation/vision_quality_lab/BL23_vision_quality_lab.ttt",
        required_paths=_VISION_QUALITY_REQUIRED_PATHS,
    ),
    "simulation/vision_code_routing_lab/scene_spec.json": _FormalScene(
        spec_relative="simulation/vision_code_routing_lab/scene_spec.json",
        scene_id="vision-code-routing-lab",
        root_path="/VisionCodeRoutingLab",
        output_relative="simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt",
        required_paths=_CODE_ROUTING_REQUIRED_PATHS,
    ),
    "simulation/vision_ocr_sorting_lab/scene_spec.json": _FormalScene(
        spec_relative="simulation/vision_ocr_sorting_lab/scene_spec.json",
        scene_id="vision-ocr-sorting-lab",
        root_path="/VisionOcrSortingLab",
        output_relative="simulation/vision_ocr_sorting_lab/BL23_vision_ocr_sorting_lab.ttt",
        required_paths=_OCR_SORTING_REQUIRED_PATHS,
    ),
    "simulation/vision_defect_sorting_lab/scene_spec.json": _FormalScene(
        spec_relative="simulation/vision_defect_sorting_lab/scene_spec.json",
        scene_id="vision-defect-sorting-lab",
        root_path="/VisionDefectSortingLab",
        output_relative="simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt",
        required_paths=_DEFECT_SORTING_REQUIRED_PATHS,
    ),
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_ocr_profile_catalog(path: Path) -> VisionProfileCatalog:
    """Load the V1-08 profile without broadening the shared catalog allowlist."""
    def _safe_float(value: Any) -> float | None:
        if type(value) not in (int, float):
            return None
        try:
            converted = float(value)
        except (OverflowError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    payload = _load(path)
    expected_fields = {
        "schema_version",
        "baseline_profile_id",
        "sensor_path",
        "camera_rig_path",
        "key_light_path",
        "fill_light_path",
        "near_clip_m",
        "far_clip_m",
        "profiles",
    }
    if (
        set(payload) != expected_fields
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or type(payload["baseline_profile_id"]) is not str
        or payload["baseline_profile_id"] != "standard"
    ):
        raise ValueError("OCR profile catalog schema is invalid")
    if payload["sensor_path"] != "/VisionOcrSortingLab/CameraRig/Camera":
        raise ValueError("OCR profile sensor path is invalid")
    if payload["camera_rig_path"] != "/VisionOcrSortingLab/CameraRig":
        raise ValueError("OCR profile camera rig path is invalid")
    if payload["key_light_path"] != "/VisionOcrSortingLab/Lighting/KeyLight":
        raise ValueError("OCR profile key-light path is invalid")
    if payload["fill_light_path"] != "/VisionOcrSortingLab/Lighting/FillLight":
        raise ValueError("OCR profile fill-light path is invalid")
    near = payload["near_clip_m"]
    far = payload["far_clip_m"]
    near_value = _safe_float(near)
    far_value = _safe_float(far)
    if (
        near_value is None
        or far_value is None
        or near_value != 0.05
        or far_value != 1.0
    ):
        raise ValueError("OCR profile clipping values are invalid")
    profiles = payload["profiles"]
    if not isinstance(profiles, list) or len(profiles) != 1:
        raise ValueError("OCR profile catalog must contain one profile")
    item = profiles[0]
    required_profile = {
        "profile_id",
        "label",
        "resolution",
        "perspective_angle_deg",
        "camera_rig_z_m",
        "key_diffuse_rgb",
        "fill_diffuse_rgb",
    }
    if not isinstance(item, dict) or set(item) != required_profile:
        raise ValueError("OCR profile fields are invalid")
    if (
        type(item["profile_id"]) is not str
        or item["profile_id"] != "standard"
        or type(item["label"]) is not str
        or item["label"] != "OCR 分拣固定视图"
    ):
        raise ValueError("OCR profile identity is invalid")
    resolution = item["resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(value) is not int for value in resolution)
        or resolution != [1024, 1024]
    ):
        raise ValueError("OCR profile resolution must be 1024x1024")
    angle = item["perspective_angle_deg"]
    rig_z = item["camera_rig_z_m"]
    angle_value = _safe_float(angle)
    rig_z_value = _safe_float(rig_z)
    if (
        angle_value is None
        or rig_z_value is None
        or angle_value != 20.0
        or rig_z_value != 0.5
    ):
        raise ValueError("OCR profile camera parameters are invalid")
    for name, value, expected_rgb in (
        ("key_diffuse_rgb", item["key_diffuse_rgb"], [0.8, 0.8, 0.8]),
        ("fill_diffuse_rgb", item["fill_diffuse_rgb"], [0.35, 0.35, 0.35]),
    ):
        converted_rgb = (
            [_safe_float(component) for component in value]
            if isinstance(value, list)
            else None
        )
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(type(component) not in (int, float) for component in value)
            or converted_rgb is None
            or any(component is None for component in converted_rgb)
            or any(component < 0.0 or component > 1.0 for component in converted_rgb if component is not None)
            or converted_rgb != expected_rgb
        ):
            raise ValueError(f"OCR profile {name} is invalid")
    profile = VisionProfile(
        profile_id=item["profile_id"],
        label=item["label"],
        resolution=tuple(resolution),
        perspective_angle_deg=item["perspective_angle_deg"],
        camera_rig_z_m=item["camera_rig_z_m"],
        key_diffuse_rgb=tuple(float(value) for value in item["key_diffuse_rgb"]),
        fill_diffuse_rgb=tuple(float(value) for value in item["fill_diffuse_rgb"]),
    )
    return VisionProfileCatalog(
        baseline_profile_id=payload["baseline_profile_id"],
        sensor_path=payload["sensor_path"],
        camera_rig_path=payload["camera_rig_path"],
        key_light_path=payload["key_light_path"],
        fill_light_path=payload["fill_light_path"],
        near_clip_m=near,
        far_clip_m=far,
        profiles=(profile,),
    )


def _load_defect_profile_catalog(path: Path) -> VisionProfileCatalog:
    """Load the V1-09 profile without changing the shared catalog allowlist."""

    payload = _load(path)
    fields = {
        "schema_version",
        "baseline_profile_id",
        "sensor_path",
        "camera_rig_path",
        "key_light_path",
        "fill_light_path",
        "near_clip_m",
        "far_clip_m",
        "profiles",
    }
    if set(payload) != fields or payload["schema_version"] != 1 or payload["baseline_profile_id"] != "standard":
        raise ValueError("V1-09 profile catalog schema is invalid")
    expected_paths = {
        "sensor_path": "/VisionDefectSortingLab/CameraRig/Camera",
        "camera_rig_path": "/VisionDefectSortingLab/CameraRig",
        "key_light_path": "/VisionDefectSortingLab/Lighting/KeyLight",
        "fill_light_path": "/VisionDefectSortingLab/Lighting/FillLight",
    }
    if any(payload[name] != value for name, value in expected_paths.items()):
        raise ValueError("V1-09 profile paths are invalid")
    if payload["near_clip_m"] != 0.05 or payload["far_clip_m"] != 1.0:
        raise ValueError("V1-09 profile clipping values are invalid")
    profiles = payload["profiles"]
    required_profile = {
        "profile_id",
        "label",
        "resolution",
        "perspective_angle_deg",
        "camera_rig_z_m",
        "key_diffuse_rgb",
        "fill_diffuse_rgb",
    }
    if not isinstance(profiles, list) or len(profiles) != 1 or set(profiles[0]) != required_profile:
        raise ValueError("V1-09 profile fields are invalid")
    item = profiles[0]
    if item["profile_id"] != "standard" or item["label"] != "表面缺陷固定视图" or item["resolution"] != [1024, 1024]:
        raise ValueError("V1-09 profile identity/resolution is invalid")
    if item["perspective_angle_deg"] != 20 or item["camera_rig_z_m"] != 0.5:
        raise ValueError("V1-09 profile camera parameters are invalid")
    if item["key_diffuse_rgb"] != [0.8, 0.8, 0.8] or item["fill_diffuse_rgb"] != [0.35, 0.35, 0.35]:
        raise ValueError("V1-09 profile lighting values are invalid")
    return VisionProfileCatalog(
        baseline_profile_id="standard",
        sensor_path=payload["sensor_path"],
        camera_rig_path=payload["camera_rig_path"],
        key_light_path=payload["key_light_path"],
        fill_light_path=payload["fill_light_path"],
        near_clip_m=0.05,
        far_clip_m=1.0,
        profiles=(
            VisionProfile(
                profile_id="standard",
                label=item["label"],
                resolution=(1024, 1024),
                perspective_angle_deg=20,
                camera_rig_z_m=0.5,
                key_diffuse_rgb=(0.8, 0.8, 0.8),
                fill_diffuse_rgb=(0.35, 0.35, 0.35),
            ),
        ),
    )


def _validate_ocr_assets_manifest(path: Path) -> dict[str, Any]:
    """Validate every scene label before opening a CoppeliaSim connection."""
    payload = _load(path)
    labels = payload.get("labels")
    if not isinstance(labels, list) or len(labels) != 4:
        raise ValueError("OCR assets manifest must contain four labels")
    expected_ids = ["A1", "A2", "B1", "B2"]
    seen_paths: set[str] = set()
    validated: list[dict[str, Any]] = []
    required_fields = {
        "identifier",
        "path",
        "purpose",
        "size_px",
        "channels",
        "sha256",
    }
    for item, expected_id in zip(labels, expected_ids):
        if not isinstance(item, dict) or set(item) != required_fields:
            raise ValueError("OCR scene label fields are invalid")
        if item["identifier"] != expected_id or item["purpose"] != "scene_label":
            raise ValueError("OCR scene labels must be A1, A2, B1, and B2")
        raw_path = item["path"]
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or raw_path != raw_path.strip()
            or Path(raw_path).is_absolute()
            or bool(Path(raw_path).anchor)
            or "\\" in raw_path
            or any(part in {".", ".."} for part in Path(raw_path).parts)
            or not raw_path.startswith("labels/")
            or raw_path in seen_paths
        ):
            raise ValueError("OCR label path must be a unique labels-relative path")
        seen_paths.add(raw_path)
        try:
            manifest_root = path.parent.resolve(strict=True)
            asset_path = manifest_root / Path(raw_path)
            resolved_asset = asset_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError("OCR label path must be a regular file") from exc
        if resolved_asset != manifest_root and manifest_root not in resolved_asset.parents:
            raise ValueError("OCR label path must stay under the manifest directory")
        current = manifest_root
        for component in Path(raw_path).parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("OCR label path must not use a symlink")
            try:
                attributes = getattr(current.lstat(), "st_file_attributes", 0)
            except OSError as exc:
                raise ValueError("OCR label path identity could not be inspected") from exc
            if attributes & _REPARSE_POINT:
                raise ValueError("OCR label path must not use a reparse point")
        if not asset_path.is_file():
            raise ValueError("OCR label path must be a regular file")
        try:
            attributes = getattr(asset_path.lstat(), "st_file_attributes", 0)
        except OSError as exc:
            raise ValueError("OCR label path identity could not be inspected") from exc
        if attributes & _REPARSE_POINT or asset_path.stat().st_nlink != 1:
            raise ValueError("OCR label path must not be an alias")
        if (
            not isinstance(item["size_px"], list)
            or len(item["size_px"]) != 2
            or any(type(value) is not int for value in item["size_px"])
            or item["size_px"] != [64, 96]
            or type(item["channels"]) is not int
            or item["channels"] != 3
        ):
            raise ValueError("OCR scene labels must be 64x96 three-channel images")
        digest = item["sha256"]
        if not isinstance(digest, str) or _LOWER_SHA256.fullmatch(digest) is None:
            raise ValueError("OCR scene label sha256 is invalid")
        if _sha256(asset_path) != digest:
            raise ValueError("OCR scene label sha256 does not match bytes")
        image = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (96, 64, 3) or image.dtype != np.uint8:
            raise ValueError("OCR scene label image cannot be decoded")
        validated.append(dict(item))
    return {"labels": validated}


def _validate_defect_assets_manifest(path: Path) -> dict[str, Any]:
    """Validate the seven deterministic V1-09 surface textures."""

    payload = _load(path)
    if set(payload) != {
        "assets",
        "generator",
        "generator_version",
        "schema_version",
        "scene_id",
        "seed",
        "source",
    }:
        raise ValueError("V1-09 defect assets manifest fields are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("V1-09 defect assets manifest schema_version is invalid")
    if payload["scene_id"] != "V1-09" or payload["seed"] != 20260803 or payload["source"] != "project-original-generated":
        raise ValueError("V1-09 defect asset provenance is invalid")
    assets = payload["assets"]
    if not isinstance(assets, list) or len(assets) != 7:
        raise ValueError("V1-09 defect assets manifest must contain seven PNGs")
    expected_ids = ("reference", "entry_a", "entry_b", "entry_c", "entry_d", "entry_e", "entry_f")
    validated: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item, expected_id in zip(assets, expected_ids):
        required = {"asset_id", "generator", "generator_version", "path", "purpose", "sha256", "size_px", "source"}
        if not isinstance(item, dict) or set(item) != required or item["asset_id"] != expected_id:
            raise ValueError("V1-09 defect asset fields are invalid")
        raw_path = item["path"]
        if (
            not isinstance(raw_path, str)
            or not raw_path.startswith("assets/")
            or Path(raw_path).is_absolute()
            or bool(Path(raw_path).anchor)
            or "\\" in raw_path
            or any(part in {".", ".."} for part in Path(raw_path).parts)
            or raw_path in seen_paths
            or not raw_path.lower().endswith(".png")
        ):
            raise ValueError("V1-09 defect asset path is invalid")
        seen_paths.add(raw_path)
        asset_path = path.parent / Path(raw_path)
        if not asset_path.is_file() or _is_reparse_or_symlink(asset_path):
            raise ValueError("V1-09 defect asset path must be a regular file")
        if item["size_px"] != [256, 256] or item["source"] != "project-original-generated":
            raise ValueError("V1-09 defect asset dimensions/provenance are invalid")
        digest = item["sha256"]
        if not isinstance(digest, str) or _LOWER_SHA256.fullmatch(digest) is None or _sha256(asset_path) != digest:
            raise ValueError("V1-09 defect asset sha256 does not match bytes")
        image = cv2.imread(str(asset_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (256, 256) or image.dtype != np.uint8:
            raise ValueError("V1-09 defect asset image must be 256x256 uint8 grayscale")
        validated.append(dict(item))
    return {
        "assets": validated,
        "generator": payload["generator"],
        "generator_version": payload["generator_version"],
        "schema_version": payload["schema_version"],
        "scene_id": payload["scene_id"],
        "seed": payload["seed"],
        "source": payload["source"],
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as exc:
        raise ValueError(f"could not inspect path identity: {path}") from exc
    return bool(attributes & _REPARSE_POINT)


def _lexical_project_path(path: str | Path, *, label: str) -> Path:
    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside PROJECT_ROOT") from exc
    return candidate


def _assert_no_alias(
    path: Path,
    *,
    label: str,
    must_exist: bool,
) -> None:
    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside PROJECT_ROOT") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if not os.path.lexists(current):
            if current == path and not must_exist:
                return
            raise ValueError(f"{label} does not exist: {current}")
        if _is_reparse_or_symlink(current):
            raise ValueError(
                f"{label} must not use a symlink, junction, or alias: {current}"
            )
    if path.is_file() and path.stat().st_nlink != 1:
        raise ValueError(f"{label} must not use a hardlink alias: {path}")


def _project_file_from_relative(
    raw: Any,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or Path(raw).is_absolute()
        or bool(Path(raw).anchor)
        or "\\" in raw
        or any(part in {".", ".."} for part in Path(raw).parts)
    ):
        raise ValueError(f"{label} must be a canonical project-relative path")
    path = _lexical_project_path(raw, label=label)
    _assert_no_alias(path.parent, label=f"{label} parent", must_exist=True)
    _assert_no_alias(path, label=label, must_exist=must_exist)
    resolved = path.resolve(strict=must_exist)
    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must stay inside PROJECT_ROOT")
    return path


def _exact_keys(payload: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return payload


def _vector(
    value: Any,
    *,
    label: str,
    length: int,
    positive: bool = False,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must be a {length}-item numeric list")
    result = []
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(f"{label} must be a {length}-item numeric list")
        if positive and float(item) <= 0:
            raise ValueError(f"{label} values must be positive")
        result.append(float(item))
    return result


def _alias_value(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) is None
    ):
        raise ValueError(f"{label} must be a safe object alias")
    return value


def _validate_workspace(payload: Any) -> None:
    workspace = _exact_keys(
        payload,
        {"alias", "size_mm", "center_mm"},
        label="workspace",
    )
    if workspace["alias"] != "Workspace":
        raise ValueError("workspace alias must be Workspace")
    _vector(workspace["size_mm"], label="workspace size_mm", length=3, positive=True)
    _vector(workspace["center_mm"], label="workspace center_mm", length=3)


def _validate_camera(payload: Any, *, root_path: str) -> None:
    camera = _exact_keys(
        payload,
        {
            "alias",
            "path",
            "resolution",
            "position_m",
            "near_clip_m",
            "far_clip_m",
            "perspective_angle_deg",
        },
        label="camera",
    )
    if camera["alias"] != "Camera" or camera["path"] != f"{root_path}/Camera":
        raise ValueError("camera alias/path must match the formal scene root")
    resolution = camera["resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(item) is not int or item <= 0 for item in resolution)
    ):
        raise ValueError("camera resolution must be two positive integers")
    _vector(camera["position_m"], label="camera position_m", length=3)
    near = _vector(
        [camera["near_clip_m"]],
        label="camera near_clip_m",
        length=1,
        positive=True,
    )[0]
    far = _vector(
        [camera["far_clip_m"]],
        label="camera far_clip_m",
        length=1,
        positive=True,
    )[0]
    angle = _vector(
        [camera["perspective_angle_deg"]],
        label="camera perspective_angle_deg",
        length=1,
        positive=True,
    )[0]
    if near >= far or angle >= 180.0:
        raise ValueError("camera clipping range or perspective angle is invalid")


def _validate_markers(payload: Any) -> None:
    if not isinstance(payload, list) or len(payload) != 4:
        raise ValueError("markers must contain four teaching points")
    aliases = []
    for item in payload:
        marker = _exact_keys(item, {"alias", "position_mm"}, label="marker")
        aliases.append(_alias_value(marker["alias"], label="marker alias"))
        _vector(marker["position_mm"], label="marker position_mm", length=3)
    if aliases != [f"TeachPoint{index}" for index in range(1, 5)]:
        raise ValueError("markers must be TeachPoint1 through TeachPoint4")


def _validate_shape_item(item: Any, *, label: str) -> None:
    shape = _exact_keys(
        item,
        {"alias", "shape", "size_mm", "position_mm", "color"},
        label=label,
    )
    _alias_value(shape["alias"], label=f"{label} alias")
    if shape["shape"] not in {"cuboid", "cylinder"}:
        raise ValueError(f"{label} shape is unsupported")
    _vector(shape["size_mm"], label=f"{label} size_mm", length=3, positive=True)
    _vector(shape["position_mm"], label=f"{label} position_mm", length=3)
    color = _vector(shape["color"], label=f"{label} color", length=3)
    if any(component < 0.0 or component > 1.0 for component in color):
        raise ValueError(f"{label} color values must be between 0 and 1")


def _validate_target(item: Any, *, label: str) -> None:
    target = _exact_keys(item, {"alias", "position_mm"}, label=label)
    _alias_value(target["alias"], label=f"{label} alias")
    _vector(target["position_mm"], label=f"{label} position_mm", length=3)


def _validate_tasks(payload: Any) -> None:
    if not isinstance(payload, dict) or list(payload) != ["Stack", "Digits", "Classes"]:
        raise ValueError("tasks must be ordered Stack, Digits, Classes")
    expected_counts = {"Stack": (6, 6), "Digits": (3, 3), "Classes": (4, 4)}
    for name, (object_count, target_count) in expected_counts.items():
        task = _exact_keys(payload[name], {"objects", "targets"}, label=f"task {name}")
        objects = task["objects"]
        targets = task["targets"]
        if not isinstance(objects, list) or len(objects) != object_count:
            raise ValueError(f"task {name} objects count is invalid")
        if not isinstance(targets, list) or len(targets) != target_count:
            raise ValueError(f"task {name} targets count is invalid")
        object_aliases = []
        for item in objects:
            if name == "Digits":
                digit = _exact_keys(
                    item,
                    {"alias", "digit", "position_mm"},
                    label="digit object",
                )
                object_aliases.append(
                    _alias_value(digit["alias"], label="digit object alias")
                )
                if type(digit["digit"]) is not int or digit["digit"] not in (1, 2, 3):
                    raise ValueError("digit object digit must be integer 1, 2, or 3")
                _vector(
                    digit["position_mm"],
                    label="digit object position_mm",
                    length=3,
                )
            else:
                _validate_shape_item(item, label=f"{name} object")
                object_aliases.append(item["alias"])
        if len(object_aliases) != len(set(object_aliases)):
            raise ValueError(f"task {name} object aliases must be unique")
        target_aliases = []
        for item in targets:
            _validate_target(item, label=f"{name} target")
            target_aliases.append(item["alias"])
        if len(target_aliases) != len(set(target_aliases)):
            raise ValueError(f"task {name} target aliases must be unique")
    digits = [item["digit"] for item in payload["Digits"]["objects"]]
    if digits != [1, 2, 3]:
        raise ValueError("Digits objects must be ordered 1, 2, 3")
    classes = {item["alias"]: item for item in payload["Classes"]["objects"]}
    if set(classes) != {"red_block", "blue_block", "green_cylinder", "yellow_cylinder"}:
        raise ValueError("Classes objects must declare the four formal classes")
    if classes["yellow_cylinder"]["position_mm"][1] != -25:
        raise ValueError("yellow_cylinder y position must be -25 mm")
    class_items = list(classes.values())
    for index, first in enumerate(class_items):
        for second in class_items[index + 1 :]:
            separated = any(
                abs(
                    float(first["position_mm"][axis])
                    - float(second["position_mm"][axis])
                )
                >= (
                    float(first["size_mm"][axis])
                    + float(second["size_mm"][axis])
                )
                / 2.0
                for axis in (0, 1)
            )
            if not separated:
                raise ValueError("Classes object footprints must not overlap")


def _validate_vision_samples(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "ReferenceRectangle", "ReferenceCircle", "ReferenceTriangle", "ResolutionTarget",
    }:
        raise ValueError("samples must contain exactly the four formal sample aliases")
    expected = (
        ("ReferenceRectangle", "cuboid", True),
        ("ReferenceCircle", "cylinder", True),
        ("ReferenceTriangle", "triangle", True),
        ("ResolutionTarget", "resolution_target", False),
    )
    for alias, shape, has_color in expected:
        keys = {"shape", "position_m", "size_m"}
        if has_color:
            keys.add("color_rgb")
        else:
            keys.add("stripe_count")
        sample = _exact_keys(payload[alias], keys, label=alias)
        if sample["shape"] != shape:
            raise ValueError(f"{alias} shape must match the formal sample")
        position = _vector(sample["position_m"], label=f"{alias} position_m", length=3)
        size = _vector(sample["size_m"], label=f"{alias} size_m", length=3, positive=True)
        if any(abs(value) > 1.0 for value in position):
            raise ValueError(f"{alias} position_m values must stay within one metre")
        if any(value > 1.0 for value in size):
            raise ValueError(f"{alias} size_m values must stay within one metre")
        if has_color:
            color = _vector(sample["color_rgb"], label=f"{alias} color_rgb", length=3)
            if any(value < 0.0 or value > 1.0 for value in color):
                raise ValueError(f"{alias} color_rgb values must be between 0 and 1")
        elif type(sample["stripe_count"]) is not int or sample["stripe_count"] != 12:
            raise ValueError("ResolutionTarget stripe_count must be integer 12")


def _validate_code_routing(spec: dict[str, Any], formal: _FormalScene) -> None:
    _exact_keys(
        spec,
        {
            "schema_version", "scene_id", "template", "output", "remove_paths",
            "root_path", "code_assets_manifest", "profiles", "workspace", "camera",
            "safe_z_mm", "parts", "bins", "reset_contract", "required_paths",
        },
        label="scene spec",
    )
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("scene schema_version must be integer 1")
    bindings = {
        "scene_id": formal.scene_id,
        "root_path": formal.root_path,
        "template": TEMPLATE_RELATIVE,
        "output": formal.output_relative,
        "code_assets_manifest": "simulation/vision_code_routing_lab/code_assets_manifest.json",
        "profiles": "simulation/vision_code_routing_lab/profiles.json",
    }
    for field, expected in bindings.items():
        if spec[field] != expected:
            raise ValueError(f"{field} must be {expected}")
    if spec["remove_paths"] != ["/VisionLab"]:
        raise ValueError("remove_paths must be exactly ['/VisionLab']")
    if spec["required_paths"] != list(formal.required_paths):
        raise ValueError("required_paths must match the formal code-routing contract")
    if type(spec["safe_z_mm"]) not in (int, float) or float(spec["safe_z_mm"]) != 110.0:
        raise ValueError("safe_z_mm must be 110 mm")

    _validate_workspace(spec["workspace"])
    camera = _exact_keys(
        spec["camera"],
        {"alias", "rig_position_m", "orientation_deg", "code_face_plane_z_mm"},
        label="code-routing camera",
    )
    if camera["alias"] != "Camera":
        raise ValueError("code-routing camera alias must be Camera")
    if _vector(camera["rig_position_m"], label="camera rig_position_m", length=3) != [0.085, 0.0, 0.5]:
        raise ValueError("code-routing camera rig position is fixed")
    if _vector(camera["orientation_deg"], label="camera orientation_deg", length=3) != [180.0, 0.0, 0.0]:
        raise ValueError("code-routing camera orientation is fixed")
    plane = _vector([camera["code_face_plane_z_mm"]], label="camera code_face_plane_z_mm", length=1)[0]
    if plane != 27.4:
        raise ValueError("code face calibration plane must be 27.4 mm")

    parts = spec["parts"]
    if not isinstance(parts, list) or len(parts) != 4:
        raise ValueError("parts must contain four code-routing parts")
    expected_parts = ("part_a", "part_b", "part_c", "part_d")
    expected_assets = {"qr_v1_07_a", "qr_v1_07_b", "ean_6901234567892", "ean_6901234567809"}
    aliases = []
    assets = []
    for item in parts:
        part = _exact_keys(item, {"alias", "code_asset_id", "position_mm", "size_mm"}, label="code-routing part")
        aliases.append(_alias_value(part["alias"], label="part alias"))
        asset_id = _alias_value(part["code_asset_id"], label="code asset id")
        assets.append(asset_id)
        _vector(part["position_mm"], label="part position_mm", length=3)
        _vector(part["size_mm"], label="part size_mm", length=3, positive=True)
    if tuple(aliases) != expected_parts or len(set(aliases)) != 4:
        raise ValueError("code-routing parts must be ordered part_a through part_d")
    if set(assets) != expected_assets:
        raise ValueError("code-routing parts must bind all four original code assets")

    bins = spec["bins"]
    if not isinstance(bins, list) or len(bins) != 2:
        raise ValueError("bins must contain route_red and route_blue")
    bin_aliases = []
    slot_positions = []
    for item in bins:
        bin_spec = _exact_keys(item, {"alias", "color_rgb", "slots"}, label="route bin")
        bin_aliases.append(_alias_value(bin_spec["alias"], label="bin alias"))
        color = _vector(bin_spec["color_rgb"], label="bin color_rgb", length=3)
        if any(value < 0.0 or value > 1.0 for value in color):
            raise ValueError("bin color_rgb values must be between 0 and 1")
        slots = bin_spec["slots"]
        if not isinstance(slots, list) or len(slots) != 2:
            raise ValueError("each route bin must contain two slots")
        slot_aliases = []
        for slot in slots:
            slot_spec = _exact_keys(slot, {"alias", "position_mm"}, label="route slot")
            slot_aliases.append(_alias_value(slot_spec["alias"], label="slot alias"))
            position = _vector(slot_spec["position_mm"], label="slot position_mm", length=3)
            slot_positions.append(tuple(position))
        if len(set(slot_aliases)) != 2:
            raise ValueError("route slot aliases must be unique")
    if bin_aliases != ["route_red", "route_blue"] or len(set(bin_aliases)) != 2:
        raise ValueError("route bins must be ordered route_red and route_blue")
    if len(set(slot_positions)) != 4:
        raise ValueError("route slot positions must be unique")

    reset = _exact_keys(spec["reset_contract"], {"strategy", "tool_off", "robot_home"}, label="reset_contract")
    if reset != {"strategy": "scene_reload", "tool_off": True, "robot_home": True}:
        raise ValueError("reset_contract must bind host-controlled scene reload")


def _validate_ocr_sorting(spec: dict[str, Any], formal: _FormalScene) -> None:
    _exact_keys(
        spec,
        {
            "schema_version",
            "scene_id",
            "template",
            "output",
            "remove_paths",
            "root_path",
            "ocr_assets_manifest",
            "profiles",
            "port",
            "workspace",
            "camera",
            "lighting",
            "safe_z_mm",
            "calibration_matrix",
            "parts",
            "routes",
            "reset_contract",
            "required_paths",
        },
        label="scene spec",
    )
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("scene schema_version must be integer 1")
    expected = {
        "scene_id": formal.scene_id,
        "root_path": formal.root_path,
        "template": TEMPLATE_RELATIVE,
        "output": formal.output_relative,
        "ocr_assets_manifest": "simulation/vision_ocr_sorting_lab/ocr_assets_manifest.json",
        "profiles": "simulation/vision_ocr_sorting_lab/profiles.json",
    }
    for field, value in expected.items():
        if spec[field] != value:
            raise ValueError(f"{field} must be {value}")
    if spec["remove_paths"] != ["/VisionLab"]:
        raise ValueError("remove_paths must be exactly ['/VisionLab']")
    if spec["required_paths"] != list(formal.required_paths):
        raise ValueError("required_paths must match the formal OCR sorting contract")
    if type(spec["port"]) is not int or spec["port"] != 23008:
        raise ValueError("port must be the dedicated V1-08 port 23008")
    if type(spec["safe_z_mm"]) not in (int, float) or float(spec["safe_z_mm"]) != 110.0:
        raise ValueError("safe_z_mm must be 110 mm")

    _validate_workspace(spec["workspace"])
    camera = _exact_keys(
        spec["camera"],
        {"alias", "path", "rig_position_m", "orientation_deg", "code_face_plane_z_mm"},
        label="OCR sorting camera",
    )
    if camera["alias"] != "Camera" or camera["path"] != f"{formal.root_path}/CameraRig/Camera":
        raise ValueError("OCR sorting camera alias/path must match the formal scene root")
    if _vector(camera["rig_position_m"], label="camera rig_position_m", length=3) != [0.085, 0.0, 0.5]:
        raise ValueError("OCR sorting camera rig position is fixed")
    if _vector(camera["orientation_deg"], label="camera orientation_deg", length=3) != [180.0, 0.0, 0.0]:
        raise ValueError("OCR sorting camera orientation is fixed")
    if _vector([camera["code_face_plane_z_mm"]], label="camera code_face_plane_z_mm", length=1)[0] != 27.4:
        raise ValueError("OCR sorting code-face calibration plane must be 27.4 mm")

    lighting = _exact_keys(spec["lighting"], {"key_path", "fill_path"}, label="OCR sorting lighting")
    if lighting != {
        "key_path": f"{formal.root_path}/Lighting/KeyLight",
        "fill_path": f"{formal.root_path}/Lighting/FillLight",
    }:
        raise ValueError("OCR sorting lighting paths are fixed")

    matrix = spec["calibration_matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in row)
            for row in matrix
        )
    ):
        raise ValueError("calibration_matrix must be a finite 2x3 matrix")
    expected_matrix = [
        [-0.1627580685211339, 0.0, 168.25075204856],
        [0.0, 0.1627580685211339, -83.25075204855999],
    ]
    if matrix != expected_matrix:
        raise ValueError("calibration_matrix must match the fixed camera calibration")

    parts = spec["parts"]
    if not isinstance(parts, list) or len(parts) != 4:
        raise ValueError("parts must contain four OCR sorting pickables")
    expected_parts = (
        ("part_a", "A1", "route_alpha"),
        ("part_b", "A2", "route_alpha"),
        ("part_c", "B1", "route_beta"),
        ("part_d", "B2", "route_beta"),
    )
    seen_assets: set[str] = set()
    for item, expected_part in zip(parts, expected_parts):
        part = _exact_keys(
            item,
            {"alias", "identifier", "label_asset_id", "route", "position_mm", "size_mm"},
            label="OCR sorting part",
        )
        alias, identifier, route = expected_part
        if (part["alias"], part["identifier"], part["route"]) != expected_part:
            raise ValueError("OCR sorting parts must be ordered part_a..part_d with fixed IDs/routes")
        if part["label_asset_id"] != identifier or identifier in seen_assets:
            raise ValueError("OCR sorting parts must bind unique scene label assets")
        seen_assets.add(identifier)
        _vector(part["position_mm"], label="OCR part position_mm", length=3)
        size = _vector(part["size_mm"], label="OCR part size_mm", length=3, positive=True)
        if size[0] < 20.0 or size[1] < 20.0 or size[2] < 8.0:
            raise ValueError("OCR part size is too small")

    routes = spec["routes"]
    if not isinstance(routes, list) or len(routes) != 2:
        raise ValueError("routes must contain route_alpha and route_beta")
    route_aliases = []
    slot_positions: list[tuple[float, float, float]] = []
    for route in routes:
        route_spec = _exact_keys(route, {"alias", "color_rgb", "slots"}, label="OCR route")
        route_aliases.append(_alias_value(route_spec["alias"], label="route alias"))
        color = _vector(route_spec["color_rgb"], label="route color_rgb", length=3)
        if any(value < 0.0 or value > 1.0 for value in color):
            raise ValueError("route color values must be between 0 and 1")
        slots = route_spec["slots"]
        if not isinstance(slots, list) or len(slots) != 2:
            raise ValueError("each OCR route must contain two fixed slots")
        slot_aliases = []
        for slot in slots:
            slot_spec = _exact_keys(slot, {"alias", "position_mm"}, label="OCR route slot")
            slot_aliases.append(_alias_value(slot_spec["alias"], label="slot alias"))
            position = _vector(slot_spec["position_mm"], label="slot position_mm", length=3)
            slot_positions.append(tuple(position))
            if not (20.0 <= position[0] <= 140.0 and -90.0 <= position[1] <= 90.0 and 10.0 <= position[2] <= 140.0):
                raise ValueError("OCR route slot is outside the workspace")
        if slot_aliases != ["slot_1", "slot_2"]:
            raise ValueError("OCR route slots must be slot_1 and slot_2")
    if route_aliases != ["route_alpha", "route_beta"] or len(set(slot_positions)) != 4:
        raise ValueError("OCR routes must be ordered and have four unique slots")

    reset = _exact_keys(spec["reset_contract"], {"strategy", "tool_off", "robot_home"}, label="reset_contract")
    if reset != {"strategy": "scene_reload", "tool_off": True, "robot_home": True}:
        raise ValueError("reset_contract must bind host-controlled scene reload")


def _validate_defect_sorting(spec: dict[str, Any], formal: _FormalScene) -> None:
    """Validate the offline, hash-bound V1-09 scene contract."""

    _exact_keys(
        spec,
        {
            "calibration_matrix",
            "camera",
            "defect_assets_manifest",
            "lighting",
            "output",
            "parts",
            "port",
            "profiles",
            "remove_paths",
            "required_paths",
            "reset_contract",
            "reference",
            "rois",
            "root_path",
            "safe_z_mm",
            "scene_id",
            "schema_version",
            "slots",
            "template",
            "workspace",
        },
        label="V1-09 scene spec",
    )
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("V1-09 scene schema_version must be integer 1")
    bindings = {
        "scene_id": formal.scene_id,
        "root_path": formal.root_path,
        "template": TEMPLATE_RELATIVE,
        "output": formal.output_relative,
        "defect_assets_manifest": "simulation/vision_defect_sorting_lab/defect_assets_manifest.json",
        "profiles": "simulation/vision_defect_sorting_lab/profiles.json",
    }
    for field, expected in bindings.items():
        if spec[field] != expected:
            raise ValueError(f"{field} must be {expected}")
    if spec["remove_paths"] != ["/VisionLab"]:
        raise ValueError("V1-09 remove_paths must be exactly ['/VisionLab']")
    if spec["required_paths"] != list(formal.required_paths):
        raise ValueError("V1-09 required paths must match the formal scene contract")
    if type(spec["port"]) is not int or spec["port"] != 23010:
        raise ValueError("V1-09 must use dedicated port 23010")
    if type(spec["safe_z_mm"]) not in (int, float) or float(spec["safe_z_mm"]) != 110.0:
        raise ValueError("V1-09 safe_z_mm must be 110 mm")

    _validate_workspace(spec["workspace"])
    workspace = spec["workspace"]
    center = _vector(workspace["center_mm"], label="workspace center_mm", length=3)
    size = _vector(workspace["size_mm"], label="workspace size_mm", length=3, positive=True)
    if center != [85.0, 0.0, 5.0] or size != [150.0, 220.0, 10.0]:
        raise ValueError("V1-09 workspace geometry is fixed")

    reference = _exact_keys(
        spec["reference"],
        {"alias", "asset_id", "position_mm", "size_mm"},
        label="V1-09 reference",
    )
    if reference["alias"] != "reference" or reference["asset_id"] != "reference":
        raise ValueError("V1-09 reference identity is fixed")
    if tuple(_vector(reference["position_mm"], label="V1-09 reference position_mm", length=3)) != (85.0, -57.0, 18.0):
        raise ValueError("V1-09 reference position is fixed")
    if _vector(reference["size_mm"], label="V1-09 reference size_mm", length=3, positive=True) != [28.0, 28.0, 16.0]:
        raise ValueError("V1-09 reference geometry is fixed")

    camera = _exact_keys(
        spec["camera"],
        {"alias", "orientation_deg", "path", "rig_position_m", "surface_face_plane_z_mm"},
        label="V1-09 camera",
    )
    if camera["alias"] != "Camera" or camera["path"] != f"{formal.root_path}/CameraRig/Camera":
        raise ValueError("V1-09 camera alias/path must match the formal scene root")
    if _vector(camera["rig_position_m"], label="camera rig_position_m", length=3) != [0.085, 0.0, 0.5]:
        raise ValueError("V1-09 camera rig position is fixed")
    if _vector(camera["orientation_deg"], label="camera orientation_deg", length=3) != [180.0, 0.0, 0.0]:
        raise ValueError("V1-09 camera orientation is fixed")
    if _vector([camera["surface_face_plane_z_mm"]], label="surface face plane", length=1)[0] != 27.4:
        raise ValueError("V1-09 surface face plane must be 27.4 mm")

    lighting = _exact_keys(spec["lighting"], {"fill_path", "key_path"}, label="V1-09 lighting")
    if lighting != {
        "key_path": f"{formal.root_path}/Lighting/KeyLight",
        "fill_path": f"{formal.root_path}/Lighting/FillLight",
    }:
        raise ValueError("V1-09 lighting paths are fixed")

    matrix = spec["calibration_matrix"]
    if not isinstance(matrix, list) or len(matrix) != 2 or any(
        not isinstance(row, list)
        or len(row) != 3
        or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in row)
        for row in matrix
    ):
        raise ValueError("V1-09 calibration_matrix must be finite 2x3")
    expected_matrix = [
        [-0.1627580685211339, 0.0, 168.25075204856],
        [0.0, 0.1627580685211339, -83.25075204855999],
    ]
    if matrix != expected_matrix:
        raise ValueError("V1-09 calibration_matrix must match the fixed camera calibration")

    rois = spec["rois"]
    expected_roi_ids = ("entry_a", "entry_b", "entry_c", "entry_d", "entry_e", "entry_f", "reference")
    if not isinstance(rois, dict) or tuple(sorted(rois)) != expected_roi_ids:
        raise ValueError("V1-09 ROIs must contain the reference and six entry IDs")
    rectangles: list[tuple[int, int, int, int]] = []
    for key in expected_roi_ids:
        roi = rois[key]
        if not isinstance(roi, list) or len(roi) != 4 or any(type(value) is not int for value in roi):
            raise ValueError("V1-09 ROI must be four integers")
        x, y, width, height = roi
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1024 or y + height > 1024:
            raise ValueError("V1-09 ROI is outside the fixed 1024x1024 frame")
        rectangles.append((x, y, width, height))
    for index, (x, y, width, height) in enumerate(rectangles):
        for other_x, other_y, other_width, other_height in rectangles[index + 1 :]:
            if not (x + width <= other_x or other_x + other_width <= x or y + height <= other_y or other_y + other_height <= y):
                raise ValueError("V1-09 ROIs must not overlap")

    expected_parts = tuple(f"part_{letter}" for letter in "abcdef")
    parts = spec["parts"]
    if not isinstance(parts, list) or len(parts) != 6:
        raise ValueError("V1-09 must contain six candidate parts")
    positions = ((140, -16, 18), (85, -16, 18), (30, -16, 18), (140, 38, 18), (85, 38, 18), (30, 38, 18))
    for index, (item, expected_alias, expected_position) in enumerate(zip(parts, expected_parts, positions)):
        part = _exact_keys(item, {"alias", "asset_id", "position_mm", "size_mm"}, label="V1-09 part")
        if part["alias"] != expected_alias or part["asset_id"] != f"entry_{expected_alias[-1]}":
            raise ValueError("V1-09 parts must bind neutral entry IDs in order")
        if tuple(_vector(part["position_mm"], label="V1-09 part position_mm", length=3)) != expected_position:
            raise ValueError("V1-09 part positions are fixed")
        if _vector(part["size_mm"], label="V1-09 part size_mm", length=3, positive=True) != [28.0, 28.0, 16.0]:
            raise ValueError("V1-09 candidate geometry must be identical")

    expected_decisions = ("qualified", "missing", "hole", "foreign", "broken", "dimension")
    slots = spec["slots"]
    if not isinstance(slots, list) or len(slots) != 6:
        raise ValueError("V1-09 must contain six decision slots")
    slot_positions: set[tuple[float, float, float]] = set()
    slot_positions_expected = ((132, -93, 22), (85, -93, 22), (38, -93, 22), (132, 75, 22), (85, 75, 22), (38, 75, 22))
    for slot, decision, expected_position in zip(slots, expected_decisions, slot_positions_expected):
        item = _exact_keys(slot, {"alias", "color_rgb", "decision", "position_mm"}, label="V1-09 slot")
        if item["alias"] != f"slot_{decision}" or item["decision"] != decision:
            raise ValueError("V1-09 slots must be ordered by decision")
        position = tuple(_vector(item["position_mm"], label="V1-09 slot position_mm", length=3))
        if position != expected_position or position in slot_positions:
            raise ValueError("V1-09 slot positions are fixed and unique")
        slot_positions.add(position)
        color = _vector(item["color_rgb"], label="V1-09 slot color_rgb", length=3)
        if any(value < 0.0 or value > 1.0 for value in color):
            raise ValueError("V1-09 slot colors must be between 0 and 1")

    reset = _exact_keys(spec["reset_contract"], {"strategy", "tool_off", "robot_home"}, label="V1-09 reset_contract")
    if reset != {"strategy": "scene_reload", "tool_off": True, "robot_home": True}:
        raise ValueError("V1-09 reset_contract must bind host-controlled scene reload")
    _project_file_from_relative(spec["profiles"], label="profiles", must_exist=True)
    _load_defect_profile_catalog(_project_file_from_relative(spec["profiles"], label="profiles", must_exist=True))
    _project_file_from_relative(spec["defect_assets_manifest"], label="defect_assets_manifest", must_exist=False)


def _validate_spec(spec: dict[str, Any], formal: _FormalScene) -> None:
    if formal.scene_id == "robot-basics":
        detail_field = "markers"
    elif formal.scene_id == "logistics-lab":
        detail_field = "tasks"
    elif formal.scene_id == "vision-quality-lab":
        _exact_keys(spec, {"schema_version", "scene_id", "template", "output", "remove_paths", "root_path", "profiles", "workspace", "camera", "samples", "required_paths"}, label="scene spec")
        if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
            raise ValueError("scene schema_version must be integer 1")
        for field, expected in {"scene_id": formal.scene_id, "root_path": formal.root_path, "template": TEMPLATE_RELATIVE, "output": formal.output_relative}.items():
            if spec[field] != expected:
                raise ValueError(f"{field} must be {expected}")
        if spec["remove_paths"] != ["/VisionLab"] or spec["required_paths"] != list(formal.required_paths):
            raise ValueError("vision-quality formal paths must match the contract")
        profiles = "simulation/vision_quality_lab/profiles.json"
        if spec["profiles"] != profiles:
            raise ValueError("profiles must be the canonical vision-quality catalog")
        load_profile_catalog(_project_file_from_relative(profiles, label="profiles", must_exist=True))
        workspace = _exact_keys(spec["workspace"], {"center_m", "size_m"}, label="workspace")
        _vector(workspace["center_m"], label="workspace center_m", length=3)
        _vector(workspace["size_m"], label="workspace size_m", length=3, positive=True)
        camera = _exact_keys(spec["camera"], {"rig_position_m", "orientation_deg"}, label="camera")
        _vector(camera["rig_position_m"], label="camera rig_position_m", length=3)
        _vector(camera["orientation_deg"], label="camera orientation_deg", length=3)
        _validate_vision_samples(spec["samples"])
        return
    elif formal.scene_id == "vision-code-routing-lab":
        _validate_code_routing(spec, formal)
        _project_file_from_relative(spec["profiles"], label="profiles", must_exist=True)
        _project_file_from_relative(
            spec["code_assets_manifest"],
            label="code_assets_manifest",
            must_exist=True,
        )
        load_profile_catalog(
            _project_file_from_relative(spec["profiles"], label="profiles", must_exist=True)
        )
        return
    elif formal.scene_id == "vision-ocr-sorting-lab":
        _validate_ocr_sorting(spec, formal)
        _project_file_from_relative(
            spec["profiles"],
            label="profiles",
            must_exist=True,
        )
        ocr_assets_path = _project_file_from_relative(
            spec["ocr_assets_manifest"],
            label="ocr_assets_manifest",
            must_exist=True,
        )
        _validate_ocr_assets_manifest(ocr_assets_path)
        _load_ocr_profile_catalog(
            _project_file_from_relative(spec["profiles"], label="profiles", must_exist=True)
        )
        return
    elif formal.scene_id == "vision-defect-sorting-lab":
        _validate_defect_sorting(spec, formal)
        return
    else:
        raise ValueError("unsupported formal scene")
    _exact_keys(
        spec,
        {
            "schema_version",
            "scene_id",
            "template",
            "output",
            "remove_paths",
            "root_path",
            "workspace",
            "camera",
            detail_field,
            "required_paths",
        },
        label="scene spec",
    )
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("scene schema_version must be integer 1")
    bindings = {
        "scene_id": formal.scene_id,
        "root_path": formal.root_path,
        "template": TEMPLATE_RELATIVE,
        "output": formal.output_relative,
    }
    for field, expected in bindings.items():
        if spec[field] != expected:
            raise ValueError(f"{field} must be {expected}")
    if spec["remove_paths"] != ["/VisionLab"]:
        raise ValueError("remove_paths must be exactly ['/VisionLab']")
    if spec["required_paths"] != list(formal.required_paths):
        raise ValueError("required_paths must match the formal scene contract")
    _validate_workspace(spec["workspace"])
    _validate_camera(spec["camera"], root_path=formal.root_path)
    if formal.scene_id == "robot-basics":
        _validate_markers(spec["markers"])
    else:
        _validate_tasks(spec["tasks"])


def _formal_spec_path(spec_path: str | Path) -> tuple[Path, _FormalScene]:
    candidate = _lexical_project_path(spec_path, label="spec_path")
    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("spec_path must stay inside PROJECT_ROOT") from exc
    formal = _FORMAL_SCENES.get(relative)
    if formal is None:
        raise ValueError("spec_path must be an approved formal spec")
    expected = root / formal.spec_relative
    if candidate != expected:
        raise ValueError("spec_path must be the canonical approved formal spec")
    _assert_no_alias(candidate, label="spec_path", must_exist=True)
    if candidate.resolve(strict=True) != expected:
        raise ValueError("spec_path must not use an alias")
    return candidate, formal


def _files_under(relative: str) -> list[Path]:
    path = _project_file_from_relative(
        relative,
        label=f"protected path {relative}",
        must_exist=True,
    )
    if path.is_file():
        return [path]
    items = sorted(path.rglob("*"))
    for item in items:
        _assert_no_alias(item, label=f"protected entry {item}", must_exist=True)
    files = [item for item in items if item.is_file()]
    if not files:
        raise ValueError(f"protected directory is empty: {relative}")
    return files


def _protected_hashes() -> dict[str, str]:
    expected_manifest = (
        PROJECT_ROOT.expanduser().resolve(strict=True)
        / "simulation/vision_lab/source_manifest.json"
    )
    manifest_path = _lexical_project_path(
        SOURCE_MANIFEST,
        label="SOURCE_MANIFEST",
    )
    if manifest_path != expected_manifest:
        raise ValueError("SOURCE_MANIFEST must use the formal project path")
    _assert_no_alias(
        manifest_path,
        label="SOURCE_MANIFEST",
        must_exist=True,
    )
    source = _load(manifest_path)
    if type(source.get("schema_version")) is not int or source["schema_version"] != 1:
        raise ValueError("SOURCE_MANIFEST schema_version must be integer 1")
    entries = source.get("protected_formal_assets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("SOURCE_MANIFEST protected_formal_assets must be non-empty")

    expected_declared = {
        path.relative_to(PROJECT_ROOT.expanduser().resolve(strict=True)).as_posix()
        for relative in DECLARED_PROTECTED_ROOTS
        for path in _files_under(relative)
    }
    declared: set[str] = set()
    result: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) not in (
            {"path", "sha256"},
            {"path", "sha256", "hash_mode"},
        ):
            raise ValueError("protected asset entry has invalid fields")
        raw = item["path"]
        path = _project_file_from_relative(
            raw,
            label="protected asset path",
            must_exist=True,
        )
        relative = path.relative_to(
            PROJECT_ROOT.expanduser().resolve(strict=True)
        ).as_posix()
        if relative not in expected_declared or relative in declared:
            raise ValueError(
                "protected asset entries must uniquely cover formal model assets"
            )
        declared.add(relative)
        expected_hash = item["sha256"]
        if (
            not isinstance(expected_hash, str)
            or _LOWER_SHA256.fullmatch(expected_hash) is None
        ):
            raise ValueError("protected asset sha256 must be lowercase hexadecimal")
        mode = item.get("hash_mode", "bytes")
        if mode not in {"bytes", "text_lf"}:
            raise ValueError("protected asset hash_mode is unsupported")
        actual = asset_sha256(path, mode=mode)
        if actual != expected_hash:
            raise RuntimeError(f"protected asset hash mismatch: {relative}")
        result[f"declared:{relative}"] = actual
    if declared != expected_declared:
        raise ValueError(
            "SOURCE_MANIFEST must hash every protected formal model asset"
        )

    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    for relative in PROTECTED_ROOTS:
        for path in _files_under(relative):
            item_relative = path.relative_to(root).as_posix()
            result[f"snapshot:{item_relative}"] = _sha256(path)
    return result


def _alias(sim: Any, handle: int, name: str) -> int:
    sim.setObjectAlias(handle, name)
    return int(handle)


def _dummy(sim: Any, name: str, parent: int | None = None) -> int:
    handle = _alias(sim, int(sim.createDummy(0.005)), name)
    if parent is not None:
        sim.setObjectParent(handle, parent, False)
    return handle


def _set_int_parameter(
    sim: Any,
    handle: int,
    parameter: int,
    value: int,
) -> None:
    setter = getattr(sim, "setObjectInt32Param", None)
    if not callable(setter):
        setter = getattr(sim, "setObjectInt32Parameter")
    setter(handle, parameter, int(value))


def _shape(
    sim: Any,
    *,
    name: str,
    shape: str,
    size_mm: list[float],
    position_mm: list[float],
    color: list[float],
    parent: int,
    respondable: bool,
) -> int:
    primitive = (
        sim.primitiveshape_cylinder
        if shape == "cylinder"
        else sim.primitiveshape_cuboid
    )
    handle = _alias(
        sim,
        int(
            sim.createPrimitiveShape(
                primitive,
                [float(value) / 1000.0 for value in size_mm],
                0,
            )
        ),
        name,
    )
    sim.setShapeColor(
        handle,
        "",
        sim.colorcomponent_ambient_diffuse,
        [float(value) for value in color],
    )
    _set_int_parameter(sim, handle, sim.shapeintparam_static, 1)
    _set_int_parameter(
        sim,
        handle,
        sim.shapeintparam_respondable,
        int(respondable),
    )
    sim.setObjectParent(handle, parent, False)
    sim.setObjectPosition(
        handle,
        [float(value) / 1000.0 for value in position_mm],
        parent,
    )
    return handle


def _camera(sim: Any, spec: dict[str, Any], parent: int) -> int:
    resolution = [int(value) for value in spec["resolution"]]
    options = 2 | 4 | 64 | 128
    handle = int(
        sim.createVisionSensor(
            options,
            [resolution[0], resolution[1], 0, 0],
            [
                float(spec["near_clip_m"]),
                float(spec["far_clip_m"]),
                math.radians(float(spec["perspective_angle_deg"])),
                0.02,
                0.0,
                0.0,
                0.08,
                0.08,
                0.10,
                0.0,
                0.0,
            ],
        )
    )
    _alias(sim, handle, spec["alias"])
    sim.setObjectPose(
        handle,
        [
            *[float(value) for value in spec["position_m"]],
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        sim.handle_world,
    )
    sim.setObjectParent(handle, parent, True)
    return handle


_SEGMENTS = {
    1: ("b", "c"),
    2: ("a", "b", "g", "e", "d"),
    3: ("a", "b", "g", "c", "d"),
}
_SEGMENT_POSES = {
    "a": ([0, 5, 10], [10, 2, 2]),
    "b": ([5, 0, 10], [2, 10, 2]),
    "c": ([5, -10, 10], [2, 10, 2]),
    "d": ([0, -15, 10], [10, 2, 2]),
    "e": ([-5, -10, 10], [2, 10, 2]),
    "g": ([0, -5, 10], [10, 2, 2]),
}


def _digit(sim: Any, item: dict[str, Any], parent: int) -> int:
    center = [float(value) for value in item["position_mm"]]
    parts = []
    for segment in _SEGMENTS[int(item["digit"])]:
        offset, size = _SEGMENT_POSES[segment]
        parts.append(
            _shape(
                sim,
                name=f"{item['alias']}_{segment}",
                shape="cuboid",
                size_mm=size,
                position_mm=[
                    center[0] + offset[0],
                    center[1] + offset[1],
                    center[2] + offset[2],
                ],
                color=[0.04, 0.04, 0.04],
                parent=parent,
                respondable=False,
            )
        )
    # CoppeliaSim uses the final grouped shape as the compound reference.
    parts.append(
        _shape(
            sim,
            name=f"{item['alias']}_plate",
            shape="cuboid",
            size_mm=[20, 30, 8],
            position_mm=center,
            color=[0.92, 0.92, 0.92],
            parent=parent,
            respondable=True,
        )
    )
    compound = int(sim.groupShapes(parts, False))
    _alias(sim, compound, item["alias"])
    sim.setObjectParent(compound, parent, True)
    return compound


def _build_workspace(sim: Any, spec: dict[str, Any], root: int) -> None:
    _shape(
        sim,
        name=spec["alias"],
        shape="cuboid",
        size_mm=spec["size_mm"],
        position_mm=spec["center_mm"],
        color=[0.30, 0.32, 0.34],
        parent=root,
        respondable=True,
    )


def _build_basics(sim: Any, spec: dict[str, Any], root: int) -> None:
    points = _dummy(sim, "TeachPoints", root)
    for marker in spec["markers"]:
        _shape(
            sim,
            name=marker["alias"],
            shape="cylinder",
            size_mm=[8, 8, 2],
            position_mm=marker["position_mm"],
            color=[0.12, 0.75, 0.85],
            parent=points,
            respondable=False,
        )


def _build_logistics(sim: Any, spec: dict[str, Any], root: int) -> None:
    tasks = _dummy(sim, "Tasks", root)
    for index, (name, task) in enumerate(spec["tasks"].items()):
        group = _dummy(sim, name, tasks)
        sim.setObjectPosition(
            group,
            [0.0, 0.0, 0.0 if index == 0 else -float(index + 2)],
            sim.handle_world,
        )
        pickables = _dummy(sim, "Pickables", group)
        targets = _dummy(sim, "Targets", group)
        for item in task["objects"]:
            if "digit" in item:
                _digit(sim, item, pickables)
            else:
                _shape(
                    sim,
                    name=item["alias"],
                    shape=item["shape"],
                    size_mm=item["size_mm"],
                    position_mm=item["position_mm"],
                    color=item["color"],
                    parent=pickables,
                    respondable=True,
                )
        for target in task["targets"]:
            handle = _shape(
                sim,
                name=target["alias"],
                shape="cuboid",
                size_mm=[24, 24, 2],
                position_mm=target["position_mm"],
                color=[0.35, 0.75, 0.95],
                parent=targets,
                respondable=False,
            )
            transparency = getattr(
                sim,
                "colorcomponent_transparency",
                None,
            )
            if transparency is not None:
                sim.setShapeColor(handle, "", transparency, [0.70])


def _metres_as_millimetres(values: Any) -> list[float]:
    return [float(value) * 1000.0 for value in values]


def _build_resolution_target(
    sim: Any,
    spec: dict[str, Any],
    parent: int,
) -> int:
    target = _dummy(sim, "ResolutionTarget", parent)
    sim.setObjectPosition(
        target,
        [float(value) for value in spec["position_m"]],
        parent,
    )
    width_m, height_m, depth_m = (
        float(value) for value in spec["size_m"]
    )
    stripe_count = int(spec["stripe_count"])
    stripe_width_m = width_m / stripe_count
    for index in range(stripe_count):
        center_x_m = (
            -width_m / 2.0
            + stripe_width_m * (index + 0.5)
        )
        _shape(
            sim,
            name=f"Stripe{index + 1:02d}",
            shape="cuboid",
            size_mm=_metres_as_millimetres(
                [stripe_width_m, height_m, depth_m]
            ),
            position_mm=_metres_as_millimetres(
                [center_x_m, 0.0, 0.0]
            ),
            color=[0.96, 0.96, 0.96]
            if index % 2 == 0
            else [0.03, 0.03, 0.03],
            parent=target,
            respondable=False,
        )
    return target


def _build_triangle_prism(
    sim: Any,
    spec: dict[str, Any],
    parent: int,
) -> int:
    center_x, center_y, center_z = (
        float(value) for value in spec["position_m"]
    )
    width_m, height_m, depth_m = (
        float(value) for value in spec["size_m"]
    )
    bands = 6
    band_height_m = height_m / bands
    parts = []
    for index in range(bands):
        band_width_m = width_m * (bands - index) / bands
        band_y_m = (
            center_y
            - height_m / 2.0
            + band_height_m * (index + 0.5)
        )
        parts.append(
            _shape(
                sim,
                name=f"ReferenceTriangleBand{index + 1:02d}",
                shape="cuboid",
                size_mm=_metres_as_millimetres(
                    [band_width_m, band_height_m, depth_m]
                ),
                position_mm=_metres_as_millimetres(
                    [center_x, band_y_m, center_z]
                ),
                color=[float(value) for value in spec["color_rgb"]],
                parent=parent,
                respondable=False,
            )
        )
    compound = int(sim.groupShapes(parts, False))
    _alias(sim, compound, "ReferenceTriangle")
    sim.setObjectParent(compound, parent, True)
    _set_int_parameter(sim, compound, sim.shapeintparam_static, 1)
    _set_int_parameter(sim, compound, sim.shapeintparam_respondable, 0)
    return compound


def _build_vision_lighting(
    sim: Any,
    catalog: Any,
    parent: int,
) -> tuple[int, int]:
    lighting = _dummy(sim, "Lighting", parent)
    profile = catalog.require(catalog.baseline_profile_id)
    definitions = (
        (
            "/DefaultLights/LightA",
            "KeyLight",
            [0.18, -0.20, 0.78],
            profile.key_diffuse_rgb,
        ),
        (
            "/DefaultLights/LightB",
            "FillLight",
            [0.52, 0.18, 0.62],
            profile.fill_diffuse_rgb,
        ),
    )
    result = []
    for source_path, alias, position, diffuse in definitions:
        source = int(sim.getObject(source_path))
        copied = sim.copyPasteObjects([source], 0)
        if not isinstance(copied, (list, tuple)) or len(copied) != 1:
            raise RuntimeError(f"could not copy scene light: {source_path}")
        handle = int(copied[0])
        sim.setLightParameters(
            source,
            0,
            None,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        _alias(sim, handle, alias)
        sim.setObjectParent(handle, lighting, False)
        sim.setObjectPosition(handle, position, sim.handle_world)
        sim.setObjectOrientation(
            handle,
            [math.radians(180.0), 0.0, 0.0],
            sim.handle_world,
        )
        sim.setLightParameters(
            handle,
            1,
            None,
            [float(value) for value in diffuse],
            [0.0, 0.0, 0.0],
        )
        result.append(handle)
    return int(result[0]), int(result[1])


def _assert_samples_in_standard_frame(
    spec: dict[str, Any],
    profile: Any,
    *,
    near_clip_m: float,
    far_clip_m: float,
) -> None:
    orientation = [float(value) for value in spec["camera"]["orientation_deg"]]
    if any(
        not math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(orientation, (180.0, 0.0, 0.0))
    ):
        raise ValueError("vision-quality camera must point vertically down")
    camera_x, camera_y, camera_z = (
        float(value) for value in spec["camera"]["rig_position_m"]
    )
    half_angle = math.radians(float(profile.perspective_angle_deg)) / 2.0
    for alias, sample in spec["samples"].items():
        center_x, center_y, center_z = (
            float(value) for value in sample["position_m"]
        )
        size_x, size_y, size_z = (
            float(value) for value in sample["size_m"]
        )
        nearest_depth = camera_z - (center_z + size_z / 2.0)
        farthest_depth = camera_z - (center_z - size_z / 2.0)
        if not (
            near_clip_m < nearest_depth
            and farthest_depth < far_clip_m
        ):
            raise ValueError(f"{alias} is outside camera clipping range")
        half_view = nearest_depth * math.tan(half_angle)
        if (
            abs(center_x - camera_x) + size_x / 2.0 >= half_view
            or abs(center_y - camera_y) + size_y / 2.0 >= half_view
        ):
            raise ValueError(f"{alias} is outside the standard camera frame")


def _build_vision_quality(
    sim: Any,
    spec: dict[str, Any],
    root: int,
    *,
    profile_catalog: Any | None = None,
) -> None:
    if profile_catalog is None:
        catalog_path = _project_file_from_relative(
            spec["profiles"],
            label="profiles",
            must_exist=True,
        )
        profile_catalog = load_profile_catalog(catalog_path)
    catalog = profile_catalog
    standard = catalog.require(catalog.baseline_profile_id)
    _assert_samples_in_standard_frame(
        spec,
        standard,
        near_clip_m=float(catalog.near_clip_m),
        far_clip_m=float(catalog.far_clip_m),
    )

    workspace = spec["workspace"]
    _shape(
        sim,
        name="InspectionBoard",
        shape="cuboid",
        size_mm=_metres_as_millimetres(workspace["size_m"]),
        position_mm=_metres_as_millimetres(workspace["center_m"]),
        color=[0.86, 0.88, 0.90],
        parent=root,
        respondable=False,
    )
    samples = _dummy(sim, "Samples", root)
    for alias in ("ReferenceRectangle", "ReferenceCircle"):
        item = spec["samples"][alias]
        _shape(
            sim,
            name=alias,
            shape=item["shape"],
            size_mm=_metres_as_millimetres(item["size_m"]),
            position_mm=_metres_as_millimetres(item["position_m"]),
            color=[float(value) for value in item["color_rgb"]],
            parent=samples,
            respondable=False,
        )
    _build_triangle_prism(
        sim,
        spec["samples"]["ReferenceTriangle"],
        samples,
    )
    _build_resolution_target(
        sim,
        spec["samples"]["ResolutionTarget"],
        samples,
    )

    rig = _dummy(sim, "CameraRig", root)
    sim.setObjectPosition(
        rig,
        [float(value) for value in spec["camera"]["rig_position_m"]],
        sim.handle_world,
    )
    options = 1 | 2 | 4 | 64 | 128
    camera = int(
        sim.createVisionSensor(
            options,
            [int(standard.resolution[0]), int(standard.resolution[1]), 0, 0],
            [
                float(catalog.near_clip_m),
                float(catalog.far_clip_m),
                math.radians(float(standard.perspective_angle_deg)),
                0.02,
                0.0,
                0.0,
                0.08,
                0.08,
                0.10,
                0.0,
                0.0,
            ],
        )
    )
    _alias(sim, camera, "Camera")
    sim.setObjectParent(camera, rig, False)
    sim.setObjectPosition(camera, [0.0, 0.0, 0.0], rig)
    sim.setObjectOrientation(
        camera,
        [
            math.radians(float(value))
            for value in spec["camera"]["orientation_deg"]
        ],
        rig,
    )
    _build_vision_lighting(sim, catalog, root)
    for path in (
        catalog.sensor_path,
        catalog.camera_rig_path,
        catalog.key_light_path,
        catalog.fill_light_path,
    ):
        sim.getObject(path)


def _code_rectangles(code_type: str, payload: str) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    if code_type == "qr":
        encoded = cv2.QRCodeEncoder_create().encode(payload)
        gray = cv2.copyMakeBorder(encoded, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    elif code_type == "ean13":
        gray = cv2.cvtColor(
            encode_ean13_payload(payload, module_px=3, bar_height_px=72),
            cv2.COLOR_BGR2GRAY,
        )
    else:
        raise ValueError(f"unsupported code type: {code_type}")
    mask = np.asarray(gray < 128, dtype=np.uint8)
    rectangles: list[tuple[int, int, int, int]] = []
    if code_type == "qr":
        for row, column in np.argwhere(mask):
            rectangles.append((int(column), int(row), 1, 1))
    else:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        for component in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[component])
            if area > 0:
                rectangles.append((x, y, width, height))
    return gray, rectangles


def _code_bar_width_mm(code_type: str, width_px: float, scale_mm_per_px: float) -> float:
    """Return the physical width of one rendered code bar."""
    del code_type
    return float(width_px) * float(scale_mm_per_px)


def _code_scale_mm_per_px(code_type: str, scale_mm_per_px: float) -> float:
    """Phase-lock EAN modules to the fixed camera's pixel grid."""
    scale = float(scale_mm_per_px)
    if code_type == "ean13":
        return max(scale, 0.117)
    return scale


def _code_render_x(code_type: str, x_px: int, width_px: int, image_width_px: int) -> int:
    """Compensate CoppeliaSim's horizontal image-axis orientation for EAN."""
    if code_type == "ean13":
        return int(image_width_px) - int(x_px) - int(width_px)
    return int(x_px)


def _code_part(sim: Any, part: dict[str, Any], asset: dict[str, Any], parent: int) -> int:
    center = [float(value) for value in part["position_mm"]]
    size = [float(value) for value in part["size_mm"]]
    pieces = [
        _shape(
            sim,
            name=f"{part['alias']}_body",
            shape="cuboid",
            size_mm=size,
            position_mm=center,
            color=[0.75, 0.75, 0.75],
            parent=parent,
            respondable=True,
        )
    ]
    face_width = size[0] - 4.0
    face_height = size[1] - 4.0
    face_z = center[2] + size[2] / 2.0 + 0.6
    pieces.append(
        _shape(
            sim,
            name=f"{part['alias']}_face_plate",
            shape="cuboid",
            size_mm=[face_width, face_height, 1.0],
            position_mm=[center[0], center[1], face_z],
            color=[0.98, 0.98, 0.98],
            parent=parent,
            respondable=False,
        )
    )
    gray, rectangles = _code_rectangles(asset["code_type"], asset["payload"])
    scale = min(face_width / float(gray.shape[1]), face_height / float(gray.shape[0]))
    scale = _code_scale_mm_per_px(asset["code_type"], scale)
    origin_x = center[0] - gray.shape[1] * scale / 2.0
    origin_y = center[1] - gray.shape[0] * scale / 2.0
    for index, (x, y, width, height) in enumerate(rectangles):
        render_x = _code_render_x(asset["code_type"], x, width, gray.shape[1])
        pieces.append(
            _shape(
                sim,
                name=f"{part['alias']}_code_{index:04d}",
                shape="cuboid",
                size_mm=[
                    _code_bar_width_mm(asset["code_type"], width, scale),
                    height * scale,
                    0.3,
                ],
                position_mm=[
                    origin_x + (render_x + width / 2.0) * scale,
                    origin_y + (y + height / 2.0) * scale,
                    face_z + 0.65,
                ],
                color=[0.01, 0.01, 0.01],
                parent=parent,
                respondable=False,
            )
        )
    compound = int(sim.groupShapes(pieces, False))
    _alias(sim, compound, part["alias"])
    sim.setObjectParent(compound, parent, True)
    relocate_frame = getattr(sim, "relocateShapeFrame", None)
    if callable(relocate_frame):
        relocate_frame(
            compound,
            [
                center[0] / 1000.0,
                center[1] / 1000.0,
                center[2] / 1000.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        )
    _set_int_parameter(sim, compound, sim.shapeintparam_static, 1)
    _set_int_parameter(sim, compound, sim.shapeintparam_respondable, 1)
    _dummy(sim, "CodeFace", compound)
    return compound


def _build_bin(sim: Any, specification: dict[str, Any], parent: int) -> None:
    group = _dummy(sim, specification["alias"], parent)
    slots = specification["slots"]
    center_x = sum(float(slot["position_mm"][0]) for slot in slots) / len(slots)
    center_y = sum(float(slot["position_mm"][1]) for slot in slots) / len(slots)
    color = [float(value) for value in specification["color_rgb"]]
    _shape(
        sim,
        name="Floor",
        shape="cuboid",
        size_mm=[34, 34, 2],
        position_mm=[center_x, center_y, 11],
        color=color,
        parent=group,
        respondable=True,
    )
    for name, dx, dy, sx, sy in (
        ("WallLeft", -18, 0, 2, 38),
        ("WallRight", 18, 0, 2, 38),
        ("WallFront", 0, -18, 38, 2),
        ("WallBack", 0, 18, 38, 2),
    ):
        _shape(
            sim,
            name=name,
            shape="cuboid",
            size_mm=[sx, sy, 12],
            position_mm=[center_x + dx, center_y + dy, 17],
            color=color,
            parent=group,
            respondable=True,
        )
    for slot in slots:
        handle = _dummy(sim, slot["alias"], group)
        sim.setObjectPosition(
            handle,
            [float(value) / 1000.0 for value in slot["position_mm"]],
            group,
        )


def _build_code_routing(
    sim: Any,
    spec: dict[str, Any],
    root: int,
    *,
    code_assets: dict[str, Any],
    profile_catalog: Any,
) -> None:
    _build_workspace(sim, spec["workspace"], root)
    parts_group = _dummy(sim, "Parts", root)
    bins_group = _dummy(sim, "Bins", root)
    camera_rig = _dummy(sim, "CameraRig", root)
    standard = profile_catalog.require(profile_catalog.baseline_profile_id)
    camera_spec = spec["camera"]
    sim.setObjectPosition(
        camera_rig,
        [float(value) for value in camera_spec["rig_position_m"]],
        sim.handle_world,
    )
    options = 1 | 2 | 4 | 64 | 128
    camera = int(
        sim.createVisionSensor(
            options,
            [int(standard.resolution[0]), int(standard.resolution[1]), 0, 0],
            [
                float(profile_catalog.near_clip_m),
                float(profile_catalog.far_clip_m),
                math.radians(float(standard.perspective_angle_deg)),
                0.02, 0.0, 0.0, 0.08, 0.08, 0.10, 0.0, 0.0,
            ],
        )
    )
    _alias(sim, camera, camera_spec["alias"])
    sim.setObjectParent(camera, camera_rig, False)
    sim.setObjectPosition(camera, [0.0, 0.0, 0.0], camera_rig)
    sim.setObjectOrientation(
        camera,
        [math.radians(float(value)) for value in camera_spec["orientation_deg"]],
        camera_rig,
    )
    assets = {entry["asset_id"]: entry for entry in code_assets["entries"]}
    if set(assets) != {part["code_asset_id"] for part in spec["parts"]}:
        raise ValueError("code assets and scene parts do not match")
    for part in spec["parts"]:
        _code_part(sim, part, assets[part["code_asset_id"]], parts_group)
    for bin_specification in spec["bins"]:
        _build_bin(sim, bin_specification, bins_group)
    _build_vision_lighting(sim, profile_catalog, root)


def _ocr_label_bitmap(label: np.ndarray) -> np.ndarray:
    """Recover the original two 5x7 glyphs from one fixed OCR label."""

    if not isinstance(label, np.ndarray) or label.dtype != np.uint8 or label.ndim != 2:
        raise ValueError("OCR label must be an 8-bit grayscale image")
    if label.shape[0] < 65 or label.shape[1] < 56:
        raise ValueError("OCR label is smaller than the fixed 5x7 source region")
    bitmap = np.full((7, 10), 255, dtype=np.uint8)
    for glyph_index, origin_x in enumerate((3, 31)):
        source = label[30:65, origin_x : origin_x + 25]
        blocks = source.reshape(7, 5, 5, 5)
        bitmap[:, glyph_index * 5 : (glyph_index + 1) * 5] = np.min(
            blocks, axis=(1, 3)
        )
    return bitmap


def _ocr_bitmap_geometry(*, face_width: float, face_height: float) -> dict[str, float]:
    """Return a gap-free pitch and cell size for the rendered 10x7 bitmap."""

    # Keep the complete 10x7 label inside the published inner ROIs while
    # leaving enough white border for the host-side segmentation contract.
    pitch_width = float(face_width) / 28.0
    # The OCR training generator renders square bitmap pixels.  Keep the
    # scene cells square in world units so the perspective camera does not
    # create a vertically stretched glyph before kernel normalization.
    pitch_height = float(face_height) / 28.0
    # Match the generator's one-pixel dilation variant.  The overlap survives
    # CoppeliaSim rasterisation at cell boundaries, including diagonal
    # 8-connected bitmap strokes, without reaching the face plate edge.
    fill_ratio = 1.18
    return {
        "pitch_width_mm": pitch_width,
        "pitch_height_mm": pitch_height,
        "cell_width_mm": pitch_width * fill_ratio,
        "cell_height_mm": pitch_height * fill_ratio,
        "glyph_gap_pitch_mm": pitch_width,
    }


def _ocr_part(
    sim: Any,
    part: dict[str, Any],
    label_path: Path,
    parent: int,
) -> int:
    """Build one OCR pickable with a deterministic bitmap label face."""
    center = [float(value) for value in part["position_mm"]]
    size = [float(value) for value in part["size_mm"]]
    pieces = [
        _shape(
            sim,
            name=f"{part['alias']}_body",
            shape="cuboid",
            size_mm=size,
            position_mm=center,
            color=[0.74, 0.75, 0.77],
            parent=parent,
            respondable=True,
        )
    ]
    face_width = size[0] - 4.0
    face_height = size[1] - 4.0
    face_z = center[2] + size[2] / 2.0 + 0.6
    pieces.append(
        _shape(
            sim,
            name=f"{part['alias']}_face_plate",
            shape="cuboid",
            size_mm=[face_width, face_height, 1.0],
            position_mm=[center[0], center[1], face_z],
            color=[0.98, 0.98, 0.98],
            parent=parent,
            respondable=False,
        )
    )
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if label is None or label.size == 0:
        raise ValueError(f"could not decode OCR scene label: {label_path}")
    # The original label is generated from two 5x7 glyphs.  Rendering a
    # compact 10x7 bitmap keeps the scene small while preserving its meaning.
    bitmap = _ocr_label_bitmap(label)
    geometry = _ocr_bitmap_geometry(face_width=face_width, face_height=face_height)
    pitch_width = geometry["pitch_width_mm"]
    pitch_height = geometry["pitch_height_mm"]
    cell_width = geometry["cell_width_mm"]
    cell_height = geometry["cell_height_mm"]
    origin_x = center[0] - 5.5 * pitch_width
    origin_y = center[1] - 3.5 * pitch_height
    bar_index = 0
    for row in range(7):
        for column in range(10):
            if int(bitmap[row, column]) >= 160:
                continue
            logical_column = column + (1 if column >= 5 else 0)
            render_column = 10 - logical_column
            pieces.append(
                _shape(
                    sim,
                    name=f"{part['alias']}_glyph_{bar_index:03d}",
                    shape="cuboid",
                    size_mm=[cell_width, cell_height, 0.3],
                    position_mm=[
                        origin_x
                        + (render_column + 0.5)
                        * pitch_width,
                        origin_y + (row + 0.5) * pitch_height,
                        face_z + 0.65,
                    ],
                    color=[0.01, 0.01, 0.01],
                    parent=parent,
                    respondable=False,
                )
            )
            bar_index += 1
    compound = int(sim.groupShapes(pieces, False))
    _alias(sim, compound, part["alias"])
    sim.setObjectParent(compound, parent, True)
    relocate_frame = getattr(sim, "relocateShapeFrame", None)
    if callable(relocate_frame):
        relocate_frame(
            compound,
            [
                center[0] / 1000.0,
                center[1] / 1000.0,
                center[2] / 1000.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        )
    _set_int_parameter(sim, compound, sim.shapeintparam_static, 1)
    _set_int_parameter(sim, compound, sim.shapeintparam_respondable, 1)
    _dummy(sim, "CodeFace", compound)
    return compound


def _build_ocr_sorting(
    sim: Any,
    spec: dict[str, Any],
    root: int,
    *,
    ocr_assets: dict[str, Any],
    profile_catalog: Any,
    assets_root: Path,
) -> None:
    _build_workspace(sim, spec["workspace"], root)
    parts_group = _dummy(sim, "Parts", root)
    routes_group = _dummy(sim, "Routes", root)
    camera_rig = _dummy(sim, "CameraRig", root)
    standard = profile_catalog.require(profile_catalog.baseline_profile_id)
    camera_spec = spec["camera"]
    sim.setObjectPosition(
        camera_rig,
        [float(value) for value in camera_spec["rig_position_m"]],
        sim.handle_world,
    )
    options = 1 | 2 | 4 | 64 | 128
    camera = int(
        sim.createVisionSensor(
            options,
            [int(standard.resolution[0]), int(standard.resolution[1]), 0, 0],
            [
                float(profile_catalog.near_clip_m),
                float(profile_catalog.far_clip_m),
                math.radians(float(standard.perspective_angle_deg)),
                0.02,
                0.0,
                0.0,
                0.08,
                0.08,
                0.10,
                0.0,
                0.0,
            ],
        )
    )
    _alias(sim, camera, camera_spec["alias"])
    sim.setObjectParent(camera, camera_rig, False)
    sim.setObjectPosition(camera, [0.0, 0.0, 0.0], camera_rig)
    sim.setObjectOrientation(
        camera,
        [math.radians(float(value)) for value in camera_spec["orientation_deg"]],
        camera_rig,
    )
    labels = {
        item["identifier"]: item
        for item in ocr_assets.get("labels", [])
        if isinstance(item, dict)
    }
    if set(labels) != {part["label_asset_id"] for part in spec["parts"]}:
        raise ValueError("OCR asset labels and scene parts do not match")
    for part in spec["parts"]:
        label_path = assets_root / labels[part["label_asset_id"]]["path"]
        _ocr_part(sim, part, label_path, parts_group)
    for route in spec["routes"]:
        _build_bin(sim, route, routes_group)
    _build_vision_lighting(sim, profile_catalog, root)


def _defect_surface_part(
    sim: Any,
    part: dict[str, Any],
    asset_path: Path,
    parent: int,
    *,
    group_alias: str | None = None,
    inspection_face_direct: bool = False,
    respondable: bool,
) -> int:
    """Build one identical pickable body with a deterministic surface face."""

    center = [float(value) for value in part["position_mm"]]
    size = [float(value) for value in part["size_mm"]]
    pieces = [
        _shape(
            sim,
            name=f"{part['alias']}_body",
            shape="cuboid",
            size_mm=size,
            position_mm=center,
            color=[0.74, 0.75, 0.77],
            parent=parent,
            respondable=respondable,
        )
    ]
    face_width = size[0] - 4.0
    face_height = size[1] - 4.0
    face_z = center[2] + size[2] / 2.0 + 0.6
    pieces.append(
        _shape(
            sim,
            name=f"{part['alias']}_face_plate",
            shape="cuboid",
            size_mm=[face_width, face_height, 1.0],
            position_mm=[center[0], center[1], face_z],
            color=[0.98, 0.98, 0.98],
            parent=parent,
            respondable=False,
        )
    )
    image = cv2.imread(str(asset_path), cv2.IMREAD_GRAYSCALE)
    bitmap = _rasterize_defect_surface(image, asset_path=asset_path)
    pitch_x = face_width / 20.0
    pitch_y = face_height / 20.0
    origin_x = center[0] - face_width / 2.0
    origin_y = center[1] - face_height / 2.0
    pixel_index = 0
    for row in range(20):
        for column in range(20):
            if int(bitmap[row, column]) >= 160:
                continue
            pieces.append(
                _shape(
                    sim,
                    name=f"{part['alias']}_surface_{pixel_index:03d}",
                    shape="cuboid",
                    size_mm=[pitch_x * 0.90, pitch_y * 0.90, 0.3],
                    position_mm=[
                        origin_x + pitch_x * (column + 0.5),
                        origin_y + pitch_y * (row + 0.5),
                        face_z + 0.65,
                    ],
                    color=[0.01, 0.01, 0.01],
                    parent=parent,
                    respondable=False,
                )
            )
            pixel_index += 1
    compound = int(sim.groupShapes(pieces, False))
    _alias(sim, compound, "InspectionFace" if inspection_face_direct else group_alias or part["alias"])
    sim.setObjectParent(compound, parent, True)
    # CoppeliaSim chooses a compound's local origin from its merged geometry.
    # Bind the public pickable alias back to the declared scene center so the
    # route/probe contract and the visual body use the same coordinate frame.
    sim.setObjectPosition(
        compound,
        [value / 1000.0 for value in center],
        getattr(sim, "handle_world", -1),
    )
    _set_int_parameter(sim, compound, sim.shapeintparam_static, 1)
    _set_int_parameter(sim, compound, sim.shapeintparam_respondable, int(respondable))
    if not inspection_face_direct:
        _dummy(sim, "InspectionFace", compound)
    return compound


def _rasterize_defect_surface(
    image: np.ndarray | None,
    *,
    asset_path: Path,
    raster_size: int = 20,
) -> np.ndarray:
    if image is None or image.shape != (256, 256) or image.dtype != np.uint8:
        raise ValueError(f"could not decode V1-09 defect surface asset: {asset_path}")
    return cv2.resize(image, (raster_size, raster_size), interpolation=cv2.INTER_NEAREST)


def _defect_slot(sim: Any, slot: dict[str, Any], parent: int) -> None:
    group = _dummy(sim, slot["alias"], parent)
    center = [float(value) for value in slot["position_mm"]]
    color = [float(value) for value in slot["color_rgb"]]
    _shape(
        sim,
        name="Floor",
        shape="cuboid",
        size_mm=[30, 30, 2],
        position_mm=[center[0], center[1], 11],
        color=color,
        parent=group,
        respondable=True,
    )
    for name, dx, dy, sx, sy in (
        ("WallLeft", -16, 0, 2, 34),
        ("WallRight", 16, 0, 2, 34),
        ("WallFront", 0, -16, 34, 2),
        ("WallBack", 0, 16, 34, 2),
    ):
        _shape(
            sim,
            name=name,
            shape="cuboid",
            size_mm=[sx, sy, 12],
            position_mm=[center[0] + dx, center[1] + dy, 17],
            color=color,
            parent=group,
            respondable=True,
        )
    marker = _dummy(sim, "Target", group)
    sim.setObjectPosition(marker, [center[0] / 1000.0, center[1] / 1000.0, center[2] / 1000.0], sim.handle_world)


def _build_defect_sorting(
    sim: Any,
    spec: dict[str, Any],
    root: int,
    *,
    defect_assets: dict[str, Any],
    profile_catalog: Any,
    assets_root: Path,
) -> None:
    _build_workspace(sim, spec["workspace"], root)
    reference_group = _dummy(sim, "Reference", root)
    parts_group = _dummy(sim, "Parts", root)
    slots_group = _dummy(sim, "Slots", root)
    camera_rig = _dummy(sim, "CameraRig", root)
    standard = profile_catalog.require(profile_catalog.baseline_profile_id)
    camera_spec = spec["camera"]
    sim.setObjectPosition(camera_rig, [float(value) for value in camera_spec["rig_position_m"]], sim.handle_world)
    options = 1 | 2 | 4 | 64 | 128
    camera = int(
        sim.createVisionSensor(
            options,
            [int(standard.resolution[0]), int(standard.resolution[1]), 0, 0],
            [
                float(profile_catalog.near_clip_m),
                float(profile_catalog.far_clip_m),
                math.radians(float(standard.perspective_angle_deg)),
                0.02,
                0.0,
                0.0,
                0.08,
                0.08,
                0.10,
                0.0,
                0.0,
            ],
        )
    )
    _alias(sim, camera, camera_spec["alias"])
    sim.setObjectParent(camera, camera_rig, False)
    sim.setObjectPosition(camera, [0.0, 0.0, 0.0], camera_rig)
    sim.setObjectOrientation(camera, [math.radians(float(value)) for value in camera_spec["orientation_deg"]], camera_rig)
    assets = {item["asset_id"]: item for item in defect_assets["assets"]}
    reference = spec["reference"]
    _defect_surface_part(
        sim,
        reference,
        assets_root / assets["reference"]["path"],
        reference_group,
        inspection_face_direct=True,
        respondable=False,
    )
    for part in spec["parts"]:
        asset = assets.get(part["asset_id"])
        if asset is None:
            raise ValueError("V1-09 defect asset IDs and scene parts do not match")
        _defect_surface_part(sim, part, assets_root / asset["path"], parts_group, respondable=True)
    for slot in spec["slots"]:
        _defect_slot(sim, slot, slots_group)
    _build_vision_lighting(sim, profile_catalog, root)


def _attach_ocr_camera_scope(sim: Any, root: int) -> int:
    return _attach_camera_scope(
        sim,
        root,
        scene_root_path="/VisionOcrSortingLab",
        camera_path="/VisionOcrSortingLab/CameraRig/Camera",
    )


def _attach_defect_camera_scope(sim: Any, root: int) -> int:
    return _attach_camera_scope(
        sim,
        root,
        scene_root_path="/VisionDefectSortingLab",
        camera_path="/VisionDefectSortingLab/CameraRig/Camera",
    )


def _attach_camera_scope(
    sim: Any,
    root: int,
    *,
    scene_root_path: str,
    camera_path: str,
) -> int:
    allowed = {
        ("/LogisticsLab", "/LogisticsLab/Camera"),
        ("/VisionCodeRoutingLab", "/VisionCodeRoutingLab/CameraRig/Camera"),
        ("/VisionOcrSortingLab", "/VisionOcrSortingLab/CameraRig/Camera"),
        ("/VisionDefectSortingLab", "/VisionDefectSortingLab/CameraRig/Camera"),
    }
    if (scene_root_path, camera_path) not in allowed:
        raise ValueError("camera render scope is not an approved formal scene")
    script_text = f"""
function sysCall_init()
    trainingCollection = sim.createCollection(1)
    sim.addItemToCollection(
        trainingCollection,
        sim.handle_tree,
        sim.getObject('{scene_root_path}'),
        0
    )
    local camera = sim.getObject('{camera_path}')
    sim.setObjectInt32Param(
        camera,
        sim.visionintparam_entity_to_render,
        trainingCollection
    )
end

function sysCall_cleanup()
    if trainingCollection then
        sim.destroyCollection(trainingCollection)
        trainingCollection = nil
    end
end
""".strip()
    handle = int(sim.createScript(sim.scripttype_simulation, script_text, 0, "lua"))
    _alias(sim, handle, "CameraRenderScope")
    sim.setObjectParent(handle, root, True)
    return handle


def _attach_logistics_camera_scope(sim: Any, root: int) -> int:
    return _attach_camera_scope(
        sim,
        root,
        scene_root_path="/LogisticsLab",
        camera_path="/LogisticsLab/Camera",
    )


def _attach_code_routing_camera_scope(sim: Any, root: int) -> int:
    return _attach_camera_scope(
        sim,
        root,
        scene_root_path="/VisionCodeRoutingLab",
        camera_path="/VisionCodeRoutingLab/CameraRig/Camera",
    )


def _stop_simulation(sim: Any) -> None:
    if int(sim.getSimulationState()) == int(sim.simulation_stopped):
        return
    sim.stopSimulation()
    deadline = time.monotonic() + 10.0
    while (
        int(sim.getSimulationState()) != int(sim.simulation_stopped)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if int(sim.getSimulationState()) != int(sim.simulation_stopped):
        raise RuntimeError("CoppeliaSim did not stop before scene build")


def _recoverable_release(
    output: Path,
    manifest_path: Path,
    formal: _FormalScene,
    template_hash: str,
    profile_catalog: tuple[str, Path, str] | None = None,
    code_assets_manifest: tuple[str, Path, str] | None = None,
    ocr_assets_manifest: tuple[str, Path, str] | None = None,
    defect_assets_manifest: tuple[str, Path, str] | None = None,
) -> _RecoverableRelease | None:
    if not output.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = _load(manifest_path)
        template = manifest.get("template")
        scene = manifest.get("scene")
        stored_profile = manifest.get("profile_catalog")
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("scene_id") != formal.scene_id
            or manifest.get("protected_assets_unchanged") is not True
            or manifest.get("required_paths")
            != list(formal.required_paths)
            or not isinstance(template, dict)
            or template.get("path") != TEMPLATE_RELATIVE
            or template.get("sha256") != template_hash
            or not isinstance(scene, dict)
            or scene.get("path") != formal.output_relative
        ):
            return None
        if formal.scene_id in {
            "vision-quality-lab",
            "vision-code-routing-lab",
            "vision-ocr-sorting-lab",
            "vision-defect-sorting-lab",
        }:
            if profile_catalog is None:
                return None
            profile_relative, profile_path, profile_hash = profile_catalog
            if (
                not isinstance(stored_profile, dict)
                or set(stored_profile) != {"path", "sha256"}
                or stored_profile.get("path") != profile_relative
                or stored_profile.get("sha256") != profile_hash
                or _sha256(profile_path) != profile_hash
            ):
                return None
            if formal.scene_id == "vision-code-routing-lab":
                if code_assets_manifest is None:
                    return None
                asset_relative, asset_path, asset_hash = code_assets_manifest
                if (
                    not isinstance(manifest.get("code_assets_manifest"), dict)
                    or set(manifest["code_assets_manifest"]) != {"path", "sha256"}
                    or manifest["code_assets_manifest"].get("path") != asset_relative
                    or manifest["code_assets_manifest"].get("sha256") != asset_hash
                    or _sha256(asset_path) != asset_hash
                ):
                    return None
            if formal.scene_id == "vision-ocr-sorting-lab":
                if ocr_assets_manifest is None:
                    return None
                asset_relative, asset_path, asset_hash = ocr_assets_manifest
                if (
                    not isinstance(manifest.get("ocr_assets_manifest"), dict)
                    or set(manifest["ocr_assets_manifest"]) != {"path", "sha256"}
                    or manifest["ocr_assets_manifest"].get("path") != asset_relative
                    or manifest["ocr_assets_manifest"].get("sha256") != asset_hash
                    or _sha256(asset_path) != asset_hash
                ):
                    return None
            if formal.scene_id == "vision-defect-sorting-lab":
                if defect_assets_manifest is None:
                    return None
                asset_relative, asset_path, asset_hash = defect_assets_manifest
                if (
                    not isinstance(manifest.get("defect_assets_manifest"), dict)
                    or set(manifest["defect_assets_manifest"]) != {"path", "sha256"}
                    or manifest["defect_assets_manifest"].get("path") != asset_relative
                    or manifest["defect_assets_manifest"].get("sha256") != asset_hash
                    or _sha256(asset_path) != asset_hash
                ):
                    return None
        sha256 = scene.get("sha256")
        size = scene.get("size_bytes")
        if (
            not isinstance(sha256, str)
            or _LOWER_SHA256.fullmatch(sha256) is None
            or type(size) is not int
            or size <= 0
        ):
            return None
        release = _RecoverableRelease(
            scene_sha256=sha256,
            scene_size=size,
        )
        if not _release_still_matches(output, release):
            return None
        return release
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _release_still_matches(
    output: Path,
    release: _RecoverableRelease,
) -> bool:
    try:
        return (
            output.is_file()
            and output.stat().st_size == release.scene_size
            and _sha256(output) == release.scene_sha256
        )
    except OSError:
        return False


def _cleanup_diagnostic(action: str, exc: BaseException) -> str:
    return f"{action}: {type(exc).__name__}: {exc}"


def _note_cleanup_errors(
    primary: BaseException,
    cleanup_errors: list[str],
) -> None:
    add_note = getattr(primary, "add_note", None)
    if not callable(add_note):
        return
    for diagnostic in cleanup_errors:
        add_note(f"cleanup error: {diagnostic}")


def build_scene(
    *,
    spec_path: Path,
    host: str,
    port: int,
) -> dict[str, Any]:
    spec_path, formal = _formal_spec_path(spec_path)
    spec = _load(spec_path)
    _validate_spec(spec, formal)
    if not isinstance(host, str) or not host.strip() or host != host.strip():
        raise ValueError("host must be a non-empty string")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")

    template = _project_file_from_relative(
        TEMPLATE_RELATIVE,
        label="template",
        must_exist=True,
    )
    output_candidate = _lexical_project_path(
        formal.output_relative,
        label="output",
    )
    output = _project_file_from_relative(
        formal.output_relative,
        label="output",
        must_exist=os.path.lexists(output_candidate),
    )
    manifest_path = spec_path.parent / "scene_manifest.json"
    _assert_no_alias(
        manifest_path.parent,
        label="manifest parent",
        must_exist=True,
    )
    _assert_no_alias(
        manifest_path,
        label="manifest",
        must_exist=os.path.lexists(manifest_path),
    )
    if output.exists() and output.samefile(template):
        raise ValueError("output must not be a template or protected alias")

    build_lock = _BuildOutputLock(output)
    build_lock.acquire()
    client = None
    temporaries: list[Path] = []
    cleanup_errors: list[str] = []
    primary_error: BaseException | None = None
    primary_traceback = None
    result: dict[str, Any] | None = None
    try:
        before = _protected_hashes()
        template_hash = _sha256(template)
        profile_catalog_path: Path | None = None
        profile_catalog_content: bytes | None = None
        profile_catalog_hash: str | None = None
        profile_catalog = None
        code_assets_path: Path | None = None
        code_assets_content: bytes | None = None
        code_assets_hash: str | None = None
        code_assets = None
        ocr_assets_path: Path | None = None
        ocr_assets_content: bytes | None = None
        ocr_assets_hash: str | None = None
        ocr_assets = None
        defect_assets_path: Path | None = None
        defect_assets_content: bytes | None = None
        defect_assets_hash: str | None = None
        defect_assets = None
        if formal.scene_id in {
            "vision-quality-lab",
            "vision-code-routing-lab",
            "vision-ocr-sorting-lab",
            "vision-defect-sorting-lab",
        }:
            profile_catalog_path = _project_file_from_relative(
                spec["profiles"],
                label="profiles",
                must_exist=True,
            )
            profile_catalog_content = profile_catalog_path.read_bytes()
            profile_catalog_hash = hashlib.sha256(
                profile_catalog_content
            ).hexdigest()
            if formal.scene_id == "vision-ocr-sorting-lab":
                profile_catalog = _load_ocr_profile_catalog(profile_catalog_path)
            elif formal.scene_id == "vision-defect-sorting-lab":
                profile_catalog = _load_defect_profile_catalog(profile_catalog_path)
            else:
                profile_catalog = load_profile_catalog_bytes(
                    profile_catalog_content
                )
        if formal.scene_id == "vision-code-routing-lab":
            code_assets_path = _project_file_from_relative(
                spec["code_assets_manifest"],
                label="code_assets_manifest",
                must_exist=True,
            )
            code_assets_content = code_assets_path.read_bytes()
            code_assets_hash = hashlib.sha256(code_assets_content).hexdigest()
            code_assets = json.loads(code_assets_content.decode("utf-8"))
        if formal.scene_id == "vision-ocr-sorting-lab":
            ocr_assets_path = _project_file_from_relative(
                spec["ocr_assets_manifest"],
                label="ocr_assets_manifest",
                must_exist=True,
            )
            ocr_assets_content = ocr_assets_path.read_bytes()
            ocr_assets_hash = hashlib.sha256(ocr_assets_content).hexdigest()
            ocr_assets = json.loads(ocr_assets_content.decode("utf-8"))
        if formal.scene_id == "vision-defect-sorting-lab":
            defect_assets_path = _project_file_from_relative(
                spec["defect_assets_manifest"],
                label="defect_assets_manifest",
                must_exist=True,
            )
            defect_assets_content = defect_assets_path.read_bytes()
            defect_assets_hash = hashlib.sha256(defect_assets_content).hexdigest()
            defect_assets = _validate_defect_assets_manifest(defect_assets_path)
        token = uuid.uuid4().hex
        staged_scene = output.parent / (
            f".{output.stem}.staged-{token}.ttt"
        )
        staged_manifest = output.parent / (
            f".scene_manifest.staged-{token}.json"
        )
        backup_manifest = output.parent / (
            f".scene_manifest.backup-{token}.json"
        )
        candidates = [staged_scene, staged_manifest, backup_manifest]
        if any(os.path.lexists(path) for path in candidates):
            raise RuntimeError("unique scene staging path already exists")
        temporaries.extend(candidates)

        client = RemoteAPIClient(host=host, port=port)
        sim = client.require("sim")
        _stop_simulation(sim)
        sim.loadScene(stage_scene_for_coppeliasim(template).as_posix())
        for path in spec["remove_paths"]:
            handle = int(sim.getObject(path))
            sim.removeObjects([handle], False)
        root_name = formal.root_path.rsplit("/", 1)[-1]
        root = _dummy(sim, root_name)
        if formal.scene_id == "robot-basics":
            _build_workspace(sim, spec["workspace"], root)
            _camera(sim, spec["camera"], root)
            _build_basics(sim, spec, root)
        elif formal.scene_id == "logistics-lab":
            _build_workspace(sim, spec["workspace"], root)
            _camera(sim, spec["camera"], root)
            _build_logistics(sim, spec, root)
            _attach_logistics_camera_scope(sim, root)
        elif formal.scene_id == "vision-quality-lab":
            _build_vision_quality(
                sim,
                spec,
                root,
                profile_catalog=profile_catalog,
            )
        elif formal.scene_id == "vision-code-routing-lab":
            _build_code_routing(
                sim,
                spec,
                root,
                code_assets=code_assets,
                profile_catalog=profile_catalog,
            )
            _attach_code_routing_camera_scope(sim, root)
        elif formal.scene_id == "vision-ocr-sorting-lab":
            if ocr_assets_path is None or ocr_assets is None:
                raise RuntimeError("OCR assets manifest was not loaded")
            _build_ocr_sorting(
                sim,
                spec,
                root,
                ocr_assets=ocr_assets,
                profile_catalog=profile_catalog,
                assets_root=ocr_assets_path.parent,
            )
            _attach_ocr_camera_scope(sim, root)
        elif formal.scene_id == "vision-defect-sorting-lab":
            if defect_assets_path is None or defect_assets is None:
                raise RuntimeError("V1-09 defect assets manifest was not loaded")
            _build_defect_sorting(
                sim,
                spec,
                root,
                defect_assets=defect_assets,
                profile_catalog=profile_catalog,
                assets_root=defect_assets_path.parent,
            )
            _attach_defect_camera_scope(sim, root)
        else:
            raise RuntimeError(f"unsupported formal scene: {formal.scene_id}")
        for path in formal.required_paths:
            sim.getObject(path)

        sim.saveScene(staged_scene.as_posix())
        if not staged_scene.is_file() or staged_scene.stat().st_size == 0:
            raise RuntimeError(
                f"CoppeliaSim did not write staged scene: {staged_scene}"
            )
        _assert_no_alias(
            staged_scene,
            label="staged scene",
            must_exist=True,
        )
        scene_hash = _sha256(staged_scene)
        scene_size = staged_scene.stat().st_size
        after = _protected_hashes()
        if after != before or _sha256(template) != template_hash:
            raise RuntimeError("protected assets changed during scene build")
        if (
            profile_catalog_path is not None
            and profile_catalog_content is not None
            and profile_catalog_path.read_bytes() != profile_catalog_content
        ):
            raise RuntimeError("vision profile catalog changed during scene build")
        if (
            code_assets_path is not None
            and code_assets_content is not None
            and code_assets_path.read_bytes() != code_assets_content
        ):
            raise RuntimeError("code assets manifest changed during scene build")
        if (
            ocr_assets_path is not None
            and ocr_assets_content is not None
            and ocr_assets_path.read_bytes() != ocr_assets_content
        ):
            raise RuntimeError("OCR assets manifest changed during scene build")
        if (
            defect_assets_path is not None
            and defect_assets_content is not None
            and defect_assets_path.read_bytes() != defect_assets_content
        ):
            raise RuntimeError("V1-09 defect assets manifest changed during scene build")

        manifest = {
            "schema_version": 1,
            "scene_id": formal.scene_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "simulation.training_scenes.build_scene",
            "template": {
                "path": TEMPLATE_RELATIVE,
                "sha256": template_hash,
            },
            "scene": {
                "path": formal.output_relative,
                "sha256": scene_hash,
                "size_bytes": scene_size,
            },
            "required_paths": list(formal.required_paths),
            "task_contracts": spec.get("tasks", {}),
            "protected_assets_unchanged": True,
        }
        if profile_catalog_path is not None:
            manifest["profile_catalog"] = {
                "path": spec["profiles"],
                "sha256": profile_catalog_hash,
            }
        if formal.scene_id == "vision-code-routing-lab":
            if not isinstance(code_assets_content, bytes) or code_assets_hash is None:
                raise RuntimeError("code assets manifest was not loaded")
            standard = profile_catalog.require(profile_catalog.baseline_profile_id)
            plane_z_mm = float(spec["camera"]["code_face_plane_z_mm"])
            distance_mm = float(standard.camera_rig_z_m) * 1000.0 - plane_z_mm
            scale_mm_per_px = (
                2.0
                * distance_mm
                * math.tan(math.radians(float(standard.perspective_angle_deg)) / 2.0)
                / float(standard.resolution[0])
            )
            pixel_center = (float(standard.resolution[0]) - 1.0) / 2.0
            rig_x_mm = float(spec["camera"]["rig_position_m"][0]) * 1000.0
            rig_y_mm = float(spec["camera"]["rig_position_m"][1]) * 1000.0
            manifest["code_assets_manifest"] = {
                "path": Path(spec["code_assets_manifest"]).name,
                "sha256": code_assets_hash,
            }
            manifest["code_routing"] = {
                "part_ids": [part["alias"] for part in spec["parts"]],
                "initial_positions_mm": {
                    part["alias"]: list(part["position_mm"]) for part in spec["parts"]
                },
                "calibration_plane_z_mm": plane_z_mm,
                "calibration_matrix": [
                    [-scale_mm_per_px, 0.0, rig_x_mm + scale_mm_per_px * pixel_center],
                    [0.0, scale_mm_per_px, rig_y_mm - scale_mm_per_px * pixel_center],
                ],
                "route_slots_mm": {
                    bin_specification["alias"]: [
                        list(slot["position_mm"])
                        for slot in bin_specification["slots"]
                    ]
                    for bin_specification in spec["bins"]
                },
            }
            manifest["reset_contract"] = dict(spec["reset_contract"])
        if formal.scene_id == "vision-ocr-sorting-lab":
            if ocr_assets_hash is None:
                raise RuntimeError("OCR assets manifest hash was not loaded")
            standard = profile_catalog.require(profile_catalog.baseline_profile_id)
            plane_z_mm = float(spec["camera"]["code_face_plane_z_mm"])
            distance_mm = float(standard.camera_rig_z_m) * 1000.0 - plane_z_mm
            scale_mm_per_px = (
                2.0
                * distance_mm
                * math.tan(math.radians(float(standard.perspective_angle_deg)) / 2.0)
                / float(standard.resolution[0])
            )
            pixel_center = (float(standard.resolution[0]) - 1.0) / 2.0
            rig_x_mm = float(spec["camera"]["rig_position_m"][0]) * 1000.0
            rig_y_mm = float(spec["camera"]["rig_position_m"][1]) * 1000.0
            manifest["ocr_assets_manifest"] = {
                "path": spec["ocr_assets_manifest"],
                "sha256": ocr_assets_hash,
            }
            manifest["ocr_sorting"] = {
                "part_ids": [part["alias"] for part in spec["parts"]],
                "identifiers": {
                    part["alias"]: part["identifier"] for part in spec["parts"]
                },
                "initial_positions_mm": {
                    part["alias"]: list(part["position_mm"]) for part in spec["parts"]
                },
                "calibration_plane_z_mm": plane_z_mm,
                "calibration_matrix": [
                    [-scale_mm_per_px, 0.0, rig_x_mm + scale_mm_per_px * pixel_center],
                    [0.0, scale_mm_per_px, rig_y_mm - scale_mm_per_px * pixel_center],
                ],
                "route_slots_mm": {
                    route["alias"]: [
                        list(slot["position_mm"]) for slot in route["slots"]
                    ]
                    for route in spec["routes"]
                },
                "safe_z_mm": float(spec["safe_z_mm"]),
            }
            manifest["reset_contract"] = dict(spec["reset_contract"])
        if formal.scene_id == "vision-defect-sorting-lab":
            if defect_assets_hash is None:
                raise RuntimeError("V1-09 defect assets manifest hash was not loaded")
            standard = profile_catalog.require(profile_catalog.baseline_profile_id)
            plane_z_mm = float(spec["camera"]["surface_face_plane_z_mm"])
            distance_mm = float(standard.camera_rig_z_m) * 1000.0 - plane_z_mm
            scale_mm_per_px = (
                2.0
                * distance_mm
                * math.tan(math.radians(float(standard.perspective_angle_deg)) / 2.0)
                / float(standard.resolution[0])
            )
            pixel_center = (float(standard.resolution[0]) - 1.0) / 2.0
            rig_x_mm = float(spec["camera"]["rig_position_m"][0]) * 1000.0
            rig_y_mm = float(spec["camera"]["rig_position_m"][1]) * 1000.0
            manifest["defect_assets_manifest"] = {
                "path": spec["defect_assets_manifest"],
                "sha256": defect_assets_hash,
            }
            manifest["defect_sorting"] = {
                "part_ids": [part["alias"] for part in spec["parts"]],
                "reference_position_mm": list(spec["reference"]["position_mm"]),
                "initial_positions_mm": {
                    part["alias"]: list(part["position_mm"]) for part in spec["parts"]
                },
                "calibration_plane_z_mm": plane_z_mm,
                "calibration_matrix": [
                    [-scale_mm_per_px, 0.0, rig_x_mm + scale_mm_per_px * pixel_center],
                    [0.0, scale_mm_per_px, rig_y_mm - scale_mm_per_px * pixel_center],
                ],
                "rois": {key: list(value) for key, value in spec["rois"].items()},
                "slot_positions_mm": {
                    slot["decision"]: list(slot["position_mm"]) for slot in spec["slots"]
                },
                "safe_z_mm": float(spec["safe_z_mm"]),
            }
            manifest["reset_contract"] = dict(spec["reset_contract"])
        _write_exclusive(staged_manifest, manifest)

        old_release = _recoverable_release(
            output,
            manifest_path,
            formal,
            template_hash,
            (
                (spec["profiles"], profile_catalog_path, profile_catalog_hash)
                if profile_catalog_path is not None
                and profile_catalog_hash is not None
                else None
            ),
            (
                (Path(spec["code_assets_manifest"]).name, code_assets_path, code_assets_hash)
                if code_assets_path is not None and code_assets_hash is not None
                else None
            ),
            (
                (spec["ocr_assets_manifest"], ocr_assets_path, ocr_assets_hash)
                if ocr_assets_path is not None and ocr_assets_hash is not None
                else None
            ),
            (
                (spec["defect_assets_manifest"], defect_assets_path, defect_assets_hash)
                if defect_assets_path is not None and defect_assets_hash is not None
                else None
            ),
        )
        if os.path.lexists(manifest_path):
            os.replace(manifest_path, backup_manifest)
        try:
            os.replace(staged_scene, output)
        except BaseException as exc:
            if (
                old_release is not None
                and os.path.lexists(backup_manifest)
                and _release_still_matches(output, old_release)
            ):
                try:
                    os.replace(backup_manifest, manifest_path)
                except BaseException as restore_exc:
                    cleanup_errors.append(
                        _cleanup_diagnostic(
                            "old manifest restore failed",
                            restore_exc,
                        )
                    )
            raise
        if (
            not output.is_file()
            or output.stat().st_size != scene_size
            or _sha256(output) != scene_hash
        ):
            raise RuntimeError("published training scene verification failed")
        if _protected_hashes() != before:
            raise RuntimeError("protected assets changed during scene publish")
        os.replace(staged_manifest, manifest_path)
        result = manifest
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    cleanup_errors.append(
                        _cleanup_diagnostic(
                            "client.close cleanup failed",
                            exc,
                        )
                    )
        for temporary in temporaries:
            try:
                if os.path.lexists(temporary):
                    temporary.unlink()
            except BaseException as exc:
                cleanup_errors.append(
                    _cleanup_diagnostic(
                        f"temporary cleanup failed for {temporary.name}",
                        exc,
                    )
                )
        cleanup_errors.extend(build_lock.release())

    if primary_error is not None:
        _note_cleanup_errors(primary_error, cleanup_errors)
        raise primary_error.with_traceback(primary_traceback)
    if result is None:
        raise RuntimeError("scene build ended without a result")
    if cleanup_errors:
        response = dict(result)
        response["cleanup_errors"] = list(cleanup_errors)
        warning_message = (
            "scene build committed with cleanup_errors: "
            + "; ".join(cleanup_errors)
        )
        try:
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
        except Warning:
            pass
        return response
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    args = parser.parse_args(argv)
    manifest = build_scene(
        spec_path=args.spec,
        host=args.host,
        port=args.port,
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
