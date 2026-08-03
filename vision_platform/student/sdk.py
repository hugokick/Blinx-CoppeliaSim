from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count
from math import isfinite
import re
from types import MappingProxyType
from typing import Any, Mapping

import cv2
import numpy as np

from vision_platform.student.protocol import CommandMessage, ResponseMessage


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PROFILE_ID = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
_SNAPSHOT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_TEMPLATE_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_TEMPLATE_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\Z")
_VISION_BUNDLE_NAME = re.compile(
    r"vision-bundle-[A-Za-z0-9][A-Za-z0-9_.-]*\.json\Z"
)
_VISION_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "resolution",
        "perspective_angle_deg",
        "camera_rig_z_m",
        "key_diffuse_rgb",
        "fill_diffuse_rgb",
    }
)
_PUBLIC_EXPERIMENT_FIELDS = frozenset(
    {
        "experiment_id",
        "experiment_version",
        "scene_sha256",
        "public_parameters",
        "hardware_status",
    }
)
_VISION2D_RESULT_FIELDS = frozenset(
    {
        "snapshot_id",
        "vision_bundle_path",
        "profile_id",
        "status",
        "image_size",
        "targets",
        "rejected_targets",
    }
)
_TEMPLATE_MATCH_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "snapshot_id",
        "vision_bundle_path",
        "template_id",
        "template_version",
        "matched",
        "status",
        "score",
        "threshold",
        "bbox_px",
        "center_px",
        "image_size",
        "search_roi_px",
        "method",
    }
)
_CODE_ROUTE_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "snapshot_id",
        "vision_bundle_path",
        "plan_id",
        "status",
        "safe_z_mm",
        "speed_mm_s",
        "entries",
    }
)
_OCR_SORTING_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "snapshot_id",
        "vision_bundle_path",
        "manifest_sha256",
        "scene_id",
        "status",
        "training",
        "entries",
        "plan_id",
        "safe_z_mm",
        "speed_mm_s",
        "evidence",
        "hardware_status",
    }
)
_OCR_TRAINING_FIELDS = frozenset(
    {"train_count", "test_count", "held_out_accuracy", "seed"}
)
_OCR_ENTRY_FIELDS = frozenset(
    {"entry_id", "part_id", "identifier", "route_id", "roi_px", "confidence", "status"}
)
_OCR_EVIDENCE_FIELDS = frozenset({"raw_path", "annotated_path", "bundle_path"})
_OCR_RECEIPT_FIELDS = frozenset(
    {"schema_version", "entry_id", "status", "plan_id", "run_id", "evidence_id", "hardware_status"}
)
_ENTRY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z")


def _copy_json_native(value: Any, *, path: str) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise RuntimeError(
                f"PROTOCOL_RESPONSE_INVALID: {path} must be finite"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise RuntimeError(
                    f"PROTOCOL_RESPONSE_INVALID: {path} keys"
                )
            result[key] = _copy_json_native(
                nested,
                path=f"{path}.{key}",
            )
        return result
    if type(value) is list:
        return [
            _copy_json_native(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise RuntimeError(f"PROTOCOL_RESPONSE_INVALID: {path} JSON type")


class _StudentProgramCancelled(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class _Rpc:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self._ids = count(1)

    def call(self, name: str, **args: Any) -> Any:
        command_id = f"{next(self._ids):06d}"
        command = CommandMessage(command_id, name, args)
        self.connection.send(command.to_dict())
        response = ResponseMessage.from_dict(self.connection.recv())
        if response.command_id != command_id:
            raise RuntimeError("PROTOCOL_CORRELATION_ERROR")
        if response.status == "CANCELLED":
            assert response.error is not None
            raise _StudentProgramCancelled(
                code=str(
                    response.error.get(
                        "code",
                        "STUDENT_PROGRAM_CANCELLED",
                    )
                ),
                message=str(
                    response.error.get(
                        "message",
                        "学生程序通信已取消",
                    )
                ),
            )
        if response.status != "PASS":
            assert response.error is not None
            raise RuntimeError(
                f"{response.error.get('code', 'COMMAND_FAILED')}: "
                f"{response.error.get('message', '命令失败')}"
            )
        return response.value


class StudentRobot:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def home(self) -> None:
        self._rpc.call("robot.home")

    def move_world(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        *,
        speed: float,
    ) -> None:
        self._rpc.call(
            "robot.move_world",
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            speed=speed,
        )

    def pose(self) -> tuple[float, float, float]:
        value = self._rpc.call("robot.pose")
        if (
            isinstance(value, (str, bytes, bytearray))
            or not isinstance(value, Sequence)
            or len(value) != 3
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: robot.pose")
        try:
            pose = (float(value[0]), float(value[1]), float(value[2]))
        except Exception as error:
            raise RuntimeError(
                "PROTOCOL_RESPONSE_INVALID: robot.pose"
            ) from error
        if not all(isfinite(component) for component in pose):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: robot.pose")
        return pose


class StudentTool:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def on(self) -> None:
        self._rpc.call("tool.on")

    def off(self) -> None:
        self._rpc.call("tool.off")


@dataclass(frozen=True)
class StudentFrame:
    snapshot_id: str
    image_bgr: np.ndarray
    width: int
    height: int
    source: str
    sequence_id: int


@dataclass(frozen=True)
class StudentVisionProfile:
    profile_id: str
    resolution: tuple[int, int]
    perspective_angle_deg: float
    camera_rig_z_m: float
    key_diffuse_rgb: tuple[float, float, float]
    fill_diffuse_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class StudentVision2DResult:
    snapshot_id: str
    vision_bundle_path: str
    profile_id: str
    status: str
    image_size: tuple[int, int]
    targets: tuple[Mapping[str, Any], ...]
    rejected_targets: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class StudentTemplateMatchResult:
    schema_version: int
    snapshot_id: str
    vision_bundle_path: str
    template_id: str
    template_version: str
    matched: bool
    status: str
    score: float
    threshold: float
    bbox_px: tuple[int, int, int, int] | None
    center_px: tuple[float, float] | None
    image_size: tuple[int, int]
    search_roi_px: tuple[int, int, int, int]
    method: str


@dataclass(frozen=True)
class StudentCodeRouteEntry:
    entry_id: str
    part_id: str
    code_type: str
    payload: str
    route_id: str
    pick_xyz_mm: tuple[float, float, float]
    drop_xyz_mm: tuple[float, float, float]
    confidence: float


@dataclass(frozen=True)
class CodeRoutePlanResult:
    schema_version: int
    snapshot_id: str
    vision_bundle_path: str
    plan_id: str
    status: str
    safe_z_mm: float
    speed_mm_s: float
    entries: tuple[StudentCodeRouteEntry, ...]


@dataclass(frozen=True)
class StudentOcrTrainingReport:
    train_count: int
    test_count: int
    held_out_accuracy: float
    seed: int


@dataclass(frozen=True)
class StudentOcrEntry:
    entry_id: str
    part_id: str
    identifier: str
    route_id: str
    roi_px: tuple[int, int, int, int]
    confidence: float
    status: str


@dataclass(frozen=True)
class StudentOcrSortingResult:
    schema_version: int
    snapshot_id: str
    vision_bundle_path: str
    manifest_sha256: str
    scene_id: str
    status: str
    training: StudentOcrTrainingReport
    entries: tuple[StudentOcrEntry, ...]
    plan_id: str
    safe_z_mm: float
    speed_mm_s: float
    evidence: Mapping[str, Any]
    hardware_status: str

    @property
    def training_report(self) -> StudentOcrTrainingReport:
        return self.training

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "vision_bundle_path": self.vision_bundle_path,
            "manifest_sha256": self.manifest_sha256,
            "scene_id": self.scene_id,
            "status": self.status,
            "training": {
                "train_count": self.training.train_count,
                "test_count": self.training.test_count,
                "held_out_accuracy": self.training.held_out_accuracy,
                "seed": self.training.seed,
            },
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "part_id": entry.part_id,
                    "identifier": entry.identifier,
                    "route_id": entry.route_id,
                    "roi_px": list(entry.roi_px),
                    "confidence": entry.confidence,
                    "status": entry.status,
                }
                for entry in self.entries
            ],
            "plan_id": self.plan_id,
            "safe_z_mm": self.safe_z_mm,
            "speed_mm_s": self.speed_mm_s,
            "evidence": _copy_json_native(dict(self.evidence), path="ocr.evidence"),
            "hardware_status": self.hardware_status,
        }


@dataclass(frozen=True)
class StudentOcrSortReceipt:
    schema_version: int
    entry_id: str
    status: str
    plan_id: str
    run_id: str
    evidence_id: str
    hardware_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entry_id": self.entry_id,
            "status": self.status,
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "evidence_id": self.evidence_id,
            "hardware_status": self.hardware_status,
        }


def _freeze_json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_native(nested) for key, nested in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json_native(nested) for nested in value)
    return value


def _vision2d_items(value: Any, *, field: str) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list:
        raise RuntimeError(
            f"PROTOCOL_RESPONSE_INVALID: vision2d.analyze {field}"
        )
    items: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise RuntimeError(
                f"PROTOCOL_RESPONSE_INVALID: vision2d.analyze {field}"
            )
        copied = _copy_json_native(
            item,
            path=f"vision2d.analyze.{field}[{index}]",
        )
        if not isinstance(copied, dict):
            raise RuntimeError(
                f"PROTOCOL_RESPONSE_INVALID: vision2d.analyze {field}"
            )
        items.append(_freeze_json_native(copied))
    return tuple(items)


def _vision2d_result(value: Any) -> StudentVision2DResult:
    if not isinstance(value, Mapping):
        raise RuntimeError("PROTOCOL_RESPONSE_INVALID: vision2d.analyze")
    if set(value) != _VISION2D_RESULT_FIELDS:
        raise RuntimeError(
            "PROTOCOL_RESPONSE_INVALID: vision2d.analyze fields"
        )

    snapshot_id = value["snapshot_id"]
    if (
        type(snapshot_id) is not str
        or _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None
    ):
        raise RuntimeError("PROTOCOL_RESPONSE_INVALID: snapshot_id")

    bundle_path = value["vision_bundle_path"]
    if (
        type(bundle_path) is not str
        or len(bundle_path) > 80
        or _VISION_BUNDLE_NAME.fullmatch(bundle_path) is None
        or "/" in bundle_path
        or "\\" in bundle_path
    ):
        raise RuntimeError(
            "PROTOCOL_RESPONSE_INVALID: vision2d.analyze bundle"
        )

    profile_id = value["profile_id"]
    if (
        type(profile_id) is not str
        or _PROFILE_ID.fullmatch(profile_id) is None
    ):
        raise RuntimeError("PROTOCOL_RESPONSE_INVALID: profile_id")

    status = value["status"]
    if type(status) is not str or status not in {
        "PASS",
        "PARTIAL",
        "NO_TARGETS",
        "REJECTED",
    }:
        raise RuntimeError("PROTOCOL_RESPONSE_INVALID: vision2d.analyze status")

    image_size = value["image_size"]
    if (
        type(image_size) is not list
        or len(image_size) != 2
        or any(type(component) is not int for component in image_size)
        or any(component <= 0 or component > 4096 for component in image_size)
    ):
        raise RuntimeError(
            "PROTOCOL_RESPONSE_INVALID: vision2d.analyze image_size"
        )

    return StudentVision2DResult(
        snapshot_id=snapshot_id,
        vision_bundle_path=bundle_path,
        profile_id=profile_id,
        status=status,
        image_size=(image_size[0], image_size[1]),
        targets=_vision2d_items(value["targets"], field="targets"),
        rejected_targets=_vision2d_items(
            value["rejected_targets"], field="rejected_targets"
        ),
    )


def _template_match_error(field: str) -> RuntimeError:
    return RuntimeError(f"PROTOCOL_RESPONSE_INVALID: vision2d.template_match {field}")


def _template_number(value: Any, field: str) -> float:
    if type(value) not in {int, float}:
        raise _template_match_error(field)
    result = float(value)
    if not isfinite(result) or not -1.0 <= result <= 1.0:
        raise _template_match_error(field)
    return result


def _template_ints(value: Any, field: str, *, count: int) -> tuple[int, ...]:
    if (
        type(value) is not list
        or len(value) != count
        or any(type(item) is not int for item in value)
    ):
        raise _template_match_error(field)
    return tuple(value)


def _template_match_result(value: Any) -> StudentTemplateMatchResult:
    if not isinstance(value, Mapping) or set(value) != _TEMPLATE_MATCH_RESULT_FIELDS:
        raise _template_match_error("fields")
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise _template_match_error("schema_version")
    snapshot_id = value["snapshot_id"]
    if type(snapshot_id) is not str or _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None:
        raise _template_match_error("snapshot_id")
    bundle_path = value["vision_bundle_path"]
    if (
        type(bundle_path) is not str
        or len(bundle_path) > 80
        or _VISION_BUNDLE_NAME.fullmatch(bundle_path) is None
        or "/" in bundle_path
        or "\\" in bundle_path
    ):
        raise _template_match_error("vision_bundle_path")
    template_id = value["template_id"]
    if type(template_id) is not str or _TEMPLATE_ID_PATTERN.fullmatch(template_id) is None:
        raise _template_match_error("template_id")
    template_version = value["template_version"]
    if (
        type(template_version) is not str
        or _TEMPLATE_VERSION_PATTERN.fullmatch(template_version) is None
    ):
        raise _template_match_error("template_version")
    matched = value["matched"]
    if type(matched) is not bool:
        raise _template_match_error("matched")
    status = value["status"]
    if type(status) is not str or status not in {"MATCHED", "NOT_MATCHED"}:
        raise _template_match_error("status")
    if matched != (status == "MATCHED"):
        raise _template_match_error("status")
    score = _template_number(value["score"], "score")
    threshold = _template_number(value["threshold"], "threshold")
    if matched != (score >= threshold):
        raise _template_match_error("matched")
    bbox_value = value["bbox_px"]
    bbox: tuple[int, int, int, int] | None
    if bbox_value is None:
        bbox = None
    else:
        parsed_bbox = _template_ints(bbox_value, "bbox_px", count=4)
        if parsed_bbox[0] < 0 or parsed_bbox[1] < 0 or parsed_bbox[2] <= 0 or parsed_bbox[3] <= 0:
            raise _template_match_error("bbox_px")
        bbox = (parsed_bbox[0], parsed_bbox[1], parsed_bbox[2], parsed_bbox[3])
    center_value = value["center_px"]
    center: tuple[float, float] | None
    if center_value is None:
        center = None
    else:
        if type(center_value) is not list or len(center_value) != 2:
            raise _template_match_error("center_px")
        if any(type(item) not in {int, float} or not isfinite(float(item)) for item in center_value):
            raise _template_match_error("center_px")
        center = (float(center_value[0]), float(center_value[1]))
    image_size = _template_ints(value["image_size"], "image_size", count=2)
    if any(component <= 0 or component > 4096 for component in image_size):
        raise _template_match_error("image_size")
    search_roi = _template_ints(value["search_roi_px"], "search_roi_px", count=4)
    if (
        search_roi[0] < 0
        or search_roi[1] < 0
        or search_roi[2] <= 0
        or search_roi[3] <= 0
        or search_roi[0] + search_roi[2] > image_size[0]
        or search_roi[1] + search_roi[3] > image_size[1]
    ):
        raise _template_match_error("search_roi_px")
    if bbox is not None and (
        bbox[0] + bbox[2] > image_size[0] or bbox[1] + bbox[3] > image_size[1]
    ):
        raise _template_match_error("bbox_px")
    if bbox is not None and (
        bbox[0] < search_roi[0]
        or bbox[1] < search_roi[1]
        or bbox[0] + bbox[2] > search_roi[0] + search_roi[2]
        or bbox[1] + bbox[3] > search_roi[1] + search_roi[3]
    ):
        raise _template_match_error("bbox_px")
    method = value["method"]
    if method != "TM_CCOEFF_NORMED":
        raise _template_match_error("method")
    if (bbox is None) != (center is None):
        raise _template_match_error("center_px")
    return StudentTemplateMatchResult(
        schema_version=schema_version,
        snapshot_id=snapshot_id,
        vision_bundle_path=bundle_path,
        template_id=template_id,
        template_version=template_version,
        matched=matched,
        status=status,
        score=score,
        threshold=threshold,
        bbox_px=bbox,
        center_px=center,
        image_size=(image_size[0], image_size[1]),
        search_roi_px=(search_roi[0], search_roi[1], search_roi[2], search_roi[3]),
        method=method,
    )


def _route_error(field: str) -> RuntimeError:
    return RuntimeError(f"PROTOCOL_RESPONSE_INVALID: vision2d.code_routes {field}")


def _route_number(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not isfinite(float(value)):
        raise _route_error(field)
    return float(value)


def _route_vector(value: Any, field: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise _route_error(field)
    return (
        _route_number(value[0], field),
        _route_number(value[1], field),
        _route_number(value[2], field),
    )


def _code_route_result(value: Any) -> CodeRoutePlanResult:
    if not isinstance(value, Mapping) or set(value) != _CODE_ROUTE_RESULT_FIELDS:
        raise _route_error("fields")
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise _route_error("schema_version")
    snapshot_id = value["snapshot_id"]
    if type(snapshot_id) is not str or _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None:
        raise _route_error("snapshot_id")
    bundle = value["vision_bundle_path"]
    if (
        type(bundle) is not str
        or len(bundle) > 80
        or _VISION_BUNDLE_NAME.fullmatch(bundle) is None
        or "/" in bundle
        or "\\" in bundle
    ):
        raise _route_error("vision_bundle_path")
    plan_id = value["plan_id"]
    if (
        type(plan_id) is not str
        or len(plan_id) != 64
        or any(character not in "0123456789abcdef" for character in plan_id)
    ):
        raise _route_error("plan_id")
    if value["status"] != "PASS":
        raise _route_error("status")
    entries_raw = value["entries"]
    if type(entries_raw) is not list or len(entries_raw) != 4:
        raise _route_error("entries")
    entry_fields = {
        "entry_id", "part_id", "code_type", "payload", "route_id",
        "pick_xyz_mm", "drop_xyz_mm", "confidence",
    }
    identifier_pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z")
    entries: list[StudentCodeRouteEntry] = []
    for index, raw in enumerate(entries_raw):
        if not isinstance(raw, Mapping) or set(raw) != entry_fields:
            raise _route_error(f"entries[{index}].fields")
        for name in ("entry_id", "part_id", "payload", "route_id"):
            if type(raw[name]) is not str or identifier_pattern.fullmatch(raw[name]) is None:
                raise _route_error(name)
        if type(raw["code_type"]) is not str or raw["code_type"] not in {"qr", "ean13"}:
            raise _route_error("code_type")
        confidence = _route_number(raw["confidence"], "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise _route_error("confidence")
        entries.append(
            StudentCodeRouteEntry(
                raw["entry_id"], raw["part_id"], raw["code_type"], raw["payload"],
                raw["route_id"], _route_vector(raw["pick_xyz_mm"], "pick_xyz_mm"),
                _route_vector(raw["drop_xyz_mm"], "drop_xyz_mm"), confidence,
            )
        )
    safe_z = _route_number(value["safe_z_mm"], "safe_z_mm")
    speed = _route_number(value["speed_mm_s"], "speed_mm_s")
    if (
        len({entry.entry_id for entry in entries}) != 4
        or len({entry.part_id for entry in entries}) != 4
        or len({(entry.code_type, entry.payload) for entry in entries}) != 4
        or len({entry.drop_xyz_mm for entry in entries}) != 4
        or speed <= 0.0
        or safe_z <= max(
            coordinate
            for entry in entries
            for coordinate in (entry.pick_xyz_mm[2], entry.drop_xyz_mm[2])
        )
    ):
        raise _route_error("entries")
    return CodeRoutePlanResult(
        1, snapshot_id, bundle, plan_id, "PASS", safe_z, speed, tuple(entries)
    )


def _ocr_error(field: str) -> RuntimeError:
    return RuntimeError(f"PROTOCOL_RESPONSE_INVALID: vision2d.ocr_sorting {field}")


def _ocr_identifier(value: Any, field: str) -> str:
    if type(value) is not str or _ENTRY_ID_PATTERN.fullmatch(value) is None:
        raise _ocr_error(field)
    return value


def _ocr_sha(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _ocr_error(field)
    return value


def _ocr_finite(value: Any, field: str) -> float:
    if type(value) not in {int, float}:
        raise _ocr_error(field)
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise _ocr_error(field) from None
    if not isfinite(normalized):
        raise _ocr_error(field)
    return normalized


def _ocr_roi(value: Any, field: str) -> tuple[int, int, int, int]:
    if (
        type(value) is not list
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise _ocr_error(field)
    x, y, width, height = value
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 4096
        or y + height > 4096
    ):
        raise _ocr_error(field)
    return x, y, width, height


def _ocr_relative_path(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 240
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise _ocr_error(field)
    return value


def _ocr_sorting_result(value: Any) -> StudentOcrSortingResult:
    if not isinstance(value, Mapping) or set(value) != _OCR_SORTING_RESULT_FIELDS:
        raise _ocr_error("fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise _ocr_error("schema_version")
    snapshot_id = value["snapshot_id"]
    if type(snapshot_id) is not str or _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None:
        raise _ocr_error("snapshot_id")
    bundle_path = value["vision_bundle_path"]
    if (
        type(bundle_path) is not str
        or len(bundle_path) > 80
        or _VISION_BUNDLE_NAME.fullmatch(bundle_path) is None
        or "/" in bundle_path
        or "\\" in bundle_path
    ):
        raise _ocr_error("vision_bundle_path")
    manifest_sha = _ocr_sha(value["manifest_sha256"], "manifest_sha256")
    scene_id = value["scene_id"]
    if type(scene_id) is not str or scene_id != "V1-08":
        raise _ocr_error("scene_id")
    if value["status"] != "PASS":
        raise _ocr_error("status")
    training = value["training"]
    if not isinstance(training, Mapping) or set(training) != _OCR_TRAINING_FIELDS:
        raise _ocr_error("training.fields")
    for field in ("train_count", "test_count", "seed"):
        if type(training[field]) is not int or training[field] <= 0:
            raise _ocr_error(f"training.{field}")
    accuracy = _ocr_finite(training["held_out_accuracy"], "training.held_out_accuracy")
    if not 0.0 <= accuracy <= 1.0:
        raise _ocr_error("training.held_out_accuracy")
    entries_raw = value["entries"]
    if type(entries_raw) is not list or len(entries_raw) != 4:
        raise _ocr_error("entries")
    entries: list[StudentOcrEntry] = []
    for index, raw in enumerate(entries_raw):
        if not isinstance(raw, Mapping) or set(raw) != _OCR_ENTRY_FIELDS:
            raise _ocr_error(f"entries[{index}].fields")
        entry_id = _ocr_identifier(raw["entry_id"], f"entries[{index}].entry_id")
        part_id = _ocr_identifier(raw["part_id"], f"entries[{index}].part_id")
        identifier = _ocr_identifier(raw["identifier"], f"entries[{index}].identifier")
        route_id = _ocr_identifier(raw["route_id"], f"entries[{index}].route_id")
        roi = _ocr_roi(raw["roi_px"], f"entries[{index}].roi_px")
        confidence = _ocr_finite(raw["confidence"], f"entries[{index}].confidence")
        if not 0.40 <= confidence <= 1.0 or raw["status"] != "APPROVED":
            raise _ocr_error(f"entries[{index}]")
        entries.append(StudentOcrEntry(entry_id, part_id, identifier, route_id, roi, confidence, "APPROVED"))
    if (
        tuple(entry.entry_id for entry in entries) != ("entry_a", "entry_b", "entry_c", "entry_d")
        or tuple(entry.part_id for entry in entries) != ("part_a", "part_b", "part_c", "part_d")
        or tuple(entry.identifier for entry in entries) != ("A1", "A2", "B1", "B2")
        or tuple(entry.route_id for entry in entries) != ("route_alpha", "route_alpha", "route_beta", "route_beta")
    ):
        raise _ocr_error("entries.whitelist")
    plan_id = _ocr_sha(value["plan_id"], "plan_id")
    safe_z = _ocr_finite(value["safe_z_mm"], "safe_z_mm")
    speed = _ocr_finite(value["speed_mm_s"], "speed_mm_s")
    if safe_z <= 0.0 or speed <= 0.0:
        raise _ocr_error("plan")
    evidence = value["evidence"]
    if not isinstance(evidence, Mapping) or set(evidence) != _OCR_EVIDENCE_FIELDS:
        raise _ocr_error("evidence.fields")
    copied_evidence: dict[str, Any] = {}
    for field in _OCR_EVIDENCE_FIELDS:
        copied_evidence[field] = _ocr_relative_path(evidence[field], f"evidence.{field}")
    hardware_status = value["hardware_status"]
    if hardware_status != "PENDING_HARDWARE":
        raise _ocr_error("hardware_status")
    return StudentOcrSortingResult(
        schema_version=1,
        snapshot_id=snapshot_id,
        vision_bundle_path=bundle_path,
        manifest_sha256=manifest_sha,
        scene_id=scene_id,
        status="PASS",
        training=StudentOcrTrainingReport(
            training["train_count"], training["test_count"], accuracy, training["seed"]
        ),
        entries=tuple(entries),
        plan_id=plan_id,
        safe_z_mm=safe_z,
        speed_mm_s=speed,
        evidence=_freeze_json_native(copied_evidence),
        hardware_status=hardware_status,
    )


def _ocr_sort_receipt(value: Any) -> StudentOcrSortReceipt:
    if not isinstance(value, Mapping) or set(value) != _OCR_RECEIPT_FIELDS:
        raise RuntimeError("PROTOCOL_RESPONSE_INVALID: vision2d.ocr_sort_entry fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise RuntimeError("PROTOCOL_RESPONSE_INVALID: vision2d.ocr_sort_entry schema_version")
    entry_id = _ocr_identifier(value["entry_id"], "ocr_sort_entry.entry_id")
    if value["status"] != "COMPLETED":
        raise RuntimeError("PROTOCOL_RESPONSE_INVALID: vision2d.ocr_sort_entry status")
    plan_id = _ocr_sha(value["plan_id"], "ocr_sort_entry.plan_id")
    run_id = value["run_id"]
    evidence_id = value["evidence_id"]
    if (
        type(run_id) is not str or _SNAPSHOT_ID_PATTERN.fullmatch(run_id) is None
        or type(evidence_id) is not str or _SNAPSHOT_ID_PATTERN.fullmatch(evidence_id) is None
    ):
        raise RuntimeError("PROTOCOL_RESPONSE_INVALID: vision2d.ocr_sort_entry evidence")
    if value["hardware_status"] != "PENDING_HARDWARE":
        raise RuntimeError("PROTOCOL_RESPONSE_INVALID: vision2d.ocr_sort_entry hardware_status")
    return StudentOcrSortReceipt(1, entry_id, "COMPLETED", plan_id, run_id, evidence_id, "PENDING_HARDWARE")


def _profile_error(field: str) -> RuntimeError:
    return RuntimeError(f"PROTOCOL_RESPONSE_INVALID: camera.profile {field}")


def _bounded_profile_number(
    value: Any,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    if type(value) not in {int, float}:
        raise _profile_error(field)
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise _profile_error(field) from error
    if not isfinite(result) or not minimum <= result <= maximum:
        raise _profile_error(field)
    return result


def _profile_rgb(value: Any, field: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise _profile_error(field)
    return (
        _bounded_profile_number(value[0], f"{field}[0]", 0.0, 1.0),
        _bounded_profile_number(value[1], f"{field}[1]", 0.0, 1.0),
        _bounded_profile_number(value[2], f"{field}[2]", 0.0, 1.0),
    )


def _vision_profile(value: Any) -> StudentVisionProfile:
    if not isinstance(value, Mapping) or set(value) != _VISION_PROFILE_FIELDS:
        raise _profile_error("fields")

    profile_id = value["profile_id"]
    if type(profile_id) is not str or _PROFILE_ID.fullmatch(profile_id) is None:
        raise _profile_error("profile_id")

    resolution = value["resolution"]
    if (
        type(resolution) is not list
        or len(resolution) != 2
        or type(resolution[0]) is not int
        or type(resolution[1]) is not int
        or not 128 <= resolution[0] <= 1024
        or not 128 <= resolution[1] <= 1024
    ):
        raise _profile_error("resolution")

    return StudentVisionProfile(
        profile_id=profile_id,
        resolution=(resolution[0], resolution[1]),
        perspective_angle_deg=_bounded_profile_number(
            value["perspective_angle_deg"],
            "perspective_angle_deg",
            20.0,
            90.0,
        ),
        camera_rig_z_m=_bounded_profile_number(
            value["camera_rig_z_m"],
            "camera_rig_z_m",
            0.50,
            0.90,
        ),
        key_diffuse_rgb=_profile_rgb(
            value["key_diffuse_rgb"],
            "key_diffuse_rgb",
        ),
        fill_diffuse_rgb=_profile_rgb(
            value["fill_diffuse_rgb"],
            "fill_diffuse_rgb",
        ),
    )


class StudentCamera:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def get_profile(self) -> StudentVisionProfile:
        return _vision_profile(self._rpc.call("camera.profile.get"))

    def apply_profile(self, profile_id: str) -> StudentVisionProfile:
        if (
            type(profile_id) is not str
            or _PROFILE_ID.fullmatch(profile_id) is None
        ):
            raise ValueError(
                "profile_id must be a published ASCII identifier"
            )
        return _vision_profile(
            self._rpc.call("camera.profile.apply", profile_id=profile_id)
        )

    def reset_profile(self) -> StudentVisionProfile:
        return _vision_profile(self._rpc.call("camera.profile.reset"))

    def capture(self) -> StudentFrame:
        value = self._rpc.call("camera.capture")
        if not isinstance(value, Mapping):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera.capture")

        snapshot_id = value.get("snapshot_id")
        if (
            type(snapshot_id) is not str
            or _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: snapshot_id")

        png_bytes = value.get("png_bytes")
        if (
            type(png_bytes) is not bytes
            or not png_bytes.startswith(_PNG_SIGNATURE)
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera.capture PNG")

        width_value = value.get("width")
        height_value = value.get("height")
        if (
            type(width_value) is not int
            or type(height_value) is not int
            or width_value <= 0
            or height_value <= 0
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera dimensions")

        source = value.get("source")
        if type(source) is not str or source != "coppeliasim":
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera source")
        sequence_id = value.get("sequence_id")
        if type(sequence_id) is not int or sequence_id < 0:
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera sequence")

        encoded = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if (
            image is None
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera.capture PNG")
        height, width = image.shape[:2]
        if width != width_value or height != height_value:
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera dimensions")

        immutable_pixels = image.tobytes(order="C")
        read_only_image = np.frombuffer(
            immutable_pixels,
            dtype=np.uint8,
        ).reshape((height, width, 3))
        return StudentFrame(
            snapshot_id=snapshot_id,
            image_bgr=read_only_image,
            width=width,
            height=height,
            source=source,
            sequence_id=sequence_id,
        )


class StudentExperiment:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def info(self) -> dict[str, Any]:
        value = self._rpc.call("experiment.info")
        if not isinstance(value, Mapping):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: experiment.info")
        if not set(value).issubset(_PUBLIC_EXPERIMENT_FIELDS):
            raise RuntimeError(
                "PROTOCOL_RESPONSE_INVALID: experiment.info public fields"
            )
        required = {
            "experiment_id",
            "public_parameters",
            "hardware_status",
        }
        if not required.issubset(value):
            raise RuntimeError(
                "PROTOCOL_RESPONSE_INVALID: experiment.info fields"
            )
        if (
            type(value.get("experiment_id")) is not str
            or not value["experiment_id"]
            or value.get("hardware_status") != "PENDING_HARDWARE"
            or not isinstance(value.get("public_parameters"), Mapping)
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: experiment.info")
        if "experiment_version" in value and (
            type(value["experiment_version"]) is not str
            or not value["experiment_version"]
        ):
            raise RuntimeError(
                "PROTOCOL_RESPONSE_INVALID: experiment_version"
            )
        if "scene_sha256" in value and (
            type(value["scene_sha256"]) is not str
            or len(value["scene_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value["scene_sha256"]
            )
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: scene_sha256")
        result = _copy_json_native(value, path="experiment.info")
        assert isinstance(result, dict)
        return result


class StudentVision2D:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def analyze(self) -> StudentVision2DResult:
        return _vision2d_result(self._rpc.call("vision2d.analyze"))

    def template_match(self) -> StudentTemplateMatchResult:
        return _template_match_result(self._rpc.call("vision2d.template_match"))

    def code_routes(self) -> CodeRoutePlanResult:
        return _code_route_result(self._rpc.call("vision2d.code_routes"))

    def ocr_sorting(self) -> StudentOcrSortingResult:
        return _ocr_sorting_result(self._rpc.call("vision2d.ocr_sorting"))

    def sort_ocr_entry(self, entry_id: str) -> StudentOcrSortReceipt:
        if type(entry_id) is not str or _ENTRY_ID_PATTERN.fullmatch(entry_id) is None:
            raise ValueError("entry_id must be a published ASCII identifier")
        return _ocr_sort_receipt(
            self._rpc.call("vision2d.ocr_sort_entry", entry_id=entry_id)
        )

    def match_template(self) -> StudentTemplateMatchResult:
        return self.template_match()


class StudentContext:
    def __init__(self, connection: Any) -> None:
        self._rpc = _Rpc(connection)
        self.robot = StudentRobot(self._rpc)
        self.tool = StudentTool(self._rpc)
        self.camera = StudentCamera(self._rpc)
        self.experiment = StudentExperiment(self._rpc)
        self.vision2d = StudentVision2D(self._rpc)

    def log(self, message: str) -> None:
        self._rpc.call("context.log", message=str(message))

    def sleep(self, seconds: float) -> None:
        self._rpc.call("context.sleep", seconds=seconds)

    def checkpoint(self, label: str) -> None:
        self._rpc.call("context.checkpoint", label=str(label))
