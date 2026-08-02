"""Pure host-owned OCR sorting plan contract for V1-08.

The adapter in this module deliberately operates on already validated data
objects.  It does not load an OCR model or an asset, capture a frame, connect
to CoppeliaSim, or issue robot/tool commands.  A caller supplies one frozen
analysis batch and the adapter returns one deterministic, immutable plan.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

from vision_platform.vision2d.ocr import OCRCharacter, OCRResult, TrainingReport


ROUTES: tuple[tuple[str, str, str, str], ...] = (
    ("entry_a", "part_a", "A1", "route_alpha"),
    ("entry_b", "part_b", "A2", "route_alpha"),
    ("entry_c", "part_c", "B1", "route_beta"),
    ("entry_d", "part_d", "B2", "route_beta"),
)

_EXPECTED_IDENTIFIERS = tuple(item[2] for item in ROUTES)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class OcrSortError(ValueError):
    """Stable fail-closed error raised by the OCR sorting contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _error(code: str, message: str) -> OcrSortError:
    return OcrSortError(code, message)


def _text(value: Any, name: str, code: str) -> str:
    if type(value) is not str or not value or len(value) > 80 or any(ord(ch) < 32 or 0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        raise _error(code, f"{name} is invalid")
    return value


def _finite(value: Any, name: str, code: str) -> float:
    if type(value) not in {int, float}:
        raise _error(code, f"{name} must be a finite built-in number")
    try:
        normalized = float(value)
    except OverflowError:
        raise _error(code, f"{name} must be a finite built-in number") from None
    if not math.isfinite(normalized):
        raise _error(code, f"{name} must be a finite built-in number")
    return normalized


def _integer(value: Any, name: str, code: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        raise _error(code, f"{name} must be a {'positive ' if positive else ''}built-in integer")
    return value


def _vector(value: Any, length: int, name: str, code: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise _error(code, f"{name} has invalid size")
    return tuple(_finite(item, f"{name}[{index}]", code) for index, item in enumerate(value))


def _integer_vector(value: Any, length: int, name: str, code: str, *, positive: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise _error(code, f"{name} has invalid size")
    return tuple(_integer(item, f"{name}[{index}]", code, positive=positive) for index, item in enumerate(value))


def _roi(value: Any, name: str, code: str) -> tuple[int, int, int, int]:
    x, y, width, height = _integer_vector(value, 4, name, code)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise _error(code, f"{name} must have a positive rectangle origin and size")
    return x, y, width, height


def _mapping(value: Any, fields: set[str], name: str, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _error(code, f"{name} schema is invalid")
    return value


def _image_size(value: Any, name: str, code: str) -> tuple[int, int]:
    width, height = _integer_vector(value, 2, name, code, positive=True)
    return width, height


@dataclass(frozen=True)
class OcrAssetSpec:
    """One hash-bound, generated OCR asset from the training manifest."""

    asset_id: str
    path: str
    sha256: str
    size_px: tuple[int, int]
    channels: int
    generator: str

    def __post_init__(self) -> None:
        code = "OCR_ASSET_INVALID"
        _text(self.asset_id, "asset_id", code)
        path = _text(self.path, "path", code)
        normalized_path = path.replace("\\", "/")
        if normalized_path.startswith("/") or ":" in normalized_path.split("/", 1)[0] or ".." in normalized_path.split("/"):
            raise _error(code, "path must be a relative asset path")
        if type(self.sha256) is not str or _HEX64.fullmatch(self.sha256) is None:
            raise _error(code, "sha256 is invalid")
        size = _integer_vector(self.size_px, 2, "size_px", code, positive=True)
        channels = _integer(self.channels, "channels", code, positive=True)
        _text(self.generator, "generator", code)
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "size_px", size)
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class OcrRouteSpec:
    """Host-owned mapping from an OCR identifier to a safe route."""

    entry_id: str
    part_id: str
    identifier: str
    route_id: str
    pick_xyz_mm: tuple[float, float, float]
    drop_xyz_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        code = "OCR_SORT_ROUTE_INVALID"
        for name in ("entry_id", "part_id", "identifier", "route_id"):
            _text(getattr(self, name), name, code)
        object.__setattr__(self, "pick_xyz_mm", _vector(self.pick_xyz_mm, 3, "pick_xyz_mm", code))
        object.__setattr__(self, "drop_xyz_mm", _vector(self.drop_xyz_mm, 3, "drop_xyz_mm", code))


@dataclass(frozen=True)
class OcrObservation:
    """One identifier ROI and its kernel OCR result."""

    identifier: str
    result: OCRResult
    roi_px: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _text(self.identifier, "identifier", "OCR_SORT_RESULT_INVALID")
        if not isinstance(self.result, OCRResult):
            raise _error("OCR_SORT_RESULT_INVALID", "result must be an OCRResult")
        object.__setattr__(self, "roi_px", _roi(self.roi_px, "roi_px", "OCR_SORT_RESULT_INVALID"))

    @property
    def text(self) -> str:
        """The recognized text, exposed without duplicating mutable state."""

        return self.result.text


@dataclass(frozen=True)
class OcrAnalysis:
    """Frozen OCR batch passed from the host service to the plan builder."""

    training_report: TrainingReport
    observations: tuple[OcrObservation, ...]
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.training_report, TrainingReport):
            raise _error("OCR_TRAINING_REPORT_INVALID", "training_report must be a TrainingReport")
        if type(self.observations) is not tuple or any(not isinstance(item, OcrObservation) for item in self.observations):
            raise _error("OCR_SORT_RESULT_INVALID", "observations must be a tuple of OcrObservation")
        object.__setattr__(self, "image_size", _image_size(self.image_size, "image_size", "OCR_SORT_RESULT_INVALID"))


@dataclass(frozen=True)
class ApprovedOcrSortEntry:
    """One approved immutable OCR pick-and-place entry."""

    entry_id: str
    part_id: str
    identifier: str
    route_id: str
    roi_px: tuple[int, int, int, int]
    pick_xyz_mm: tuple[float, float, float]
    drop_xyz_mm: tuple[float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        code = "OCR_SORT_ENTRY_INVALID"
        for name in ("entry_id", "part_id", "identifier", "route_id"):
            _text(getattr(self, name), name, code)
        object.__setattr__(self, "roi_px", _roi(self.roi_px, "roi_px", code))
        object.__setattr__(self, "pick_xyz_mm", _vector(self.pick_xyz_mm, 3, "pick_xyz_mm", code))
        object.__setattr__(self, "drop_xyz_mm", _vector(self.drop_xyz_mm, 3, "drop_xyz_mm", code))
        confidence = _finite(self.confidence, "confidence", code)
        if not 0.0 <= confidence <= 1.0:
            raise _error(code, "confidence is outside [0, 1]")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class OcrSortPlan:
    """A complete four-entry OCR sorting plan with deterministic status."""

    plan_id: str
    entries: tuple[ApprovedOcrSortEntry, ...]
    safe_z_mm: float
    speed_mm_s: float
    status: str = "PASS"

    def __post_init__(self) -> None:
        code = "OCR_SORT_PLAN_INVALID"
        if type(self.plan_id) is not str or _HEX64.fullmatch(self.plan_id) is None:
            raise _error(code, "plan_id is invalid")
        if type(self.entries) is not tuple or len(self.entries) != 4 or any(not isinstance(item, ApprovedOcrSortEntry) for item in self.entries):
            raise _error(code, "entries must contain exactly four approved entries")
        if self.status != "PASS":
            raise _error(code, "status must be PASS")
        safe_z = _finite(self.safe_z_mm, "safe_z_mm", code)
        speed = _finite(self.speed_mm_s, "speed_mm_s", code)
        if speed <= 0.0 or safe_z <= max(
            coordinate
            for item in self.entries
            for coordinate in (item.pick_xyz_mm[2], item.drop_xyz_mm[2])
        ):
            raise _error(code, "plan safety parameters are invalid")
        object.__setattr__(self, "safe_z_mm", safe_z)
        object.__setattr__(self, "speed_mm_s", speed)
        if len({item.entry_id for item in self.entries}) != 4 or len({item.part_id for item in self.entries}) != 4 or len({item.identifier for item in self.entries}) != 4 or len({item.drop_xyz_mm for item in self.entries}) != 4:
            raise _error(code, "entry, part, identifier, and drop slots must be unique")


def validate_training_report(report: TrainingReport, minimum_accuracy: float) -> TrainingReport:
    """Validate a kernel report and enforce the configured accuracy floor."""

    code = "OCR_TRAINING_REPORT_INVALID"
    if not isinstance(report, TrainingReport):
        raise _error(code, "report must be a TrainingReport")
    minimum = _finite(minimum_accuracy, "minimum_accuracy", code)
    if not 0.0 <= minimum <= 1.0:
        raise _error(code, "minimum_accuracy is outside [0, 1]")
    if type(report.train_count) is not int or report.train_count <= 0 or type(report.test_count) is not int or report.test_count <= 0 or type(report.seed) is not int:
        raise _error(code, "training report counts and seed are invalid")
    accuracy = _finite(report.held_out_accuracy, "held_out_accuracy", code)
    if not 0.0 <= accuracy <= 1.0 or accuracy < minimum:
        raise _error(code, "held_out_accuracy is below the configured minimum")
    return report


def _validate_workspace(config: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float]:
    code = "OCR_SORT_CONFIG_INVALID"
    workspace = _mapping(config["workspace"], {"x_mm", "y_mm", "z_mm", "safe_z_mm"}, "workspace", code)
    x_range = _vector(workspace["x_mm"], 2, "workspace.x_mm", code)
    y_range = _vector(workspace["y_mm"], 2, "workspace.y_mm", code)
    z_range = _vector(workspace["z_mm"], 2, "workspace.z_mm", code)
    workspace_safe = _finite(workspace["safe_z_mm"], "workspace.safe_z_mm", code)
    if not x_range[0] < x_range[1] or not y_range[0] < y_range[1] or not z_range[0] < z_range[1]:
        raise _error(code, "workspace ranges are invalid")
    if not z_range[0] <= workspace_safe <= z_range[1]:
        raise _error(code, "workspace.safe_z_mm is outside the workspace z range")
    return (x_range[0], x_range[1]), (y_range[0], y_range[1]), (z_range[0], z_range[1]), workspace_safe


def _validate_routes(config: Mapping[str, Any], workspace: tuple[tuple[float, float], tuple[float, float], tuple[float, float], float], safe_z: float) -> tuple[OcrRouteSpec, ...]:
    code = "OCR_SORT_CONFIG_INVALID"
    raw_routes = config["routes"]
    if not isinstance(raw_routes, (list, tuple)) or len(raw_routes) != 4:
        raise _error(code, "routes must contain exactly four entries")
    fields = {"entry_id", "part_id", "identifier", "route_id", "pick_xyz_mm", "drop_xyz_mm"}
    routes: list[OcrRouteSpec] = []
    seen_entries: set[str] = set()
    seen_parts: set[str] = set()
    seen_identifiers: set[str] = set()
    seen_drops: set[tuple[float, float, float]] = set()
    for raw in raw_routes:
        row = _mapping(raw, fields, "routes[]", code)
        route = OcrRouteSpec(
            entry_id=_text(row["entry_id"], "entry_id", code),
            part_id=_text(row["part_id"], "part_id", code),
            identifier=_text(row["identifier"], "identifier", code),
            route_id=_text(row["route_id"], "route_id", code),
            pick_xyz_mm=_vector(row["pick_xyz_mm"], 3, "pick_xyz_mm", code),
            drop_xyz_mm=_vector(row["drop_xyz_mm"], 3, "drop_xyz_mm", code),
        )
        key = (route.entry_id, route.part_id, route.identifier, route.route_id)
        if key not in ROUTES or route.entry_id in seen_entries or route.part_id in seen_parts or route.identifier in seen_identifiers or route.drop_xyz_mm in seen_drops:
            raise _error(code, "routes must match the fixed unique whitelist")
        seen_entries.add(route.entry_id)
        seen_parts.add(route.part_id)
        seen_identifiers.add(route.identifier)
        seen_drops.add(route.drop_xyz_mm)
        routes.append(route)
    if {route.identifier for route in routes} != set(_EXPECTED_IDENTIFIERS):
        raise _error(code, "route identifier set is incomplete")
    x_range, y_range, z_range, workspace_safe = workspace
    if safe_z != workspace_safe or safe_z <= max(max(route.pick_xyz_mm[2], route.drop_xyz_mm[2]) for route in routes):
        raise _error(code, "safe_z_mm is inconsistent or unsafe")
    for route in routes:
        for point in (route.pick_xyz_mm, route.drop_xyz_mm):
            if not (x_range[0] <= point[0] <= x_range[1] and y_range[0] <= point[1] <= y_range[1] and z_range[0] <= point[2] <= z_range[1]):
                raise _error(code, "route point is outside workspace")
    return tuple(routes)


def _validate_observation(observation: OcrObservation, *, expected_identifier: str, confidence_min: float, image_size: tuple[int, int]) -> float:
    code = "OCR_SORT_RESULT_INVALID"
    if not isinstance(observation, OcrObservation) or observation.identifier != expected_identifier:
        raise _error(code, "observation identifier is unknown, duplicate, or mismatched")
    result = observation.result
    if (
        not isinstance(result, OCRResult)
        or result.status != "PASS"
        or result.text != expected_identifier
        or len(result.text) != 2
        or result.failure_code is not None
        or type(result.schema_version) is not int
        or result.schema_version != 1
        or type(result.character_count) is not int
        or result.character_count != 2
        or type(result.characters) is not tuple
        or len(result.characters) != 2
        or type(result.threshold_method) is not str
        or not result.threshold_method
        or type(result.confidence_method) is not str
        or not result.confidence_method
    ):
        raise _error(code, "OCR result must be a complete two-character PASS result")
    result_width, result_height = _image_size(result.image_size, "result.image_size", code)
    processing_ms = _finite(result.processing_ms, "processing_ms", code)
    if processing_ms < 0.0:
        raise _error(code, "processing_ms must be non-negative")
    x, y, width, height = _roi(observation.roi_px, "roi_px", code)
    frame_width, frame_height = image_size
    if x + width > frame_width or y + height > frame_height:
        raise _error(code, "ROI is outside the frame")
    local_coordinates = (result_width, result_height) == (width, height)
    frame_coordinates = (result_width, result_height) == (frame_width, frame_height)
    if not local_coordinates and not frame_coordinates:
        raise _error(code, "result dimensions must describe the ROI or source frame")
    confidences: list[float] = []
    for index, character in enumerate(result.characters):
        if not isinstance(character, OCRCharacter) or character.character != expected_identifier[index]:
            raise _error(code, "OCR character sequence is invalid")
        bbox = character.bbox_px
        if type(bbox) is not tuple or len(bbox) != 4 or any(type(value) is not int for value in bbox):
            raise _error(code, "OCR character bounding box is invalid")
        bx, by, bw, bh = bbox
        if bx < 0 or by < 0 or bw <= 0 or bh <= 0 or bx + bw > result_width or by + bh > result_height:
            raise _error(code, "OCR character bounding box is outside the ROI")
        if frame_coordinates and (bx < x or by < y or bx + bw > x + width or by + bh > y + height):
            raise _error(code, "OCR character bounding box is outside the ROI")
        confidence = _finite(character.confidence, "character.confidence", code)
        if not 0.0 <= confidence <= 1.0 or confidence < confidence_min:
            raise _error(code, "OCR character confidence is below the configured minimum")
        confidences.append(confidence)
    return min(confidences)


def build_ocr_sort_plan(config: Mapping[str, Any], results: OcrAnalysis, *, scene_part_ids: Collection[str]) -> OcrSortPlan:
    """Build a deterministic four-entry OCR sort plan from pure data."""

    code = "OCR_SORT_CONFIG_INVALID"
    required = {"schema_version", "expected_count", "training_accuracy_min", "confidence_min", "safe_z_mm", "speed_mm_s", "workspace", "routes"}
    if not isinstance(config, Mapping) or set(config) != required:
        raise _error(code, "configuration schema is invalid")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or type(config["expected_count"]) is not int or config["expected_count"] != 4:
        raise _error(code, "schema_version and expected_count are invalid")
    training_min = _finite(config["training_accuracy_min"], "training_accuracy_min", code)
    confidence_min = _finite(config["confidence_min"], "confidence_min", code)
    if not 0.0 <= training_min <= 1.0 or not 0.0 <= confidence_min <= 1.0:
        raise _error(code, "accuracy/confidence thresholds are outside [0, 1]")
    safe_z = _finite(config["safe_z_mm"], "safe_z_mm", code)
    speed = _finite(config["speed_mm_s"], "speed_mm_s", code)
    if speed <= 0.0:
        raise _error(code, "speed_mm_s must be positive")
    workspace = _validate_workspace(config)
    routes = _validate_routes(config, workspace, safe_z)

    if isinstance(scene_part_ids, (str, bytes)) or not isinstance(scene_part_ids, Collection):
        raise _error(code, "scene_part_ids is invalid")
    scene_parts = tuple(scene_part_ids)
    if any(type(part) is not str for part in scene_parts) or len(scene_parts) != len(set(scene_parts)) or set(scene_parts) != {route.part_id for route in routes}:
        raise _error(code, "scene part set does not match the route whitelist")

    if not isinstance(results, OcrAnalysis):
        raise _error("OCR_SORT_RESULT_INVALID", "results must be an OcrAnalysis batch")
    validate_training_report(results.training_report, training_min)
    if type(results.observations) is not tuple or len(results.observations) != 4:
        raise _error("OCR_SORT_RESULT_INVALID", "analysis must contain exactly four observations")
    expected_by_identifier = {route.identifier: route for route in routes}
    seen: set[str] = set()
    approved: list[ApprovedOcrSortEntry] = []
    for observation in results.observations:
        if not isinstance(observation, OcrObservation) or observation.identifier in seen or observation.identifier not in expected_by_identifier:
            raise _error("OCR_SORT_RESULT_INVALID", "observations must be a complete unique identifier set")
        route = expected_by_identifier[observation.identifier]
        confidence = _validate_observation(
            observation,
            expected_identifier=route.identifier,
            confidence_min=confidence_min,
            image_size=results.image_size,
        )
        approved.append(
            ApprovedOcrSortEntry(
                entry_id=route.entry_id,
                part_id=route.part_id,
                identifier=route.identifier,
                route_id=route.route_id,
                roi_px=observation.roi_px,
                pick_xyz_mm=route.pick_xyz_mm,
                drop_xyz_mm=route.drop_xyz_mm,
                confidence=confidence,
            )
        )
        seen.add(observation.identifier)
    if seen != set(_EXPECTED_IDENTIFIERS):
        raise _error("OCR_SORT_RESULT_INVALID", "observations are missing an expected identifier")
    approved.sort(key=lambda item: _EXPECTED_IDENTIFIERS.index(item.identifier))
    canonical = {
        "entries": [
            {
                "entry_id": item.entry_id,
                "part_id": item.part_id,
                "identifier": item.identifier,
                "route_id": item.route_id,
                "roi_px": list(item.roi_px),
                "pick_xyz_mm": list(item.pick_xyz_mm),
                "drop_xyz_mm": list(item.drop_xyz_mm),
                "confidence": item.confidence,
            }
            for item in approved
        ],
        "safe_z_mm": safe_z,
        "speed_mm_s": speed,
    }
    plan_id = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return OcrSortPlan(plan_id=plan_id, entries=tuple(approved), safe_z_mm=safe_z, speed_mm_s=speed, status="PASS")


def ocr_sort_plan_to_dict(plan: OcrSortPlan) -> dict[str, Any]:
    """Return a fresh JSON-native copy of an approved plan."""

    if not isinstance(plan, OcrSortPlan):
        raise _error("OCR_SORT_PLAN_INVALID", "plan type is invalid")
    payload = {
        "plan_id": str(plan.plan_id),
        "entries": [
            {
                "entry_id": str(entry.entry_id),
                "part_id": str(entry.part_id),
                "identifier": str(entry.identifier),
                "route_id": str(entry.route_id),
                "roi_px": [int(value) for value in entry.roi_px],
                "pick_xyz_mm": [float(value) for value in entry.pick_xyz_mm],
                "drop_xyz_mm": [float(value) for value in entry.drop_xyz_mm],
                "confidence": float(entry.confidence),
            }
            for entry in plan.entries
        ],
        "safe_z_mm": float(plan.safe_z_mm),
        "speed_mm_s": float(plan.speed_mm_s),
        "status": plan.status,
    }
    return payload


__all__ = [
    "ROUTES",
    "ApprovedOcrSortEntry",
    "OcrAnalysis",
    "OcrAssetSpec",
    "OcrObservation",
    "OcrRouteSpec",
    "OcrSortError",
    "OcrSortPlan",
    "build_ocr_sort_plan",
    "ocr_sort_plan_to_dict",
    "validate_training_report",
]
