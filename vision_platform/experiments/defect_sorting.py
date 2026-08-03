"""Pure host-owned V1-09 defect decision and sorting contracts.

This module deliberately contains no image loading, device, CoppeliaSim, or
student-runtime code.  It accepts an already validated batch of kernel
results, converts the batch into one immutable six-entry plan, and provides
JSON-native serializers for evidence and receipts.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

from vision_platform.vision2d.defect_detection import DefectResult


APPROVED_DEFECTS = frozenset({"missing", "hole", "foreign", "broken", "dimension"})
EXPECTED_ENTRIES: tuple[tuple[str, str, str, str, str], ...] = (
    ("entry_a", "part_a", "qualified", "route_qualified", "slot_qualified"),
    ("entry_b", "part_b", "missing", "route_missing", "slot_missing"),
    ("entry_c", "part_c", "hole", "route_hole", "slot_hole"),
    ("entry_d", "part_d", "foreign", "route_foreign", "slot_foreign"),
    ("entry_e", "part_e", "broken", "route_broken", "slot_broken"),
    ("entry_f", "part_f", "dimension", "route_dimension", "slot_dimension"),
)

_EXPECTED_BY_ENTRY = {item[0]: item for item in EXPECTED_ENTRIES}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT = 128


class DefectSortError(ValueError):
    """Stable fail-closed error for the V1-09 pure contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _error(code: str, message: str) -> DefectSortError:
    return DefectSortError(code, message)


def _text(value: Any, name: str, code: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_TEXT:
        raise _error(code, f"{name} is invalid")
    if any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value):
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
        qualifier = "positive " if positive else ""
        raise _error(code, f"{name} must be a {qualifier}built-in integer")
    return value


def _vector(value: Any, length: int, name: str, code: str) -> tuple[float, ...]:
    if type(value) not in {tuple, list} or len(value) != length:
        raise _error(code, f"{name} has invalid size")
    return tuple(_finite(item, f"{name}[{index}]", code) for index, item in enumerate(value))


def _roi(value: Any, name: str, code: str) -> tuple[int, int, int, int]:
    if type(value) not in {tuple, list} or len(value) != 4:
        raise _error(code, f"{name} has invalid size")
    values = tuple(_integer(item, f"{name}[{index}]", code) for index, item in enumerate(value))
    x, y, width, height = values
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise _error(code, f"{name} must be a positive in-frame rectangle")
    return values


def _digest(value: Any, name: str, code: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise _error(code, f"{name} must be a lowercase SHA-256 digest")
    return value


def _image_size(value: Any, name: str, code: str) -> tuple[int, int]:
    if type(value) not in {tuple, list} or len(value) != 2:
        raise _error(code, f"{name} has invalid size")
    width = _integer(value[0], f"{name}[0]", code, positive=True)
    height = _integer(value[1], f"{name}[1]", code, positive=True)
    return width, height


def _mapping(value: Any, fields: set[str], name: str, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _error(code, f"{name} schema is invalid")
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error("DEFECT_SORT_CONTRACT_INVALID", "payload is not canonical JSON") from exc


def _finding_payload(finding: Any) -> dict[str, Any]:
    return {
        "defect_type": str(finding.defect_type),
        "bbox_px": [int(value) for value in finding.bbox_px],
        "area_px2": float(finding.area_px2),
        "relative_area": float(finding.relative_area),
        "metric": float(finding.metric),
        "threshold": float(finding.threshold),
        "confidence": float(finding.confidence),
    }


def _findings_digest(result: DefectResult) -> str:
    payload = {
        "status": result.status,
        "failure_code": result.failure_code,
        "defects": [_finding_payload(finding) for finding in result.defects],
        "thresholds": {str(key): float(value) for key, value in sorted(result.thresholds.items())},
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class DefectObservation:
    """One fixed ROI and its already computed kernel result."""

    entry_id: str
    part_id: str
    result: DefectResult
    roi_px: tuple[int, int, int, int]
    reference_crop_sha256: str
    candidate_crop_sha256: str

    def __post_init__(self) -> None:
        code = "DEFECT_SORT_OBSERVATION_INVALID"
        _text(self.entry_id, "entry_id", code)
        _text(self.part_id, "part_id", code)
        if not isinstance(self.result, DefectResult):
            raise _error(code, "result must be a DefectResult")
        object.__setattr__(self, "roi_px", _roi(self.roi_px, "roi_px", code))
        object.__setattr__(self, "reference_crop_sha256", _digest(self.reference_crop_sha256, "reference_crop_sha256", code))
        object.__setattr__(self, "candidate_crop_sha256", _digest(self.candidate_crop_sha256, "candidate_crop_sha256", code))


@dataclass(frozen=True)
class ApprovedDefectSortEntry:
    """One immutable host-approved defect sorting route."""

    entry_id: str
    part_id: str
    decision: str
    route_id: str
    slot_id: str
    roi_px: tuple[int, int, int, int]
    pick_xyz_mm: tuple[float, float, float]
    drop_xyz_mm: tuple[float, float, float]
    reference_crop_sha256: str
    candidate_crop_sha256: str
    findings_sha256: str
    run_id: str
    frame_id: str
    scene_sha256: str
    config_sha256: str
    asset_manifest_sha256: str

    def __post_init__(self) -> None:
        code = "DEFECT_SORT_ENTRY_INVALID"
        for name in ("entry_id", "part_id", "decision", "route_id", "slot_id", "run_id", "frame_id"):
            _text(getattr(self, name), name, code)
        object.__setattr__(self, "roi_px", _roi(self.roi_px, "roi_px", code))
        object.__setattr__(self, "pick_xyz_mm", _vector(self.pick_xyz_mm, 3, "pick_xyz_mm", code))
        object.__setattr__(self, "drop_xyz_mm", _vector(self.drop_xyz_mm, 3, "drop_xyz_mm", code))
        for name in ("reference_crop_sha256", "candidate_crop_sha256", "findings_sha256", "scene_sha256", "config_sha256", "asset_manifest_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name, code))

    @property
    def defect_type(self) -> str | None:
        return None if self.decision == "qualified" else self.decision


@dataclass(frozen=True)
class DefectSortPlan:
    """Complete, immutable six-entry plan created from one frame."""

    plan_id: str
    entries: tuple[ApprovedDefectSortEntry, ...]
    run_id: str
    frame_id: str
    scene_sha256: str
    config_sha256: str
    asset_manifest_sha256: str
    image_size: tuple[int, int]
    safe_z_mm: float
    speed_mm_s: float
    status: str = "PASS"
    schema_version: int = 1

    def __post_init__(self) -> None:
        code = "DEFECT_SORT_PLAN_INVALID"
        _digest(self.plan_id, "plan_id", code)
        if type(self.entries) is not tuple or len(self.entries) != 6 or any(not isinstance(item, ApprovedDefectSortEntry) for item in self.entries):
            raise _error(code, "entries must contain exactly six approved entries")
        for name in ("run_id", "frame_id"):
            _text(getattr(self, name), name, code)
        for name in ("scene_sha256", "config_sha256", "asset_manifest_sha256"):
            _digest(getattr(self, name), name, code)
        object.__setattr__(self, "image_size", _image_size(self.image_size, "image_size", code))
        if self.status != "PASS" or type(self.schema_version) is not int or self.schema_version != 1:
            raise _error(code, "plan status/schema is invalid")
        safe_z = _finite(self.safe_z_mm, "safe_z_mm", code)
        speed = _finite(self.speed_mm_s, "speed_mm_s", code)
        if speed <= 0.0:
            raise _error(code, "speed_mm_s must be positive")
        object.__setattr__(self, "safe_z_mm", safe_z)
        object.__setattr__(self, "speed_mm_s", speed)
        expected_ids = tuple(item[0] for item in EXPECTED_ENTRIES)
        if tuple(item.entry_id for item in self.entries) != expected_ids:
            raise _error(code, "entry order or entry whitelist is invalid")
        if {item.part_id for item in self.entries} != {item[1] for item in EXPECTED_ENTRIES}:
            raise _error(code, "part whitelist is invalid")
        if {item.decision for item in self.entries} != {item[2] for item in EXPECTED_ENTRIES}:
            raise _error(code, "decision set must contain one qualified and five defect classes")
        for entry in self.entries:
            expected = _EXPECTED_BY_ENTRY[entry.entry_id]
            if (entry.part_id, entry.decision, entry.route_id, entry.slot_id) != (
                expected[1],
                expected[2],
                expected[3],
                expected[4],
            ):
                raise _error(code, "entry decision and route whitelist do not match")
            if (entry.run_id, entry.frame_id, entry.scene_sha256, entry.config_sha256, entry.asset_manifest_sha256) != (
                self.run_id,
                self.frame_id,
                self.scene_sha256,
                self.config_sha256,
                self.asset_manifest_sha256,
            ):
                raise _error(code, "entry evidence binding does not match plan")
            if safe_z <= max(entry.pick_xyz_mm[2], entry.drop_xyz_mm[2]):
                raise _error(code, "safe_z_mm is below a route point")


@dataclass(frozen=True)
class DefectSortReceipt:
    """One post-probe completion receipt; it is not a device command."""

    run_id: str
    plan_id: str
    entry_id: str
    part_id: str
    decision: str
    slot_id: str
    status: str
    evidence_id: str
    hardware_status: str
    scene_sha256: str | None = None
    evidence_sha256: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        code = "DEFECT_SORT_RECEIPT_INVALID"
        for name in ("run_id", "entry_id", "part_id", "decision", "slot_id", "status", "evidence_id", "hardware_status"):
            _text(getattr(self, name), name, code)
        _digest(self.plan_id, "plan_id", code)
        if self.scene_sha256 is not None:
            _digest(self.scene_sha256, "scene_sha256", code)
        if self.evidence_sha256 is not None:
            _digest(self.evidence_sha256, "evidence_sha256", code)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise _error(code, "schema_version is invalid")


def classify_defect_result(result: DefectResult) -> str:
    """Convert one kernel result into a single approved decision."""

    if not isinstance(result, DefectResult):
        raise _error("DEFECT_SORT_ANALYSIS_REJECTED", "kernel result type is invalid")
    if result.status == "PASS" and result.failure_code is None and not result.defects:
        return "qualified"
    if result.status != "PARTIAL" or result.failure_code != "DEFECTS_FOUND" or not result.defects:
        raise _error("DEFECT_SORT_ANALYSIS_REJECTED", "kernel result cannot activate a route")
    kinds = {finding.defect_type for finding in result.defects}
    if not kinds <= APPROVED_DEFECTS:
        raise _error("DEFECT_SORT_ANALYSIS_REJECTED", "unknown defect type")
    if len(kinds) != 1:
        raise _error("DEFECT_SORT_DECISION_AMBIGUOUS", "multiple defect classes")
    return next(iter(kinds))


def _validate_workspace(config: Mapping[str, Any], safe_z: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    code = "DEFECT_SORT_CONFIG_INVALID"
    workspace = _mapping(config["workspace"], {"x_mm", "y_mm", "z_mm", "safe_z_mm"}, "workspace", code)
    x_range = _vector(workspace["x_mm"], 2, "workspace.x_mm", code)
    y_range = _vector(workspace["y_mm"], 2, "workspace.y_mm", code)
    z_range = _vector(workspace["z_mm"], 2, "workspace.z_mm", code)
    workspace_safe = _finite(workspace["safe_z_mm"], "workspace.safe_z_mm", code)
    if not x_range[0] < x_range[1] or not y_range[0] < y_range[1] or not z_range[0] < z_range[1]:
        raise _error(code, "workspace ranges are invalid")
    if workspace_safe != safe_z or not z_range[0] <= safe_z <= z_range[1]:
        raise _error(code, "safe_z_mm is inconsistent or outside workspace")
    return (x_range[0], x_range[1]), (y_range[0], y_range[1]), (z_range[0], z_range[1])


def _validate_config(config: Mapping[str, Any]) -> tuple[tuple[dict[str, Any], ...], tuple[int, int], float, float]:
    code = "DEFECT_SORT_CONFIG_INVALID"
    required = {"schema_version", "expected_count", "image_size", "safe_z_mm", "speed_mm_s", "workspace", "routes"}
    if not isinstance(config, Mapping) or set(config) != required:
        raise _error(code, "configuration schema is invalid")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise _error(code, "schema_version is invalid")
    if type(config["expected_count"]) is not int or config["expected_count"] != 6:
        raise _error(code, "expected_count must be six")
    image_size = _image_size(config["image_size"], "image_size", code)
    if image_size != (1024, 1024):
        raise _error(code, "V1-09 image_size must be 1024x1024")
    safe_z = _finite(config["safe_z_mm"], "safe_z_mm", code)
    speed = _finite(config["speed_mm_s"], "speed_mm_s", code)
    if speed <= 0.0:
        raise _error(code, "speed_mm_s must be positive")
    workspace = _validate_workspace(config, safe_z)
    raw_routes = config["routes"]
    if type(raw_routes) not in {tuple, list} or len(raw_routes) != 6:
        raise _error(code, "routes must contain six entries")
    fields = {"entry_id", "part_id", "route_id", "slot_id", "pick_xyz_mm", "drop_xyz_mm"}
    routes: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_routes):
        row = _mapping(raw, fields, f"routes[{index}]", code)
        expected = EXPECTED_ENTRIES[index]
        if tuple(row[name] for name in ("entry_id", "part_id", "route_id", "slot_id")) != (expected[0], expected[1], expected[3], expected[4]):
            raise _error(code, "routes must match the fixed whitelist")
        pick = _vector(row["pick_xyz_mm"], 3, f"routes[{index}].pick_xyz_mm", code)
        drop = _vector(row["drop_xyz_mm"], 3, f"routes[{index}].drop_xyz_mm", code)
        for name, point in (("pick_xyz_mm", pick), ("drop_xyz_mm", drop)):
            if not (
                workspace[0][0] <= point[0] <= workspace[0][1]
                and workspace[1][0] <= point[1] <= workspace[1][1]
                and workspace[2][0] <= point[2] <= workspace[2][1]
            ):
                raise _error(code, f"routes[{index}].{name} is outside workspace")
        routes.append({"entry_id": expected[0], "part_id": expected[1], "route_id": expected[3], "slot_id": expected[4], "pick_xyz_mm": pick, "drop_xyz_mm": drop})
    return tuple(routes), image_size, safe_z, speed


def _validate_observation(observation: DefectObservation, expected: tuple[str, str, str, str, str], image_size: tuple[int, int]) -> str:
    code = "DEFECT_SORT_OBSERVATION_INVALID"
    if not isinstance(observation, DefectObservation) or (observation.entry_id, observation.part_id) != (expected[0], expected[1]):
        raise _error(code, "observation entry/part is unknown, duplicate, or mismatched")
    x, y, width, height = observation.roi_px
    if x + width > image_size[0] or y + height > image_size[1]:
        raise _error(code, "observation ROI is outside the frame")
    result = observation.result
    crop_size = (width, height)
    if result.image_size != crop_size:
        raise _error(code, "kernel image_size does not match the observation crop")
    for finding in result.defects:
        bx, by, bw, bh = finding.bbox_px
        if type(bx) is not int or type(by) is not int or type(bw) is not int or type(bh) is not int or bx < 0 or by < 0 or bw <= 0 or bh <= 0 or bx + bw > crop_size[0] or by + bh > crop_size[1]:
            raise _error(code, "finding bbox is outside the observation crop")
    return classify_defect_result(result)


def build_defect_sort_plan(
    config: Mapping[str, Any],
    observations: Collection[DefectObservation],
    *,
    run_id: str,
    frame_id: str,
    scene_sha256: str,
    config_sha256: str,
    asset_manifest_sha256: str,
) -> DefectSortPlan:
    """Build one deterministic plan from exactly six host observations."""

    code = "DEFECT_SORT_PLAN_INVALID"
    routes, image_size, safe_z, speed = _validate_config(config)
    _text(run_id, "run_id", code)
    _text(frame_id, "frame_id", code)
    _digest(scene_sha256, "scene_sha256", code)
    _digest(config_sha256, "config_sha256", code)
    _digest(asset_manifest_sha256, "asset_manifest_sha256", code)
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Collection):
        raise _error(code, "observations must be a collection")
    items = tuple(observations)
    if len(items) != 6:
        raise _error("DEFECT_SORT_PLAN_INCOMPLETE", "six observations are required")
    if len({getattr(item, "entry_id", None) for item in items}) != 6:
        raise _error(code, "observation entry IDs must be unique")
    if len({getattr(item, "part_id", None) for item in items}) != 6:
        raise _error(code, "observation part IDs must be unique")
    by_entry = {item.entry_id: item for item in items if isinstance(item, DefectObservation)}
    if set(by_entry) != {item[0] for item in EXPECTED_ENTRIES}:
        raise _error("DEFECT_SORT_PLAN_INCOMPLETE", "observation IDs do not cover the fixed batch")
    entries: list[ApprovedDefectSortEntry] = []
    decisions: list[str] = []
    candidate_digests: set[str] = set()
    for index, expected in enumerate(EXPECTED_ENTRIES):
        observation = by_entry[expected[0]]
        decision = _validate_observation(observation, expected, image_size)
        candidate_digests.add(observation.candidate_crop_sha256)
        route = routes[index]
        if (decision, route["route_id"], route["slot_id"]) != (expected[2], expected[3], expected[4]):
            raise _error("DEFECT_SORT_PLAN_INCOMPLETE", "decision does not match the fixed route whitelist")
        entry = ApprovedDefectSortEntry(
            entry_id=expected[0],
            part_id=expected[1],
            decision=decision,
            route_id=route["route_id"],
            slot_id=route["slot_id"],
            roi_px=observation.roi_px,
            pick_xyz_mm=route["pick_xyz_mm"],
            drop_xyz_mm=route["drop_xyz_mm"],
            reference_crop_sha256=observation.reference_crop_sha256,
            candidate_crop_sha256=observation.candidate_crop_sha256,
            findings_sha256=_findings_digest(observation.result),
            run_id=run_id,
            frame_id=frame_id,
            scene_sha256=scene_sha256,
            config_sha256=config_sha256,
            asset_manifest_sha256=asset_manifest_sha256,
        )
        entries.append(entry)
        decisions.append(decision)
    if len(candidate_digests) != 6:
        raise _error(code, "candidate crop digests must be unique")
    if set(decisions) != {item[2] for item in EXPECTED_ENTRIES}:
        raise _error("DEFECT_SORT_PLAN_INCOMPLETE", "decisions must contain one qualified and each defect class once")
    canonical = {
        "schema_version": 1,
        "entries": [
            {
                "entry_id": item.entry_id,
                "part_id": item.part_id,
                "decision": item.decision,
                "route_id": item.route_id,
                "slot_id": item.slot_id,
                "roi_px": list(item.roi_px),
                "pick_xyz_mm": list(item.pick_xyz_mm),
                "drop_xyz_mm": list(item.drop_xyz_mm),
                "reference_crop_sha256": item.reference_crop_sha256,
                "candidate_crop_sha256": item.candidate_crop_sha256,
                "findings_sha256": item.findings_sha256,
            }
            for item in entries
        ],
        "run_id": run_id,
        "frame_id": frame_id,
        "scene_sha256": scene_sha256,
        "config_sha256": config_sha256,
        "asset_manifest_sha256": asset_manifest_sha256,
        "image_size": list(image_size),
        "safe_z_mm": safe_z,
        "speed_mm_s": speed,
        "status": "PASS",
    }
    plan_id = hashlib.sha256(_canonical_bytes(canonical)).hexdigest()
    return DefectSortPlan(
        plan_id=plan_id,
        entries=tuple(entries),
        run_id=run_id,
        frame_id=frame_id,
        scene_sha256=scene_sha256,
        config_sha256=config_sha256,
        asset_manifest_sha256=asset_manifest_sha256,
        image_size=image_size,
        safe_z_mm=safe_z,
        speed_mm_s=speed,
    )


def defect_sort_plan_to_dict(plan: DefectSortPlan) -> dict[str, Any]:
    """Return a fresh JSON-native representation of an approved plan."""

    if not isinstance(plan, DefectSortPlan):
        raise _error("DEFECT_SORT_PLAN_INVALID", "plan type is invalid")
    return {
        "schema_version": int(plan.schema_version),
        "plan_id": str(plan.plan_id),
        "status": str(plan.status),
        "run_id": str(plan.run_id),
        "frame_id": str(plan.frame_id),
        "scene_sha256": str(plan.scene_sha256),
        "config_sha256": str(plan.config_sha256),
        "asset_manifest_sha256": str(plan.asset_manifest_sha256),
        "image_size": [int(value) for value in plan.image_size],
        "safe_z_mm": float(plan.safe_z_mm),
        "speed_mm_s": float(plan.speed_mm_s),
        "entries": [
            {
                "entry_id": str(entry.entry_id),
                "part_id": str(entry.part_id),
                "decision": str(entry.decision),
                "defect_type": entry.defect_type,
                "route_id": str(entry.route_id),
                "slot_id": str(entry.slot_id),
                "roi_px": [int(value) for value in entry.roi_px],
                "pick_xyz_mm": [float(value) for value in entry.pick_xyz_mm],
                "drop_xyz_mm": [float(value) for value in entry.drop_xyz_mm],
                "reference_crop_sha256": str(entry.reference_crop_sha256),
                "candidate_crop_sha256": str(entry.candidate_crop_sha256),
                "findings_sha256": str(entry.findings_sha256),
            }
            for entry in plan.entries
        ],
    }


def defect_sort_receipt_to_dict(receipt: DefectSortReceipt) -> dict[str, Any]:
    """Return a fresh JSON-native representation of a completion receipt."""

    if not isinstance(receipt, DefectSortReceipt):
        raise _error("DEFECT_SORT_RECEIPT_INVALID", "receipt type is invalid")
    payload: dict[str, Any] = {
        "schema_version": int(receipt.schema_version),
        "run_id": str(receipt.run_id),
        "plan_id": str(receipt.plan_id),
        "entry_id": str(receipt.entry_id),
        "part_id": str(receipt.part_id),
        "decision": str(receipt.decision),
        "defect_type": None if receipt.decision == "qualified" else str(receipt.decision),
        "slot_id": str(receipt.slot_id),
        "status": str(receipt.status),
        "evidence_id": str(receipt.evidence_id),
        "hardware_status": str(receipt.hardware_status),
    }
    if receipt.scene_sha256 is not None:
        payload["scene_sha256"] = str(receipt.scene_sha256)
    if receipt.evidence_sha256 is not None:
        payload["evidence_sha256"] = str(receipt.evidence_sha256)
    return payload


__all__ = [
    "APPROVED_DEFECTS",
    "EXPECTED_ENTRIES",
    "ApprovedDefectSortEntry",
    "DefectObservation",
    "DefectSortError",
    "DefectSortPlan",
    "DefectSortReceipt",
    "build_defect_sort_plan",
    "classify_defect_result",
    "defect_sort_plan_to_dict",
    "defect_sort_receipt_to_dict",
]
